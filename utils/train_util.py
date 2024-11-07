import copy
import functools
import os

# import blobfile as bf
import os
import warnings

import numpy as np
import torch
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import datetime
from utils import dist_util
from improved_diffusion import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from model.nn import update_ema
from improved_diffusion.resample import LossAwareSampler, UniformSampler
# from improved_diffusion.metric import AnalysisPanAcc
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

# def sample_main(model):
#     from improved_diffusion.script_util import (
#     NUM_CLASSES,
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict,
#     )
#     from pancollection.common.psdata import PansharpeningSession as DataSession
#     import einops
#     from configs.option_DPM_pansharpening import parser_args

#     sample_args = parser_args()
#     sample_args.timestep_respacing = "ddim100"

#     _, sample_diffusion = create_model_and_diffusion(
#         **args_to_dict(sample_args, model_and_diffusion_defaults().keys())
#     )
#     all_images = []
#     sample_session = DataSession(sample_args)
#     sample_data, _ = sample_session.get_eval_dataloader(sample_args.dataset['test'], False)    
#     sample_dl = iter(sample_data)
#     sample_data4gt = []
#     for i in range(1):
#         sample_batch = next(sample_dl)
#         pan_ori, lms_ori, ms_ori, gt = sample_batch['pan'], sample_batch['lms'], sample_batch['ms'], sample_batch['gt']
#         gt =  einops.rearrange(gt, 'b k1 k2 c -> b c k1 k2', k1=256, k2=256)
#         sample_data4gt.append(gt[0])

#         for j in range (1):
            
#             pan, lms, ms = map(lambda x: x.to(dist_util.dev()), (pan_ori, lms_ori, ms_ori))
#             # model_kwargs.update(lms=lms, pan=pan, ms=ms)
#             sample_fn = (
#                 sample_diffusion.p_sample_loop if not sample_args.use_ddim else sample_diffusion.ddim_sample_loop
#             )

#             kwargs_data = {"lms": lms, "pan": pan, "ms": ms}
        
#             kwargs_sample = sample_fn(
#                         model,
#                         shape=(sample_args.crop_batch_size, 8, sample_args.image_size, sample_args.image_size),
#                         model_kwargs=kwargs_data,
#                         clip_denoised=sample_args.clip_denoised,
#                         progress=False)
            
#             # model.forward_chop -> val_step = forward + p_mean_variance -> forward_chop
            
#             kwargs_sample = kwargs_sample.contiguous()  # sample[:, [4,2,0]]
#             kwargs_sample = (kwargs_sample).clamp(0, 1)
#             gathered_samples = [kwargs_sample]
#             all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

#             logger.log(f"created {len(all_images) * sample_args.crop_batch_size} samples")

#     # print(len(all_images))
#     sample_arr = np.concatenate(all_images, axis=0)
#     sample_arr = sample_arr[: sample_args.num_samples]

#     d = dict(  # [b, h, w, c], wv3 [0, 2047]
#             gt=[sample.cpu().numpy() for sample in sample_data4gt],
#             sr=[sample for sample in sample_arr],
#         )
#     return d

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        device,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.device = device
        self.batch_size = batch_size
        self.microbatch = microbatch+1 if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        # self.global_batch = self.batch_size * dist.get_world_size()
        self.loca=datetime.datetime.now().strftime('%m-%d-%H-%M')
        
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = False
        self.ddp_model = self.model.to(self.device)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            cond = {}
            batch = next(iter(self.data), None)
            # print(batch['gt'].shape)
            if 'lms' in batch.keys() and 'ms' in batch.keys():
                cond.update(lms=batch['lms'], pan=batch['pan'], ms=batch['ms'])
                
            elif 'up' in batch.keys():
                cond.update(lms=batch['up'], pan=batch['rgb'])
            else:
                print(batch.keys(), cond.keys())
                raise KeyError

            self.run_step(batch, cond, self.batch_size)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond, total_size):
        self.forward_backward(batch, cond, total_size)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond, total_size):
        zero_grad(self.model_params)
        for i in range(0, total_size, self.microbatch):

            x = batch['gt'].to(self.device)
            micro = x[i : i + self.microbatch].to(self.device)      # gt
            micro_cond = {                                              # pan lms ms
                k: v[i : i + self.microbatch].to(self.device)
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= x.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)


            if last_batch or not self.use_ddp:
                losses = self.diffusion.training_losses(
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond)
                 
                
            else:
                with self.ddp_model.no_sync():
                    losses = self.diffusion.training_losses(
                            self.ddp_model,
                            micro,
                            t,
                            model_kwargs=micro_cond)

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()
            else:
                warnings.warn("p.grad is None")
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if not rate:
                filename = f"model{(self.step + self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
            
            os.makedirs("results/" + str(self.loca), exist_ok=True)
            th.save(state_dict, os.path.join("results/" + str(self.loca) + "/", filename))
            

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
