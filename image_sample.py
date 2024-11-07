"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from functools import partial
import os
import time
import torch as th

from torch.utils.data import DataLoader

from configs.option_DPM_pansharpening import parser_args
from scipy.io import savemat
import einops
import numpy as np
import datetime
import torch.distributed as dist
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import spectral as spy
from improved_diffusion import logger
from utils.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from pancollection.common.psdata import PansharpeningSession as DataSession

rootPath = os.path.abspath(os.path.dirname(__file__))

# HWLOC_COMPONENTS=-gl python DPM/iddpm/scripts/image_sample.py 
"""
analysis_ref_batched_images('/Data2/YuZhong/AE_SSDiff_main/logs/samp_reduced_20_256_10-10-21-06.mat', 4, 0, 2047)
analysis_ref_batched_images('/Data2/YuZhong/AEM/DPM/iddpm/logs/usable_res/samp_reduced284_20_256_10-09-16-19.mat', 4, 0, 2047)

nohup python /Data2/YuZhong/AEM/DPM/iddpm/scripts/image_sample.py >> discuss-LotteryTH-04-22-sample.out &
fulldata: /Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv3_OrigScale_multiExm1.h5
""" 


def main(
    device='cuda:0',
    crop_batch_size=10,
    timestep_respacing="ddim10"
    ):


    args = parser_args()

    if device is not None:
        args.device = device
    th.cuda.set_device(args.device)
    
    if crop_batch_size is not None:
        args.crop_batch_size = crop_batch_size
    if timestep_respacing is not None:
        args.timestep_respacing = timestep_respacing
    

    logger.configure(dir='/'.join([rootPath, 'logs/train_logs/']))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(th.load(args.model_path, map_location=lambda storage, loc: storage.cuda()))
    model.cuda()
    model.eval()

    logger.log("sampling...")
    logger.log("model_path: ", args.model_path)
    all_images = []
    session = DataSession(args)
    data, _ = session.get_eval_dataloader(args.dataset['test'], False)    
    dl = iter(data)
    print("batch_size: ", args.crop_batch_size)


    data4gt = []
    # image_num = len(data)
    image_num = 20
    print("image_num:", image_num)
    gt_dim = 8
    
    tic = time.time()
    for i in range(image_num):
        batch = next(dl)
        pan_ori, lms_ori, ms_ori, gt = batch['pan'], batch['lms'], batch['ms'], batch['gt']
        gt =  einops.rearrange(gt, 'b k1 k2 c -> b c k1 k2', k1=256, k2=256)

        data4gt.append(gt[0])

        pan, lms, ms = map(lambda x: x.cuda(), (pan_ori, lms_ori, ms_ori))
        logger.log(f"test [{i}]/[{image_num}],  {args.timestep_respacing}", pan.shape, lms.shape, ms.shape)

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        kwargs_data = {"lms": lms, "pan": pan, "ms": ms}

        sample = sample_fn(
                    model,
                    shape=(args.crop_batch_size, gt_dim, args.image_size, args.image_size),
                    model_kwargs=kwargs_data,
                    clip_denoised=args.clip_denoised,
                    progress=False)

        sample_d =  einops.rearrange(sample, '1 c k1 k2 -> k1 k2 c', k1=256, k2=256)
        sample_d = sample_d.contiguous()  # sample[:, [4,2,0]]
        sample_d = (sample_d * 2047.).clamp(0, 2047)
        d = dict(  # [b, h, w, c], wv3 [0, 2047]
            sr=[sample_d.cpu().numpy()],
        )

        sample = sample.contiguous()  # sample[:, [4,2,0]]
        sample = (sample * 2047.).clamp(0, 2047)
        gathered_samples = [sample]
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        logger.log(f"created {len(all_images) * args.crop_batch_size} samples")

    print(time.time() - tic)
    print(len(all_images))
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    d = dict(  # [b, h, w, c], wv3 [0, 2047]
            gt=[sample.cpu().numpy()*2047 for sample in data4gt],
            sr=[sample for sample in arr],
        )

    
    loca=datetime.datetime.now().strftime('%m-%d-%H-%M')
    out_path = '/'.join([rootPath, f'logs/samp_reduced_{len(arr)}_256_{str(loca)}.mat'])
    
    savemat(out_path, d)
    logger.log(f"saving to {out_path}")
    print("save result")
    logger.log("sampling complete")
    
    return out_path


def loss_trend():
    import pandas as pd
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    data = pd.read_csv(r'DPM/iddpm/logs/train_log/log-2023-06-22-22-55-52-808391/progress.csv',sep=',',header=0)
    train_loss = list(data['loss'])
    step = list(data['step'])
    print(len(train_loss), len(step))

    log_dir = "/Data2/YuZhong/AEM/DPM/iddpm/logs/train_log/log-2023-06-22-22-55-52-808391"
    train_writer = SummaryWriter(log_dir=log_dir)
    for i in range(len(train_loss)):
        train_writer.add_scalar('Loss', train_loss[i], step[i])

def showmat(out_path=None):
    from matplotlib import pyplot as plt
    import scipy.io
    import cv2
    # matpath = "/Data2/YuZhong/AEM/DPM/iddpm/logs/usable_res/samples_reduced_20x256x256x8_624.mat"
    if out_path != None:
        matpath = out_path
    gt = scipy.io.loadmat(matpath)['gt']
    sr = scipy.io.loadmat(matpath)['sr']
    sr =  einops.rearrange(sr, 'b c k1 k2 -> b k1 k2 c', k1=256, k2=256)
    gt =  einops.rearrange(gt, 'b c k1 k2 -> b k1 k2 c', k1=256, k2=256)
    res = cv2.absdiff(gt,sr)
    for k in range(len(gt)):
        filepath = f'DPM/iddpm/logs/result/reduced_' + str(k) + '/'
        if not os.path.exists(filepath): 
            os.makedirs(filepath)
        plt.imshow(gt[k][:,:,[4,2,0]]/2047)
        plt.savefig(filepath + 'gt_' + str(k) + '_rgb_.png')

        plt.imshow(sr[k][:,:,[4,2,0]]/2047)
        plt.savefig(filepath + 'pred_' + str(k) +'_rgb_.png')

        plt.imshow(res[k][:,:,[4,2,0]]/2047)
        plt.savefig(filepath + 'res_' + str(k) +'_rgb_.png')


def showimg():
    args = parser_args()

    session = DataSession(args)
    data, _ = session.get_eval_dataloader(args.dataset['test'], False)    
    dl = iter(data)
    # image_num = len(data)
    image_num = 20
    print("image_num:", image_num)

    for i in range(image_num):
        plt.axis('off')  # 去掉坐标轴
        # plt.imshow(sample[0][:,:,(4, 2, 0)].cpu().numpy())
        plt.tight_layout()
        # model_kwargs = {}
        batch = next(dl)
        pan_ori, lms_ori, ms_ori, gt = batch['pan'], batch['lms'], batch['ms'], batch['gt']
        # gt =  einops.rearrange(gt, 'b c k1 k2 -> b k1 k2 c', k1=256, k2=256)
        pan_ori =  einops.rearrange(pan_ori, 'b c k1 k2 -> b k1 k2 c', k1=256, k2=256)
        lms_ori =  einops.rearrange(lms_ori, 'b c k1 k2 -> b k1 k2 c', k1=256, k2=256)
        ms_ori =  einops.rearrange(ms_ori, 'b c k1 k2 -> b k1 k2 c', k1=64, k2=64)
        filepath = f'DPM/iddpm/logs/result/pan_lms_' + str(i) + '/'
        if not os.path.exists(filepath): 
            os.makedirs(filepath)
        plt.imshow(gt[0][:,:,[4,2,0]])
        plt.savefig(filepath + 'gt_' + str(i) + '_rgb_.png')

        plt.imshow(pan_ori[0][:,:,:], cmap ='gray')
        plt.imsave(filepath + 'pan_' + str(i) +'_rgb_.png', pan_ori[0][:,:,:],  cmap ='gray')

        plt.imshow(ms_ori[0][:,:,[4,2,0]])
        plt.savefig(filepath + 'ms_' + str(i) +'_rgb_.png')
        
        plt.imshow(lms_ori[0][:,:,[4,2,0]])
        plt.savefig(filepath + 'lms_' + str(i) +'_rgb_.png')
        
if __name__ == "__main__":
    # loss_trend()
    out_path = main()
    # showmat(out_path)
    # showimg()
