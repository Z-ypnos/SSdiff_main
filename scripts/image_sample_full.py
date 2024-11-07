"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from functools import partial
import sys
import os
sys.path.append('/Data2/YuZhong/AEM/configs')
sys.path.append('/Data2/YuZhong/AEM')

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(2, "/Data2/YuZhong/AEM")
sys.path.insert(2, '/Data2/YuZhong/AEM/DPM/iddpm')
sys.path.insert(2, '/Data2/YuZhong/AEM/DPM/iddpm/improved_diffusion')
import torch as th
th.cuda.set_device('cuda:2')
# import argparse
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
from utils import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from pancollection.common.psdata import PansharpeningSession as DataSession



# HWLOC_COMPONENTS=-gl python DPM/iddpm/scripts/image_sample.py 
"""
analysis_ref_batched_images('/Data2/YuZhong/AEM/DPM/iddpm/logs/samp_reduced_1_256_09-02-15-05.mat', 4, 0, 2047)
analysis_unref_batched_images('/Data2/YuZhong/AEM/DPM/iddpm/logs/samp_full_2051209-13-22-27.mat', 4, 'WV3')
analysis_unref_batched_images('/Data2/YuZhong/AEM/DPM/iddpm/logs/samp_full_2051201-11-02-07.mat', 4, 'GF2')
analysis_unref_batched_images('/Data2/YuZhong/AEM/DPM/iddpm/logs/samp_full_2051211-11-11-20.mat', 4, 'WV2')
analysis_unref_batched_images('/Data2/YuZhong/AEM/DPM/iddpm/logs/samp_full_2051201-11-02-08.mat', 4, 'QB')



nohup python /Data2/YuZhong/AEM/DPM/iddpm/scripts/sample_full.py >> 10-13-wv3-u2-2e4-highpass-freeu-exp1-144-sample-full.out &
fulldata: /Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv3_OrigScale_multiExm1.h5
""" 



def main():

    args = parser_args()

    args.timestep_respacing = "ddim10"
    crop_batch_size = 40
    args.crop_batch_size = crop_batch_size
    args.dataset = {'train': 'wv3', 'val': 'wv3', 'test': 'test_wv3_OrigScale_multiExm1.h5'}  # full

    logger.configure(dir='/Data2/YuZhong/AEM/DPM/iddpm/logs/sample_log/')

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

    image_num = 20
    print("image_num:", image_num)
    gt_dim = 8
    list_pan=[]
    list_ms=[]
    list_lms=[]
    
    for i in range(image_num):

        batch = next(dl)

        pan_ori, lms_ori, ms_ori = batch['pan'], batch['lms'], batch['ms']
        list_pan.append(pan_ori[0])
        list_lms.append(lms_ori[0])
        list_ms.append(ms_ori[0])
        for j in range (1):

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

            sample = sample.contiguous()  # sample[:, [4,2,0]]
            sample = (sample * 2047.).clamp(0, 2047)

            sample_d =  einops.rearrange(sample, '1 c k1 k2 -> k1 k2 c', k1=512, k2=512)
            
            d = dict(  # [b, h, w, c], wv3 [0, 2047]
            sr=[sample_d.cpu().numpy()][0],
            )

            out_path = f"results/Full_WV3/output_mulExm_{i}.mat" 
            savemat(out_path, d)

            gathered_samples = [sample]
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])



    print(len(all_images))
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    d = dict(  # [b, h, w, c], wv3 [0, 2047]
            sr=[sample for sample in arr],
            ms=[sample.cpu().numpy()*2047 for sample in list_ms],
            lms=[sample.cpu().numpy()*2047 for sample in list_lms],
            pan=[sample.cpu().numpy()*2047 for sample in list_pan]
        )


    loca=datetime.datetime.now().strftime('%m-%d-%H-%M')
    out_path = f"/Data2/YuZhong/AEM/DPM/iddpm/logs/samp_full_{len(arr)}512{str(loca)}.mat" 
    savemat(out_path, d)
    logger.log(f"saving to {out_path}")
    print("save result")
    logger.log("sampling complete")
    return out_path


if __name__ == "__main__":
    out_path = None
    out_path = main()
