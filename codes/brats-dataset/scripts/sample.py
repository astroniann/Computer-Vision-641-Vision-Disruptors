"""
A script for sampling (inference) from a diffusion model for paired image-to-image translation.

Uses BraTS20Dataset (our custom loader) instead of the original BRATSVolumes,
so it works with the BraTS2024 folder structure:
    data_dir/
        BraTS2024-BraTS-GLI-ValidationData/
            validation_data/
                BraTS-GLI-00000-000/
                    BraTS-GLI-00000-000-t1n.nii.gz  ...

Default configuration: T1C agent — synthesises T1C from {T1N, T2W, T2F} conditions.
Checkpoint: runs/t1-c/brats_045000.pt

Usage (Windows / Git Bash):
    python scripts/sample.py ^
        --data_dir D:/user/BraTS2024-GLI ^
        --contr t1c ^
        --model_path runs/t1-c/brats_045000.pt ^
        --output_dir ./results/t1c/
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F

sys.path.append(".")

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D
from brats_dataset import get_dataloader


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())

    # Use our BraTS20Dataset — works with BraTS2024 folder layout on Windows
    datal = get_dataloader(
        data_root=args.data_dir,
        split="validation",
        batch_size=args.batch_size,
        num_workers=0,           # keep 0 on Windows to avoid multiprocessing issues
        dropout_modality=False,  # validation: all modalities present
    )

    model.eval()
    idwt = IDWT_3D("haar")
    dwt  = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for batch in iter(datal):
        batch['t1n'] = batch['t1n'].to(dist_util.dev())
        batch['t1c'] = batch['t1c'].to(dist_util.dev())
        batch['t2w'] = batch['t2w'].to(dist_util.dev())
        batch['t2f'] = batch['t2f'].to(dist_util.dev())

        # batch['subj'] is the full patient directory path; take just the patient folder name
        subj = pathlib.Path(batch['subj'][0]).name
        print(subj)

        if args.contr == 't1n':
            target = batch['t1n']
            cond_1 = batch['t1c']
            cond_2 = batch['t2w']
            cond_3 = batch['t2f']
        elif args.contr == 't1c':
            target = batch['t1c']
            cond_1 = batch['t1n']
            cond_2 = batch['t2w']
            cond_3 = batch['t2f']
        elif args.contr == 't2w':
            target = batch['t2w']
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2f']
        elif args.contr == 't2f':
            target = batch['t2f']
            cond_1 = batch['t1n']
            cond_2 = batch['t1c']
            cond_3 = batch['t2w']
        else:
            print("This contrast can't be synthesized.")
            continue

        # Conditioning vector — DWT each of the 3 condition modalities
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
        cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_2)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_3)
        cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        noise = th.randn(args.batch_size, 8, 96, 96, 96).to(dist_util.dev())

        sample = diffusion.p_sample_loop(
            model=model,
            shape=noise.shape,
            noise=noise,
            cond=cond,
            clip_denoised=args.clip_denoised,
            model_kwargs={},
        )

        B, _, H, W, D = sample.size()
        sample = idwt(
            sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
            sample[:, 1, :, :, :].view(B, 1, D, H, W),
            sample[:, 2, :, :, :].view(B, 1, D, H, W),
            sample[:, 3, :, :, :].view(B, 1, D, H, W),
            sample[:, 4, :, :, :].view(B, 1, D, H, W),
            sample[:, 5, :, :, :].view(B, 1, D, H, W),
            sample[:, 6, :, :, :].view(B, 1, D, H, W),
            sample[:, 7, :, :, :].view(B, 1, D, H, W),
        )

        sample[sample <= 0] = 0
        sample[sample >= 1] = 1

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)  # [B,1,H,W,D] -> [B,H,W,D]

        sample[cond_1.squeeze(1) == 0] = 0  # zero out non-brain voxels (both now [B,H,W,D])

        # Volumes are already 192^3 (padded/cropped by BraTS20Dataset).
        # No depth crop needed — we save the full 192-slice volume.

        if len(target.shape) == 5:
            target = target.squeeze(dim=1)

        out_dir = pathlib.Path(args.output_dir) / subj
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(sample.shape[0]):
            out_sample = str(out_dir / 'sample.nii.gz')
            nib.save(nib.Nifti1Image(sample.detach().cpu().numpy()[i], np.eye(4)), out_sample)
            print(f'Saved sample to {out_sample}')

            out_target = str(out_dir / 'target.nii.gz')
            nib.save(nib.Nifti1Image(target.detach().cpu().numpy()[i], np.eye(4)), out_target)


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="D:/user/BraTS2024-GLI",   # remote desktop BraTS root
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="./runs/t1-c/brats_045000.pt",  # t1c agent checkpoint
        devices=[0],
        output_dir='./results/t2f',
        mode='default',
        renormalize=False,
        half_res_crop=False,
        concat_coords=False,
        contr="t2f",
        # Architecture — must match training run exactly
        image_size=96,
        num_channels=64,
        channel_mult="1,2,4,4",
        in_channels=32,
        out_channels=8,
        dims=3,
        num_res_blocks=2,
        num_heads=1,
        num_groups=32,
        attention_resolutions="12",
        bottleneck_attention=True,
        resample_2d=False,
        additive_skips=False,
        use_freq=True,
        use_cross_attn=True,
        cond_channels=24,
        predict_xstart=True,
        noise_schedule="linear",
        diffusion_steps=1000,
        use_fp16=True,
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items()
                     if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()


















