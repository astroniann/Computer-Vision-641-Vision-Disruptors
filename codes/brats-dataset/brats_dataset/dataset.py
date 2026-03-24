# PyTorch Dataset for BraTS brain tumour data, aligned with cwdm's BRATSVolumes.
#
# Key design decisions matching cwdm (guided_diffusion/bratsloader.py):
#   - __getitem__ returns {'t1n', 't1c', 't2w', 't2f'} as separate (1, H, W, D) tensors
#   - Normalisation is clip-and-normalize to [0, 1] (not Z-score)
#   - No DWT is done here — cwdm does DWT inside gaussian_diffusion.training_losses()
#   - No target/condition split here — cwdm does that inside training_losses() via `contr` arg
#   - Volumes are zero-padded to (1, 240, 240, 160) then centre-cropped to (1, 224, 224, 160)
#     matching cwdm's exact spatial handling

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .patient import load_patient


# Canonical BraTS depth: 155 slices padded to 160, then no depth crop
PAD_H, PAD_W, PAD_D = 240, 240, 160
CROP_H_START, CROP_H_END = 8, -8   # 240 -> 224
CROP_W_START, CROP_W_END = 8, -8   # 240 -> 224
# depth stays at 160 (cwdm does NOT crop depth)
OUT_H, OUT_W, OUT_D = 224, 224, 160


class BraTS20Dataset(Dataset):
    """
    Parameters
    ----------
    data_root   : root folder; patient subfolders sit directly inside
                  (e.g. data_root/BraTS-GLI-00000-000/)
    csv_path    : optional path to an Excel/CSV file with a 'BraTS Subject ID'
                  column; if given, only listed patients are loaded
    split       : 'train' | 'validation' | 'additional'
                  Controls whether segmentation masks are loaded.

    Returns (per __getitem__)
    -------
    dict with keys:
        't1n'        : torch.FloatTensor (1, 224, 224, 160)  range [0, 1]
        't1c'        : torch.FloatTensor (1, 224, 224, 160)  range [0, 1]
        't2w'        : torch.FloatTensor (1, 224, 224, 160)  range [0, 1]
        't2f'        : torch.FloatTensor (1, 224, 224, 160)  range [0, 1]
        'missing'    : str  always 'none' (kept for cwdm train_util compatibility)
        'patient_id' : str
        'seg'        : torch.LongTensor (H, W, D)  only when split in {'train','validation'}
    """

    HAS_SEG = {"train", "validation"}

    SPLIT_DIRS = {
        "train":      "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2",
        "validation": "BraTS2024-BraTS-GLI-ValidationData/validation_data",
        "additional": "BraTS2024-BraTS-GLI-AdditionalTrainingData/training_data_additional",
    }

    def __init__(
        self,
        data_root: str,
        csv_path: Optional[str] = None,
        split: str = "train",
    ):
        assert split in self.SPLIT_DIRS, \
            f"split must be one of {list(self.SPLIT_DIRS)}, got '{split}'"

        self.split   = split
        self.has_seg = split in self.HAS_SEG

        split_dir = Path(data_root) / self.SPLIT_DIRS[split]
        self.patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if csv_path and Path(csv_path).exists():
            df        = pd.read_excel(csv_path)
            valid_ids = set(df["BraTS Subject ID"].astype(str))
            self.patient_dirs = [
                d for d in self.patient_dirs if d.name in valid_ids
            ]

        print(f"[BraTS20Dataset | {split}] {len(self.patient_dirs)} patients loaded.")

    # ------------------------------------------------------------------
    # Internal helper: pad + crop to match cwdm spatial handling
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_and_crop(vol_np: np.ndarray) -> torch.Tensor:
        """
        vol_np : (1, H, W, D)  raw loaded volume, D may be 155

        Returns torch.FloatTensor (1, 224, 224, 160)
        matching cwdm's:
            t1n = torch.zeros(1, 240, 240, 160)
            t1n[:, :, :, :155] = tensor(vol)
            t1n = t1n[:, 8:-8, 8:-8, :]
        """
        _, H, W, D = vol_np.shape
        pad = torch.zeros(1, PAD_H, PAD_W, PAD_D, dtype=torch.float32)
        t   = torch.from_numpy(vol_np).float()
        pad[:, :H, :W, :min(D, PAD_D)] = t[:, :PAD_H, :PAD_W, :PAD_D]
        # centre-crop H and W
        return pad[:, CROP_H_START:CROP_H_END, CROP_W_START:CROP_W_END, :]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> dict:
        patient_dir = self.patient_dirs[idx]

        # load_patient now returns a dict of (1, H, W, D) np arrays + optional seg
        modality_vols, seg = load_patient(str(patient_dir), load_seg=self.has_seg)

        sample = {
            't1n':        self._pad_and_crop(modality_vols['t1n']),
            't1c':        self._pad_and_crop(modality_vols['t1c']),
            't2w':        self._pad_and_crop(modality_vols['t2w']),
            't2f':        self._pad_and_crop(modality_vols['t2f']),
            'missing':    'none',   # cwdm train_util expects this key
            'patient_id': patient_dir.name,
        }

        if seg is not None:
            sample['seg'] = torch.from_numpy(seg).long()

        return sample
