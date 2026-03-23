# PyTorch Dataset for BraTS20 brain tumour segmentation with DWT support.

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .patient import load_patient
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


class BraTS20Dataset(Dataset):
    """
    Parameters
    ----------
    data_root   : root folder containing train/ validation/ additional/
    csv_path    : path to brats20_data.csv (or None to load all patients found)
    split       : 'train' | 'validation' | 'additional'
    wavename    : pywt wavelet name passed to DWT_3D / IDWT_3D, e.g. 'haar'
    use_dwt     : if True, each image sample is wavelet-decomposed before
                  being returned.  The dataset item will then contain:
                      'image'          – original tensor  (4, H, W, D)
                      'image_dwt_LLL'  – (4, H/2, W/2, D/2)
                      'image_dwt_LLH'  – (4, H/2, W/2, D/2)
                      'image_dwt_LHL'  – (4, H/2, W/2, D/2)
                      'image_dwt_LHH'  – (4, H/2, W/2, D/2)
                      'image_dwt_HLL'  – (4, H/2, W/2, D/2)
                      'image_dwt_HLH'  – (4, H/2, W/2, D/2)
                      'image_dwt_HHL'  – (4, H/2, W/2, D/2)
                      'image_dwt_HHH'  – (4, H/2, W/2, D/2)
                  When False, only 'image' is returned (original behaviour).
    """

    HAS_SEG = {"train", "validation"}

    def __init__(
        self,
        data_root: str,
        csv_path: Optional[str],
        split: str = "train",
        wavename: str = "haar",
        use_dwt: bool = False,
    ):
        assert split in {"train", "validation", "additional"}, \
            f"split must be 'train', 'validation', or 'additional', got '{split}'"

        self.split    = split
        self.has_seg  = split in self.HAS_SEG
        self.wavename = wavename
        self.use_dwt  = use_dwt

        # Lazy-initialised wavelet modules (CPU; move to GPU in training loop)
        self._dwt  = None
        self._idwt = None

        SPLIT_DIRS = {
            "train":      "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2",
            "validation": "BraTS2024-BraTS-GLI-ValidationData/validation_data",
            "additional": "BraTS2024-BraTS-GLI-AdditionalTrainingData/training_data_additional",
        }
        split_dir = Path(data_root) / SPLIT_DIRS[split]
        self.patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        if csv_path and Path(csv_path).exists():
            df        = pd.read_excel(csv_path)
            valid_ids = set(df["BraTS Subject ID"].astype(str))
            self.patient_dirs = [
                d for d in self.patient_dirs if d.name in valid_ids
            ]

        print(f"[{split}] {len(self.patient_dirs)} patients loaded. "
              f"DWT={'on (' + wavename + ')' if use_dwt else 'off'}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_dwt(self) -> DWT_3D:
        if self._dwt is None:
            self._dwt = DWT_3D(self.wavename)
        return self._dwt

    def _get_idwt(self) -> IDWT_3D:
        if self._idwt is None:
            self._idwt = IDWT_3D(self.wavename)
        return self._idwt

    # ------------------------------------------------------------------
    # Public wavelet helpers (useful in training loops / inference)
    # ------------------------------------------------------------------

    SUB_BAND_NAMES = ('LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH')

    def dwt(self, image: torch.Tensor):
        """
        Apply 3-D DWT to a single image tensor.

        Parameters
        ----------
        image : (C, H, W, D)  or  (B, C, H, W, D)

        Returns
        -------
        Tuple of 8 sub-band tensors in order (LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH),
        each (C, H/2, W/2, D/2) or (B, C, H/2, W/2, D/2).
        """
        squeeze = image.dim() == 4
        if squeeze:
            image = image.unsqueeze(0)
        sub_bands = self._get_dwt()(image)
        if squeeze:
            return tuple(s.squeeze(0) for s in sub_bands)
        return sub_bands

    def idwt(self, sub_bands) -> torch.Tensor:
        """
        Apply 3-D IDWT to recover the original image.

        Parameters
        ----------
        sub_bands : tuple of 8 tensors (LLL, LLH, …, HHH),
                    each (C, H/2, W/2, D/2) or (B, C, H/2, W/2, D/2)

        Returns
        -------
        (C, H, W, D)  or  (B, C, H, W, D)
        """
        squeeze = sub_bands[0].dim() == 4
        if squeeze:
            sub_bands = tuple(s.unsqueeze(0) for s in sub_bands)
        out = self._get_idwt()(*sub_bands)
        return out.squeeze(0) if squeeze else out

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.patient_dirs)

    def __getitem__(self, idx: int) -> dict:
        patient_dir = self.patient_dirs[idx]

        image, seg = load_patient(str(patient_dir), load_seg=self.has_seg)
        image_t = torch.from_numpy(image).float()   # (4, H, W, D)

        sample = {
            "image":      image_t,
            "patient_id": patient_dir.name,
        }

        if self.use_dwt:
            sub_bands = self.dwt(image_t)   # tuple of 8, each (4, H/2, W/2, D/2)
            for name, band in zip(self.SUB_BAND_NAMES, sub_bands):
                sample[f"image_dwt_{name}"] = band

        if seg is not None:
            sample["seg"] = torch.from_numpy(seg).long()   # (H, W, D)

        return sample
