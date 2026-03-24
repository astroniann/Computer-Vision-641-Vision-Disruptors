# Loads all modalities (+ optional segmentation) for a single BraTS patient.
# Normalisation: clip to [0.1th, 99.9th] percentile then scale to [0, 1],
# matching cwdm's clip_and_normalize() in guided_diffusion/bratsloader.py.

import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .io     import load_volume
from .labels import remap_labels

MODALITIES     = ["t2f", "t1n", "t1c", "t2w"]   # BraTS 2024: flair, t1, t1ce, t2
NUM_MODALITIES = len(MODALITIES)                  # 4 input channels


def clip_and_normalize(img: np.ndarray) -> np.ndarray:
    """
    Clip to [0.1th, 99.9th] percentile then min-max scale to [0, 1].
    Matches cwdm guided_diffusion/bratsloader.py clip_and_normalize() exactly.
    """
    img_clipped = np.clip(
        img,
        np.quantile(img, 0.001),
        np.quantile(img, 0.999),
    )
    img_min = np.min(img_clipped)
    img_max = np.max(img_clipped)
    if img_max - img_min < 1e-8:
        return np.zeros_like(img_clipped, dtype=np.float32)
    return ((img_clipped - img_min) / (img_max - img_min)).astype(np.float32)


def load_patient(
    patient_dir: str,
    load_seg: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load all four modalities and (optionally) the segmentation mask
    for a single patient.

    Parameters
    ----------
    patient_dir : path to the patient folder
    load_seg    : whether to load the segmentation mask

    Returns
    -------
    image : dict with keys 't2f', 't1n', 't1c', 't2w'
            each value is np.ndarray of shape (1, H, W, D), float32, range [0, 1]
    seg   : np.ndarray shape (H, W, D) uint8, or None
    """
    patient_dir = Path(patient_dir)
    patient_id  = patient_dir.name

    modality_vols = {}
    for mod in MODALITIES:
        files = glob.glob(str(patient_dir / f"{patient_id}-{mod}.nii*"))
        if not files:
            raise FileNotFoundError(
                f"Missing modality '{mod}' for {patient_id}"
            )
        vol = load_volume(files[0]).astype(np.float32)
        modality_vols[mod] = clip_and_normalize(vol)[np.newaxis]  # (1, H, W, D)

    seg = None
    if load_seg:
        files = glob.glob(str(patient_dir / f"{patient_id}-seg.nii*"))
        if not files:
            raise FileNotFoundError(
                f"Missing segmentation mask for {patient_id}"
            )
        seg = load_volume(files[0]).astype(np.uint8)
        seg = remap_labels(seg)

    return modality_vols, seg
