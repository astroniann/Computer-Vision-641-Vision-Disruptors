
# Loads all modalities (+ optional segmentation) for a single BraTS20 patient.

import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .io     import load_volume
from .labels import remap_labels

MODALITIES     = ["t2f", "t1n", "t1c", "t2w"]   # BraTS 2024: flair, t1, t1ce, t2
NUM_MODALITIES = len(MODALITIES)   # 4 input channels


def normalize_volume(image: np.ndarray) -> np.ndarray:
    """ 
    Z-score normalize only the non-zero (brain) regions per modality. 
    Leaves the background (0) as 0. 
    """
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        channel = image[c]
        mask = channel > 0
        if mask.any():
            mean = channel[mask].mean()
            std  = channel[mask].std()
            if std > 1e-8:
                out[c][mask] = (channel[mask] - mean) / std
            else:
                out[c][mask] = channel[mask] - mean
    return out

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
    image : np.ndarray  shape (4, H, W, D)  float32 (normalized)
    seg   : np.ndarray  shape (H, W, D)     uint8, or None
    """
    patient_dir = Path(patient_dir)
    patient_id  = patient_dir.name

    modality_vols = []
    for mod in MODALITIES:
        files = glob.glob(str(patient_dir / f"{patient_id}-{mod}.nii*"))
        if not files:
            raise FileNotFoundError(
                f"Missing modality '{mod}' for {patient_id}"
            )
        modality_vols.append(load_volume(files[0]))

    image = np.stack(modality_vols, axis=0).astype(np.float32)   # (4, H, W, D)
    image = normalize_volume(image)

    seg = None
    if load_seg:
        files = glob.glob(str(patient_dir / f"{patient_id}-seg.nii*"))
        if not files:
            raise FileNotFoundError(
                f"Missing segmentation mask for {patient_id}"
            )
        seg = load_volume(files[0]).astype(np.uint8)
        seg = remap_labels(seg)

    return image, seg