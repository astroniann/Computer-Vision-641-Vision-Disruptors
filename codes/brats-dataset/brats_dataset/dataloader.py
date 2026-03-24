# DataLoader factory — aligned with cwdm's train.py usage.
#
# cwdm's train.py does:
#   ds = BRATSVolumes(args.data_dir, mode='train')
#   datal = DataLoader(ds, batch_size=..., shuffle=True, num_workers=...)
#
# We replicate that pattern exactly using our BraTS20Dataset,
# which already returns the same {'t1n','t1c','t2w','t2f',...} dict format.

from typing import Optional

from torch.utils.data import DataLoader

from .dataset import BraTS20Dataset


def get_dataloader(
    data_root: str,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 0,
    csv_path: Optional[str] = None,
) -> DataLoader:
    """
    Build a DataLoader for a given split.

    Parameters
    ----------
    data_root   : root folder containing patient subfolders
    split       : 'train' | 'validation' | 'additional'
    batch_size  : patients per batch (typically 1 for 3D MRI)
    num_workers : parallel workers for loading (cwdm default is 0 in train.py)
    csv_path    : optional path to subject-ID filter file

    Returns
    -------
    torch.utils.data.DataLoader  yielding dicts with keys:
        't1n', 't1c', 't2w', 't2f'  → FloatTensor (B, 1, 224, 224, 160)
        'missing'                    → list of str (always 'none')
        'patient_id'                 → list of str
        'seg'                        → LongTensor (B, H, W, D)  [train/val only]
    """
    dataset = BraTS20Dataset(
        data_root=data_root,
        csv_path=csv_path,
        split=split,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )
