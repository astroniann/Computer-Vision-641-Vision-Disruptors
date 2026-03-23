"""
main.py
-------
Entry point — loads all three splits and prints a batch summary.
Replace DATA_ROOT and CSV_PATH with your actual paths.
"""

from dataset import get_dataloader

DATA_ROOT = "/path/to/brats20_data"
CSV_PATH  = "/path/to/brats20_data.csv"

train_loader = get_dataloader(DATA_ROOT, CSV_PATH, split="train")
val_loader   = get_dataloader(DATA_ROOT, CSV_PATH, split="validation")
add_loader   = get_dataloader(DATA_ROOT, CSV_PATH, split="additional")

for batch in train_loader:
    print("image :", batch["image"].shape)   # (B, 4, H, W, D)
    print("seg   :", batch["seg"].shape)     # (B, H, W, D)
    break