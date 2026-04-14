"""
Visualisation script — finds the best patient from metrics_t1c.csv (highest PSNR),
then plots axial / coronal / sagittal slices of synthesised vs ground-truth T1C
side-by-side with difference map and a metrics table.

Usage:
    py -3.9 scripts/visualise.py \
        --results_dir ./results/t1c \
        --metrics_csv ./results/t1c/metrics_t1c.csv \
        --out_png    ./results/t1c/best_patient_comparison.png
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless — saves PNG without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vol(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata().astype(np.float32)


def mid_slices(vol: np.ndarray):
    """Return axial / coronal / sagittal mid-slices from a (H, W, D) volume."""
    H, W, D = vol.shape
    return (
        vol[H // 2, :, :],   # axial (coronal cut through middle of H)
        vol[:, W // 2, :],   # coronal
        vol[:, :, D // 2],   # sagittal
    )


def find_best_patient(metrics_csv: Path) -> tuple:
    """Read CSV, return (patient_id, mse, psnr, ssim) of highest-PSNR row."""
    best = None
    with open(metrics_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["patient_id"] in ("MEAN", "STD", "MEDIAN"):
                continue
            psnr = float(row["PSNR_dB"])
            if best is None or psnr > best[2]:
                best = (row["patient_id"],
                        float(row["MSE"]),
                        psnr,
                        float(row["SSIM"]))
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise best T1C synthesis vs ground truth."
    )
    parser.add_argument("--results_dir", default="./results/t1c",
                        help="Root results dir (per-patient sub-folders inside).")
    parser.add_argument("--metrics_csv",  default="./results/t1c/metrics_t1c.csv",
                        help="CSV produced by evaluate.py.")
    parser.add_argument("--out_png",      default="./results/t1c/best_patient_comparison.png",
                        help="Output PNG path.")
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    best = find_best_patient(metrics_csv)
    if best is None:
        raise RuntimeError("No valid patients found in CSV.")

    patient_id, mse, psnr, ssim = best
    print(f"Best patient: {patient_id}  PSNR={psnr:.3f} dB  SSIM={ssim:.4f}  MSE={mse:.6f}")

    patient_dir = Path(args.results_dir) / patient_id
    sample = load_vol(patient_dir / "sample.nii.gz")
    target = load_vol(patient_dir / "target.nii.gz")
    diff   = np.abs(sample - target)

    # ---- layout: 3 rows (axial / coronal / sagittal) x 3 cols (pred / target / diff) ----
    planes   = ["Axial", "Coronal", "Sagittal"]
    samp_sl  = mid_slices(sample)
    targ_sl  = mid_slices(target)
    diff_sl  = mid_slices(diff)

    fig = plt.figure(figsize=(14, 13), facecolor="#111111")
    fig.suptitle(
        f"T1C Synthesis — {patient_id}\n"
        f"PSNR = {psnr:.2f} dB   SSIM = {ssim:.4f}   MSE = {mse:.6f}",
        color="white", fontsize=13, y=0.98,
    )

    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08)

    col_titles = ["Synthesised T1C", "Ground Truth T1C", "Absolute Difference"]
    cmaps      = ["gray", "gray", "hot"]

    for row_idx, (plane, s, t, d) in enumerate(zip(planes, samp_sl, targ_sl, diff_sl)):
        inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[row_idx], wspace=0.04)
        imgs  = [np.rot90(s), np.rot90(t), np.rot90(d)]
        vmaxes = [1.0, 1.0, diff.max() if diff.max() > 0 else 1.0]

        for col_idx, (img, cmap, vmax, col_title) in enumerate(
                zip(imgs, cmaps, vmaxes, col_titles)):
            ax = fig.add_subplot(inner[col_idx])
            ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, aspect="auto",
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")

            if row_idx == 0:
                ax.set_title(col_title, color="white", fontsize=10, pad=4)
            if col_idx == 0:
                ax.set_ylabel(plane, color="#cccccc", fontsize=9, labelpad=4)

    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
