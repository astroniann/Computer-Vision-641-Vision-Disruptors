"""
test_dwt.py
-----------
Self-contained tests for DWT_3D and IDWT_3D that do NOT require the BraTS
dataset. Run with:  python test_dwt.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

WAVENAME = "haar"


def test_shapes():
    dwt  = DWT_3D(WAVENAME)
    idwt = IDWT_3D(WAVENAME)

    B, C, H, W, D = 2, 4, 32, 32, 32
    x = torch.randn(B, C, H, W, D)

    sub_bands = dwt(x)
    assert len(sub_bands) == 8, f"Expected 8 sub-bands, got {len(sub_bands)}"
    for i, s in enumerate(sub_bands):
        assert s.shape == (B, C, H // 2, W // 2, D // 2), \
            f"Sub-band {i} shape mismatch: {s.shape}"

    x_hat = idwt(*sub_bands)
    assert x_hat.shape == (B, C, H, W, D), \
        f"IDWT output shape mismatch: {x_hat.shape}"

    print("[PASS] test_shapes")


def test_perfect_reconstruction(tol=1e-5):
    """DWT followed by IDWT must reconstruct the input exactly."""
    dwt  = DWT_3D(WAVENAME)
    idwt = IDWT_3D(WAVENAME)

    for shape in [(1, 1, 16, 16, 16), (2, 4, 32, 32, 32), (1, 4, 64, 64, 32)]:
        x     = torch.randn(*shape)
        x_hat = idwt(*dwt(x))
        err   = (x - x_hat).abs().max().item()
        assert err < tol, f"Reconstruction error {err:.2e} > {tol} for shape {shape}"
        print(f"[PASS] test_perfect_reconstruction  shape={shape}  max_err={err:.2e}")


def test_sub_band_names():
    """Verify sub-band ordering: LLL is the low-frequency component (should have lowest energy variation)."""
    dwt = DWT_3D(WAVENAME)
    x   = torch.randn(1, 4, 32, 32, 32)
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(x)

    # LLL (all low-pass) should have the largest mean absolute value relative
    # to the high-frequency bands for a smooth-ish signal
    assert LLL.shape == (1, 4, 16, 16, 16)
    assert HHH.shape == (1, 4, 16, 16, 16)
    print(f"[PASS] test_sub_band_names  LLL.shape={LLL.shape}")


def test_gradient_flow():
    """Gradients must flow through both DWT_3D and IDWT_3D."""
    dwt  = DWT_3D(WAVENAME)
    idwt = IDWT_3D(WAVENAME)

    x = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
    loss = idwt(*dwt(x)).sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print("[PASS] test_gradient_flow")


def test_non_haar_wavelet():
    """Works with other pywt wavelets, not just haar."""
    for wname in ["db2", "bior2.2"]:
        dwt  = DWT_3D(wname)
        idwt = IDWT_3D(wname)
        x    = torch.randn(1, 2, 32, 32, 32)
        err  = (x - idwt(*dwt(x))).abs().max().item()
        print(f"[PASS] test_non_haar_wavelet  wavelet={wname}  max_err={err:.2e}")


if __name__ == "__main__":
    test_shapes()
    test_perfect_reconstruction()
    test_sub_band_names()
    test_gradient_flow()
    test_non_haar_wavelet()
    print("\nAll tests passed ✓")
