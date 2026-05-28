"""
split_paper_figures.py
======================
Tach Figure 3-4-5 thanh 3 file rieng biet tu du lieu LDCT test set.

Dau ra:
  results/fig3_qualitative_comparison.png   -- So sanh anh xam A/B/C/D
  results/fig4_difference_maps.png          -- Difference map vs NDCT
  results/fig5_sobel_edge_maps.png          -- Sobel edge maps

Cach chay:
  python split_paper_figures.py

Yeu cau:
  - Da chay evaluate_statistical_test.py (co data mayo tren may)
  - pip install scikit-image
"""

import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import filters

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ─── Fix Unicode ─────────────────────────────────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.makedirs("results", exist_ok=True)

# ─── Load models & dataset (reuse app logic) ─────────────────────────────────
print("[INFO] Import modules...")
import torch
sys.path.insert(0, ".")

# Try to import project modules
try:
    from dataset import LDCTDataset
    from models import REDCNNWrapper, EDRREDNet, VariantB, VariantC
    HAS_MODELS = True
except Exception as e:
    print(f"  [WARN] Khong import duoc models: {e}")
    HAS_MODELS = False

# ─── Settings ─────────────────────────────────────────────────────────────────
PATIENT  = "L150"
SLICE_ID = 77
ROI      = (180, 300, 220, 340)   # (y1, y2, x1, x2)
HU_MIN, HU_MAX = -160, 240
DPI = 300

# ─── Window HU ────────────────────────────────────────────────────────────────
def window_image(img_hu, hu_min=HU_MIN, hu_max=HU_MAX):
    img = np.clip(img_hu, hu_min, hu_max)
    return (img - hu_min) / (hu_max - hu_min)

# ─── Load pre-saved crops from fig_paper_L150_sl77.png (fallback) ────────────
# If models not available, crop rows from the combined figure using PIL
def split_from_combined_png():
    """Crop row-by-row from the existing combined figure."""
    from PIL import Image

    src = os.path.join("results", "fig_paper_L150_sl77.png")
    if not os.path.exists(src):
        print(f"[ERR] Khong tim thay: {src}")
        print("      -> Hay chay app.py va click 'Tao Figure tong hop' truoc.")
        sys.exit(1)

    print(f"[INFO] Doc tu: {src}")
    img = Image.open(src)
    W, H = img.size
    print(f"       Kich thuoc goc: {W} x {H} px")

    # Estimate row boundaries (3 equal rows minus title bar at top)
    # The combined figure has a suptitle + 3 rows
    # We detect by scanning for white strips
    img_arr = np.array(img.convert("RGB"))

    # Find row boundaries using horizontal mean brightness
    # Title region is at the very top (~3-5% height)
    # Then 3 equal content rows
    title_end = int(H * 0.04)   # ~4% for suptitle
    content_h = H - title_end
    row_h = content_h // 3

    rows_bounds = []
    for i in range(3):
        y_start = title_end + i * row_h
        y_end   = title_end + (i + 1) * row_h
        rows_bounds.append((y_start, y_end))

    # Small left margin for row labels (~5%)
    x_start = int(W * 0.05)

    out_paths = [
        os.path.join("results", "fig3_qualitative_comparison.png"),
        os.path.join("results", "fig4_difference_maps.png"),
        os.path.join("results", "fig5_sobel_edge_maps.png"),
    ]
    titles = [
        "Figure 3. Qualitative Comparison: LDCT, A, B, C, D, NDCT",
        "Figure 4. Difference Maps relative to NDCT",
        "Figure 5. Sobel Edge Map Comparison",
    ]

    for i, ((y0, y1), out_p, title) in enumerate(zip(rows_bounds, out_paths, titles)):
        row_img = img.crop((x_start, y0, W, y1))
        # Add title bar on top
        fig, ax = plt.subplots(figsize=(row_img.width / 300 * (W/300), row_img.height / 300 + 0.4))
        ax.imshow(np.array(row_img))
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", color="#0d1b2a", pad=8)
        plt.tight_layout(pad=0.2)
        fig.savefig(out_p, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [OK] {out_p}")

    return out_paths


def generate_from_models():
    """Generate 3 figures from raw model inference."""
    print("[INFO] Generating from model inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load dataset
    dataset = LDCTDataset("data/mayo_test", mode="test")
    pat_idx = next(i for i, s in enumerate(dataset.samples) if s["info"]["id"] == PATIENT)
    batch   = dataset[pat_idx]
    x_np    = batch["x"][SLICE_ID].numpy()          # (512,512)
    y_np    = batch["y"][SLICE_ID].numpy()          # (512,512) NDCT

    # HU denorm
    hu_mean = dataset.hu_mean if hasattr(dataset, 'hu_mean') else 0
    hu_std  = dataset.hu_std  if hasattr(dataset, 'hu_std')  else 1
    def to_hu(t):
        return t.cpu().numpy().squeeze() * hu_std + hu_mean

    x_t = torch.tensor(x_np).unsqueeze(0).unsqueeze(0).to(device)

    # Load models
    model_paths = {
        "A — Baseline":   "results/models/variant_a_best.pth",
        "B — +EdgeBlock": "results/models/variant_b_best.pth",
        "C — +Sobel":     "results/models/variant_c_best.pth",
        "D — EDR-REDNet": "results/models/variant_d_best.pth",
    }
    imgs = {"LDCT": x_np * hu_std + hu_mean}
    for name, path in model_paths.items():
        if os.path.exists(path):
            # model loading logic here
            pass
    imgs["NDCT (GT)"] = y_np * hu_std + hu_mean

    y1_, y2_, x1_, x2_ = ROI
    cols = list(imgs.keys())

    # Crop
    crops = {k: v[y1_:y2_, x1_:x2_] for k, v in imgs.items()}
    ndct_crop = crops["NDCT (GT)"]

    _save_fig3(crops, cols)
    _save_fig4(crops, cols, ndct_crop)
    _save_fig5(crops, cols)


def _save_fig3(crops, cols):
    """Figure 3: Qualitative comparison (gray)."""
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Figure 3. Qualitative Comparison — Patient {PATIENT}, Slice {SLICE_ID}",
                 fontsize=12, fontweight="bold", color="#0d1b2a", y=1.02)
    for ax, k in zip(axes, cols):
        ax.imshow(window_image(crops[k]), cmap="gray", interpolation="nearest")
        ax.set_title(k, fontsize=10, fontweight="bold", color="#0d1b2a", pad=5)
        ax.axis("off")
    plt.tight_layout(pad=0.4)
    out = os.path.join("results", "fig3_qualitative_comparison.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {out}")


def _save_fig4(crops, cols, ndct_crop):
    """Figure 4: Difference maps."""
    diff_keys = [k for k in cols if k not in ["LDCT", "NDCT (GT)"]]
    n = len(diff_keys)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Figure 4. Difference Maps relative to NDCT — Patient {PATIENT}, Slice {SLICE_ID}",
                 fontsize=12, fontweight="bold", color="#0d1b2a", y=1.02)
    if n == 1: axes = [axes]
    for ax, k in zip(axes, diff_keys):
        diff = crops[k] - ndct_crop
        im = ax.imshow(diff, cmap="seismic", vmin=-200, vmax=200, interpolation="nearest")
        ax.set_title(f"Δ {k}", fontsize=10, fontweight="bold", color="#0d1b2a", pad=5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0.4)
    out = os.path.join("results", "fig4_difference_maps.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {out}")


def _save_fig5(crops, cols):
    """Figure 5: Sobel edge maps."""
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"Figure 5. Sobel Edge Map Comparison — Patient {PATIENT}, Slice {SLICE_ID}",
                 fontsize=12, fontweight="bold", color="#0d1b2a", y=1.02)
    for ax, k in zip(axes, cols):
        edge = filters.sobel(crops[k])
        ax.imshow(edge, cmap="hot", interpolation="nearest")
        ax.set_title(f"Sobel: {k}", fontsize=10, fontweight="bold", color="#0d1b2a", pad=5)
        ax.axis("off")
    plt.tight_layout(pad=0.4)
    out = os.path.join("results", "fig5_sobel_edge_maps.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n======================================")
    print(" TACH FIGURE 3-4-5 (Bach rieng biet)")
    print("======================================")

    # Method: crop from existing combined PNG
    # (Phuong phap nay khong can model/data, chi can fig_paper_L150_sl77.png)
    out_paths = split_from_combined_png()

    print("\n======================================")
    print(" HOAN THANH!")
    print("======================================")
    for p in out_paths:
        print(f"  {p}")
    print("\n  3 file da san sang cho bai bao!")
