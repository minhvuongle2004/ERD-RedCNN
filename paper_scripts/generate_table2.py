"""
generate_table2.py
==================
Tong hop Table 2: Validation Performance Across Seeds

Muc tieu:
  - Doc file Metrics.csv cua Variant B, C, D (moi variant 3 seeds)
  - Lay Best Val SSIM (max) va Best Val PSNR tuong ung
  - Tinh Mean +- Std qua 3 seeds
  - In bang dep ra terminal
  - Luu ra results/evaluation/table2_validation_seeds.csv

Ghi chu:
  - Variant A (RED-CNN) la pretrained, khong co training log → ghi N/A
  - "Best" = epoch co SSIM cao nhat tren validation set

Chay:
  python generate_table2.py
"""

import os
import sys
import glob
import pandas as pd
import numpy as np

# Fix encoding Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ─── Config ────────────────────────────────────────────────────────────────
TRAINING_ROOT = os.path.join("results", "training")
OUT_DIR       = os.path.join("results", "evaluation")
OUT_CSV       = os.path.join(OUT_DIR, "table2_validation_seeds.csv")
SEEDS         = [1339, 2024, 42]

# Map variant → subfolder pattern → seed subfolder pattern
VARIANT_MAP = {
    "B": {
        1339: "VariantB/Seed1339/variantB_seed1339_Metrics.csv",
        2024: "VariantB/Seed2024/variantB_seed2024_Metrics.csv",
        42:   "VariantB/Seed42/variantB_seed42_Metrics.csv",
    },
    "C": {
        1339: "VariantC/Seed1339/variantC_seed1339_Metrics.csv",
        2024: "VariantC/Seed2024/variantC_seed2024_Metrics.csv",
        42:   "VariantC/Seed42/variantC_seed42_Metrics.csv",
    },
    "D": {
        1339: "VariantD/seed1339/variantD_seed1339_Metrics.csv",
        2024: "VariantD/seed2024/seed2024_Metrics.csv",
        42:   "VariantD/seed42/variantD_seed42_Metrics.csv",
    },
}

VARIANT_LABEL = {
    "A": "A — RED-CNN (Baseline, pretrained)",
    "B": "B — + EdgeDilatedResidualBlock",
    "C": "C — + FixedSobelLayer",
    "D": "D — Full EDR-REDNet (Ours)",
}

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Read best SSIM & PSNR for each variant/seed ──────────────────────────
def read_best(csv_path):
    """Return (best_ssim, best_psnr) at the epoch with max SSIM."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Normalize column names
    ssim_col = next((c for c in df.columns if 'SSIM' in c.upper()), None)
    psnr_col = next((c for c in df.columns if 'PSNR' in c.upper()), None)
    if ssim_col is None or psnr_col is None:
        raise ValueError(f"Cannot find SSIM/PSNR columns in {csv_path}: {df.columns.tolist()}")
    df[ssim_col] = pd.to_numeric(df[ssim_col], errors='coerce')
    df[psnr_col] = pd.to_numeric(df[psnr_col], errors='coerce')
    best_idx = df[ssim_col].idxmax()
    return float(df.loc[best_idx, ssim_col]), float(df.loc[best_idx, psnr_col])


rows = []

# Variant A — pretrained, no training log
rows.append({
    "Variant": VARIANT_LABEL["A"],
    "Seed 1339 — Val SSIM": "N/A",
    "Seed 1339 — Val PSNR": "N/A",
    "Seed 2024 — Val SSIM": "N/A",
    "Seed 2024 — Val PSNR": "N/A",
    "Seed 42   — Val SSIM": "N/A",
    "Seed 42   — Val PSNR": "N/A",
    "Mean SSIM ± Std": "N/A",
    "Mean PSNR ± Std": "N/A",
    "Best Seed": "—",
})

# Variants B, C, D
for v in ["B", "C", "D"]:
    ssim_vals, psnr_vals = [], []
    seed_results = {}
    for seed in SEEDS:
        rel_path = VARIANT_MAP[v][seed]
        full_path = os.path.join(TRAINING_ROOT, rel_path)
        if not os.path.exists(full_path):
            print(f"  [WARN] Not found: {full_path}")
            ssim_vals.append(np.nan)
            psnr_vals.append(np.nan)
            seed_results[seed] = (np.nan, np.nan)
        else:
            ssim, psnr = read_best(full_path)
            ssim_vals.append(ssim)
            psnr_vals.append(psnr)
            seed_results[seed] = (ssim, psnr)

    mean_ssim = np.nanmean(ssim_vals)
    std_ssim  = np.nanstd(ssim_vals)
    mean_psnr = np.nanmean(psnr_vals)
    std_psnr  = np.nanstd(psnr_vals)
    best_seed = SEEDS[int(np.nanargmax(ssim_vals))]

    rows.append({
        "Variant": VARIANT_LABEL[v],
        "Seed 1339 — Val SSIM": f"{seed_results[1339][0]:.4f}" if not np.isnan(seed_results[1339][0]) else "N/A",
        "Seed 1339 — Val PSNR": f"{seed_results[1339][1]:.2f}" if not np.isnan(seed_results[1339][1]) else "N/A",
        "Seed 2024 — Val SSIM": f"{seed_results[2024][0]:.4f}" if not np.isnan(seed_results[2024][0]) else "N/A",
        "Seed 2024 — Val PSNR": f"{seed_results[2024][1]:.2f}" if not np.isnan(seed_results[2024][1]) else "N/A",
        "Seed 42   — Val SSIM": f"{seed_results[42][0]:.4f}"   if not np.isnan(seed_results[42][0])   else "N/A",
        "Seed 42   — Val PSNR": f"{seed_results[42][1]:.2f}"   if not np.isnan(seed_results[42][1])   else "N/A",
        "Mean SSIM ± Std": f"{mean_ssim:.4f} ± {std_ssim:.4f}",
        "Mean PSNR ± Std": f"{mean_psnr:.2f} ± {std_psnr:.2f}",
        "Best Seed": str(best_seed),
    })

df_table2 = pd.DataFrame(rows)
df_table2.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n[OK] Da luu: {os.path.abspath(OUT_CSV)}")

# ─── In bang dep ───────────────────────────────────────────────────────────
print("\n" + "="*80)
print(" TABLE 2. Validation Performance Across Seeds (Best Val SSIM per seed)")
print("="*80)
print(f"\n{'Variant':<42} | {'Seed 1339':^22} | {'Seed 2024':^22} | {'Seed 42':^22} | {'Mean ± Std':^24} | Best")
print(f"{'':42} | {'SSIM':^11}{'PSNR':^11} | {'SSIM':^11}{'PSNR':^11} | {'SSIM':^11}{'PSNR':^11} | {'SSIM':^12}{'PSNR':^12} |")
print("-"*160)

for r in rows:
    v     = r["Variant"][:41]
    s1s   = r["Seed 1339 — Val SSIM"]
    s1p   = r["Seed 1339 — Val PSNR"]
    s2s   = r["Seed 2024 — Val SSIM"]
    s2p   = r["Seed 2024 — Val PSNR"]
    s42s  = r["Seed 42   — Val SSIM"]
    s42p  = r["Seed 42   — Val PSNR"]
    ms    = r["Mean SSIM ± Std"]
    mp    = r["Mean PSNR ± Std"]
    best  = r["Best Seed"]
    print(f"{v:<42} | {s1s:^11}{s1p:^11} | {s2s:^11}{s2p:^11} | {s42s:^11}{s42p:^11} | {ms:^24} | {best}")

print("="*80)
print("\nGhi chu:")
print("  - Variant A la pretrained RED-CNN (khong co training log).")
print("  - 'Best' = epoch co Val SSIM cao nhat trong qua trinh training.")
print("  - 'Best Seed' = seed cho ket qua Val SSIM cao nhat, dung de bao cao test.")
print(f"\n  File CSV: {os.path.abspath(OUT_CSV)}")
