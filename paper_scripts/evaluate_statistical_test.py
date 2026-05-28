"""
evaluate_statistical_test.py
============================
Giai doan 3 - Kiem dinh Thong ke & Danh gia tren toan tap Test (100 benh nhan)

Muc tieu:
  1. Tai cac Best Models cua Variant A (Baseline), B, C, D len GPU.
  2. Chay inference qua toan bo tap Test (100 benh nhan) tren GPU.
  3. Tinh cac chi so: PSNR, SSIM, Edge SSIM (theo HU windowing chuan).
  4. Luu diem tung benh nhan vao CSV.
  5. Chay kiem dinh Wilcoxon (scipy) de so sanh Variant D voi A, B, C.
  6. Xuat bang ket qua Mean +/- Std va bang p-value ra CSV.

Cach chay:
  # Debug (chi 2 benh nhan - kiem tra nhanh):
  python evaluate_statistical_test.py --debug

  # Chay day du (toan bo 100 benh nhan, dung GPU):
  python evaluate_statistical_test.py
"""

# Fix Unicode encoding on Windows terminal (cp1252 does not support emoji)
import sys
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import ndimage, stats
from skimage import metrics

# ──────────────────────────────────────────────────────────────────────────────
# 1. PATCH TORCH.LOAD (nhất quán với app.py)
# ──────────────────────────────────────────────────────────────────────────────
_original_load = torch.load


def _safe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    if "map_location" not in kwargs:
        kwargs["map_location"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _original_load(*args, **kwargs)


torch.load = _safe_load


def _safe_load_yaml(path: str):
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


import ldctbench.evaluate.utils as eval_utils
import ldctbench.utils as lct_utils

eval_utils.torch.load = _safe_load  # type: ignore
eval_utils.load_yaml = _safe_load_yaml  # type: ignore
lct_utils.load_yaml = _safe_load_yaml  # type: ignore

from ldctbench.data import TestData
from ldctbench.evaluate import setup_trained_model
from ldctbench.hub import load_model

# ──────────────────────────────────────────────────────────────────────────────
# 2. CẤU HÌNH CÁC BEST MODELS (nhất quán với app.py)
# ──────────────────────────────────────────────────────────────────────────────
CFG_PATH = r"configs/edrrednet.yaml"

VARIANT_CONFIG = {
    "A — RED-CNN (Baseline)": {
        "mode": "hub",        # dùng load_model từ hub
        "hub_name": "redcnn",
    },
    "B — + EdgeBlock": {
        "mode": "wandb",
        "run_name": "edr_variant_b",
        "ckpt_path": r"results/training/VariantB/Seed1339/variantB_seed1339_best_SSIM.pt",
        "cfg_overrides": {"use_sobel_input": False},
    },
    "C — + Sobel Input": {
        "mode": "wandb",
        "run_name": "edr_variant_c",
        "ckpt_path": r"results/training/VariantC/Seed1339/variantC_seed1339_best_SSIM.pt",
        "cfg_overrides": {"use_sobel_input": True},
    },
    "D — Full EDR-REDNet (Ours)": {
        "mode": "wandb",
        "run_name": "edr_redcnn_seed2024",
        "ckpt_path": r"results/training/VariantD/seed2024/seed2024_best_SSIM.pt",
        "cfg_overrides": {},
    },
}

# HU windowing theo chuẩn ldct-benchmark (Chest exam)
HU_VMIN = -1024.0
HU_VMAX = 3000.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def setup_wandb_dir(run_name: str, ckpt_path: str, cfg_path: str, overrides: dict = None):
    """Copy checkpoint + config vào fake wandb dir (giống app.py)."""
    d = os.path.join("wandb", run_name, "files")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(ckpt_path) and os.path.exists(cfg_path):
        if overrides:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg.update(overrides)
            with open(os.path.join(d, "args.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(cfg, f)
        else:
            shutil.copy(cfg_path, os.path.join(d, "args.yaml"))
        shutil.copy(ckpt_path, os.path.join(d, "best_SSIM.pt"))
        return True
    else:
        missing = []
        if not os.path.exists(ckpt_path):
            missing.append(f"checkpoint: {ckpt_path}")
        if not os.path.exists(cfg_path):
            missing.append(f"config: {cfg_path}")
        warnings.warn(f"[{run_name}] Không tìm thấy file: {', '.join(missing)}")
        return False


def to_numpy_hu(tensor: torch.Tensor, dataset: TestData) -> np.ndarray:
    """Chuyển tensor (normalized) → numpy HU array."""
    img_np = dataset.denormalize(tensor.cpu().squeeze()).numpy()
    return dataset._convert_hu(img_np, to_hu=True)


def get_edge_map(img_norm: np.ndarray) -> np.ndarray:
    """Tính Sobel edge map của ảnh đã normalize về [0, 1]."""
    edge = np.hypot(ndimage.sobel(img_norm, axis=0), ndimage.sobel(img_norm, axis=1))
    rng = edge.max() - edge.min() + 1e-8
    return (edge - edge.min()) / rng


def normalize_for_metric(img_hu: np.ndarray) -> np.ndarray:
    """Clip và normalize HU về [0, 1] để tính SSIM/PSNR."""
    p = np.clip(img_hu, HU_VMIN, HU_VMAX)
    return (p - HU_VMIN) / (HU_VMAX - HU_VMIN)


def compute_slice_metrics(pred_hu: np.ndarray, target_hu: np.ndarray):
    """
    Tính PSNR, SSIM, và Edge SSIM cho một lát cắt (slice-level).

    Returns
    -------
    dict với keys: PSNR, SSIM, Edge_SSIM
    """
    p_n = normalize_for_metric(pred_hu)
    t_n = normalize_for_metric(target_hu)

    ssim_val = metrics.structural_similarity(t_n, p_n, data_range=1.0)
    psnr_val = metrics.peak_signal_noise_ratio(t_n, p_n, data_range=1.0)

    ep = get_edge_map(p_n)
    et = get_edge_map(t_n)
    edge_ssim_val = metrics.structural_similarity(et, ep, data_range=1.0)

    return {"PSNR": psnr_val, "SSIM": ssim_val, "Edge_SSIM": edge_ssim_val}


# ──────────────────────────────────────────────────────────────────────────────
# 4. LOAD MODELS
# ──────────────────────────────────────────────────────────────────────────────

def load_all_models(device: torch.device):
    """Tải tất cả Best Models lên GPU."""
    print("\n" + "=" * 60)
    print("📦 ĐANG TẢI CÁC MODEL...")
    print("=" * 60)

    # Setup wandb dirs trước
    for vname, vcfg in VARIANT_CONFIG.items():
        if vcfg["mode"] == "wandb":
            ok = setup_wandb_dir(
                run_name=vcfg["run_name"],
                ckpt_path=vcfg["ckpt_path"],
                cfg_path=CFG_PATH,
                overrides=vcfg.get("cfg_overrides", {}),
            )
            status = "✅" if ok else "❌ THIẾU FILE"
            print(f"  [{status}] Variant wandb dir: {vcfg['run_name']}")

    # Load từng model
    networks = {}
    for vname, vcfg in VARIANT_CONFIG.items():
        try:
            if vcfg["mode"] == "hub":
                net = load_model(vcfg["hub_name"], eval=True).to(device)
            else:
                net = setup_trained_model(
                    run_name=vcfg["run_name"],
                    device=device,
                    network_name="Model",
                    state_dict="best_SSIM",
                    eval=True,
                )
            networks[vname] = net
            print(f"  ✅ Loaded: {vname}")
        except Exception as e:
            warnings.warn(f"  ❌ Không load được {vname}: {e}")
            networks[vname] = None

    return networks


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN EVALUATION LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(networks: dict, dataset: TestData, device: torch.device,
                   debug: bool = False):
    """
    Chạy inference qua toàn bộ bệnh nhân, tính metrics từng lát cắt.

    Returns
    -------
    all_scores : dict[variant_name → list of per-patient-mean metrics]
    per_patient_df : DataFrame chứa điểm từng bệnh nhân
    """
    variant_names = [v for v, net in networks.items() if net is not None]

    # Cấu trúc lưu điểm: {vname: {metric: [per_patient_mean_values]}}
    all_scores = {v: {"PSNR": [], "SSIM": [], "Edge_SSIM": []} for v in variant_names}
    per_patient_rows = []

    n_patients = len(dataset.samples)
    if debug:
        n_patients = min(2, n_patients)
        print(f"\n🐛 DEBUG MODE: Chỉ chạy {n_patients} bệnh nhân đầu tiên.\n")
    else:
        print(f"\n🚀 Bắt đầu đánh giá trên {n_patients} bệnh nhân (GPU: {device})...\n")

    total_start = time.time()

    for pat_idx in range(n_patients):
        patient = dataset[pat_idx]
        patient_id = patient["info"]["id"]
        n_slices = patient["info"]["n_slices"]
        exam_type = patient["info"].get("exam_type", "C")

        print(f"  [{pat_idx + 1:3d}/{n_patients}] {patient_id} ({n_slices} slices) ...", end=" ")
        t0 = time.time()

        # Chứa điểm từng slice của bệnh nhân này
        pat_slice_scores = {v: {"PSNR": [], "SSIM": [], "Edge_SSIM": []} for v in variant_names}

        for sl_idx in range(n_slices):
            x_tensor = patient["x"][sl_idx].unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
            y_np = patient["y"][sl_idx].numpy()  # normalized target

            # Target HU
            target_hu = to_numpy_hu(torch.tensor(y_np), dataset)

            # Inference mỗi variant
            with torch.no_grad():
                for vname in variant_names:
                    net = networks[vname]
                    pred_tensor = net(x_tensor)
                    pred_hu = to_numpy_hu(pred_tensor, dataset)
                    sl_m = compute_slice_metrics(pred_hu, target_hu)
                    for mk, mv in sl_m.items():
                        pat_slice_scores[vname][mk].append(mv)

        # Tính mean của tất cả slice trong bệnh nhân này
        row = {"Patient ID": patient_id, "Exam Type": exam_type}
        for vname in variant_names:
            for mk in ["PSNR", "SSIM", "Edge_SSIM"]:
                pat_mean = float(np.mean(pat_slice_scores[vname][mk]))
                all_scores[vname][mk].append(pat_mean)
                col_name = f"{vname.split('—')[0].strip()}_{mk}"
                row[col_name] = round(pat_mean, 6)
        per_patient_rows.append(row)

        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

    total_time = time.time() - total_start
    print(f"\n✅ Hoàn thành! Tổng thời gian: {total_time:.1f}s ({total_time / 60:.1f} phút)")

    per_patient_df = pd.DataFrame(per_patient_rows)
    return all_scores, per_patient_df


# ──────────────────────────────────────────────────────────────────────────────
# 6. STATISTICAL TESTS & SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary_and_stats(all_scores: dict):
    """
    Tính Mean ± Std và chạy Wilcoxon signed-rank test.

    Returns
    -------
    summary_df  : DataFrame với Mean ± Std của từng Variant và Metric
    pvalue_df   : DataFrame với p-value (Variant D vs mỗi variant khác)
    """
    print("\n" + "=" * 60)
    print("📊 TỔNG HỢP KẾT QUẢ & KIỂM ĐỊNH THỐNG KÊ")
    print("=" * 60)

    variant_names = list(all_scores.keys())
    metrics_list = ["PSNR", "SSIM", "Edge_SSIM"]

    # ── Summary Table ────────────────────────────────────────────────────────
    summary_rows = []
    for vname in variant_names:
        row = {"Variant": vname}
        for mk in metrics_list:
            vals = np.array(all_scores[vname][mk])
            mean_v = vals.mean()
            std_v = vals.std()
            row[mk] = f"{mean_v:.4f} ± {std_v:.4f}"
            row[f"{mk}_mean"] = mean_v
            row[f"{mk}_std"] = std_v
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # ── Wilcoxon Test (Variant D vs tất cả còn lại) ─────────────────────────
    target_variant = [v for v in variant_names if "D —" in v or "Full" in v]
    if not target_variant:
        print("  ⚠️  Không tìm thấy Variant D để chạy kiểm định.")
        return summary_df, pd.DataFrame()

    target_variant = target_variant[0]
    print(f"\n  So sánh '{target_variant}' với các Variant khác:\n")

    pvalue_rows = []
    for vname in variant_names:
        if vname == target_variant:
            continue
        row = {"So sánh": f"D vs {vname.split('—')[0].strip()}"}
        for mk in metrics_list:
            d_scores = np.array(all_scores[target_variant][mk])
            v_scores = np.array(all_scores[vname][mk])
            try:
                stat, pval = stats.wilcoxon(d_scores, v_scores, alternative="greater")
                significance = "✅ p<0.05 (có ý nghĩa)" if pval < 0.05 else "❌ p≥0.05 (không có ý nghĩa)"
                row[mk] = f"{pval:.4f}"
                row[f"{mk}_sig"] = significance
                print(f"  {mk:12s} | {vname.split('—')[0].strip():15s} | p={pval:.4f}  {significance}")
            except Exception as e:
                row[mk] = f"Error: {e}"
        pvalue_rows.append(row)

    pvalue_df = pd.DataFrame(pvalue_rows)
    return summary_df, pvalue_df


# ──────────────────────────────────────────────────────────────────────────────
# 7. EXPORT KẾT QUẢ
# ──────────────────────────────────────────────────────────────────────────────

def export_results(per_patient_df: pd.DataFrame, summary_df: pd.DataFrame,
                   pvalue_df: pd.DataFrame, debug: bool = False):
    """Lưu các bảng kết quả ra thư mục results/evaluation/."""
    suffix = "_debug" if debug else ""
    out_dir = os.path.join("results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Điểm từng bệnh nhân
    path_per_patient = os.path.join(out_dir, f"per_patient_scores{suffix}.csv")
    per_patient_df.to_csv(path_per_patient, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Điểm từng bệnh nhân → {path_per_patient}")

    # 2. Bảng summary Mean ± Std
    cols_display = ["Variant", "PSNR", "SSIM", "Edge_SSIM"]
    path_summary = os.path.join(out_dir, f"summary_mean_std{suffix}.csv")
    summary_df[cols_display].to_csv(path_summary, index=False, encoding="utf-8-sig")
    print(f"  💾 Bảng Summary (Mean ± Std) → {path_summary}")

    # 3. Bảng p-value
    if not pvalue_df.empty:
        path_pvalue = os.path.join(out_dir, f"wilcoxon_pvalues{suffix}.csv")
        pvalue_df.to_csv(path_pvalue, index=False, encoding="utf-8-sig")
        print(f"  💾 Bảng p-value (Wilcoxon) → {path_pvalue}")

    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Giai đoạn 3 — Đánh giá thống kê EDR-REDNet trên tập Test"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Chạy chế độ debug (chỉ 2 bệnh nhân đầu tiên để kiểm tra nhanh).",
    )
    args = parser.parse_args()

    # ── Setup device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n🖥️  GPU phát hiện: {gpu_name}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  Không phát hiện GPU. Đang chạy bằng CPU (có thể chậm hơn).")

    # ── Load Dataset ─────────────────────────────────────────────────────────
    print("\n📂 Đang tải TestData...")
    dataset = TestData("data", "meanstd")
    print(f"   Tổng số bệnh nhân: {len(dataset.samples)}")

    # ── Load Models ──────────────────────────────────────────────────────────
    networks = load_all_models(device)

    # Kiểm tra có ít nhất 1 model được load thành công
    loaded = {k: v for k, v in networks.items() if v is not None}
    if not loaded:
        print("\n❌ Không load được model nào. Kiểm tra lại đường dẫn checkpoint và config.")
        sys.exit(1)
    print(f"\n✅ Đã load thành công {len(loaded)}/{len(networks)} models.")

    # ── Evaluation Loop ───────────────────────────────────────────────────────
    all_scores, per_patient_df = run_evaluation(loaded, dataset, device, debug=args.debug)

    # ── Statistical Analysis ─────────────────────────────────────────────────
    summary_df, pvalue_df = compute_summary_and_stats(all_scores)

    # ── Print Summary Table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📋 BẢNG KẾT QUẢ CHÍNH (Mean ± Std):")
    print("=" * 60)
    cols_display = ["Variant", "PSNR", "SSIM", "Edge_SSIM"]
    print(summary_df[cols_display].to_string(index=False))

    # ── Export ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("💾 XUẤT KẾT QUẢ:")
    print("=" * 60)
    out_dir = export_results(per_patient_df, summary_df, pvalue_df, debug=args.debug)

    print(f"\n🎉 HOÀN THÀNH! Tất cả kết quả đã lưu tại: {os.path.abspath(out_dir)}")
    if args.debug:
        print("\n   ⚠️  Đây là kết quả DEBUG (chỉ dùng 2 bệnh nhân).")
        print("   ▶  Chạy không có --debug để lấy kết quả chính thức:\n")
        print("       python evaluate_statistical_test.py\n")


if __name__ == "__main__":
    main()
