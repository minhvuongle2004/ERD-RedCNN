"""
measure_model_stats.py
======================
Giai doan 3 (Script 2) - Do luong Hieu nang Mo hinh (Efficiency)

Muc tieu:
  1. Do so luong tham so (Parameters) cua 4 Variant A, B, C, D.
  2. Tinh toan so phep tinh (MACs / FLOPs) qua thu vien thop/ptflops.
  3. Do Inference Time (ms/anh) tren GPU bang torch.cuda.Event (chinh xac).
  4. In ket qua ra terminal theo dang bang dep.
  5. Luu ket qua ra results/evaluation/model_efficiency.csv.

Cach chay:
  python measure_model_stats.py

Yeu cau:
  pip install thop   (de tinh MACs)
"""

# Fix Unicode encoding on Windows terminal
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import os
import time
import warnings
import argparse
from argparse import Namespace

import torch
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORT MODELS
# ──────────────────────────────────────────────────────────────────────────────
from ldctbench.methods.redcnn.network import Model as REDCNNModel
from ldctbench.methods.edrrednet.network import Model as EDRREDNetModel

# ──────────────────────────────────────────────────────────────────────────────
# 2. VARIANT DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
VARIANT_CONFIGS = {
    "A (RED-CNN)": {
        "model_class": REDCNNModel,
        "args": None,
    },
    "B (No Sobel)": {
        "model_class": EDRREDNetModel,
        "args": Namespace(num_edge_blocks=2, use_sobel_input=False),
    },
    "C (No EdgeBlock)": {
        "model_class": EDRREDNetModel,
        "args": Namespace(num_edge_blocks=0, use_sobel_input=True),
    },
    "D (EDR-REDNet)": {
        "model_class": EDRREDNetModel,
        "args": Namespace(num_edge_blocks=2, use_sobel_input=True),
    },
}

INPUT_SIZE = (1, 1, 512, 512)   # Batch=1, Channel=1, H=512, W=512
WARMUP_RUNS = 20                 # So lan warm up GPU truoc khi do
MEASURE_RUNS = 100               # So lan do de lay trung binh


# ──────────────────────────────────────────────────────────────────────────────
# 3. UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """Dem tong so tham so co the huan luyen."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_total(model: torch.nn.Module) -> int:
    """Dem tong so tham so (ca non-trainable nhu FixedSobelLayer buffer)."""
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return total + buffers


def compute_macs(model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
    """
    Tinh so Multiply-Accumulate Operations (MACs) bang thu vien thop.
    Neu khong co thop, tra ve -1.
    """
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
        return int(macs)
    except ImportError:
        return -1
    except Exception as e:
        warnings.warn(f"Khong tinh duoc MACs: {e}")
        return -1


def measure_inference_time_gpu(model: torch.nn.Module, device: torch.device,
                                input_size=INPUT_SIZE,
                                warmup=WARMUP_RUNS,
                                n_runs=MEASURE_RUNS) -> float:
    """
    Do Inference Time chinh xac tren GPU bang torch.cuda.Event.
    Neu CPU, dung time.perf_counter.

    Returns: thoi gian trung binh (ms/anh)
    """
    model.eval()
    x = torch.randn(input_size).to(device)

    if device.type == 'cuda':
        # GPU timing voi CUDA Events - chinh xac hon time.time()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        # Warm up
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
        torch.cuda.synchronize()

        # Measure
        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                starter.record()
                _ = model(x)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))

        return float(np.mean(timings))

    else:
        # CPU timing
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)

        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(x)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000)  # ms

        return float(np.mean(timings))


def human_readable(n: int) -> str:
    """Chuyen so lon sang dang doc duoc (M, G, T)."""
    if n < 0:
        return "N/A"
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


# ──────────────────────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Setup device ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n GPU phat hien: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("\n Khong co GPU. Do tren CPU (ket qua co the cham hon).")

    print(f" Input size: {INPUT_SIZE} | Warm-up: {WARMUP_RUNS} | Measure: {MEASURE_RUNS} runs")

    # ── Prepare dummy input for MACs ─────────────────────────────────────────
    dummy_input = torch.randn(INPUT_SIZE).to(device)

    # ── Measure each variant ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" DO LUONG HIEU NANG - BAT DAU...")
    print("=" * 70)

    rows = []
    baseline_params = None
    baseline_time = None

    for vname, vcfg in VARIANT_CONFIGS.items():
        print(f"\n  [{vname}]")

        # Init model
        model = vcfg["model_class"](vcfg["args"]).to(device)
        model.eval()

        # Count parameters
        params = count_parameters(model)
        params_total = count_parameters_total(model)
        print(f"    Params (trainable): {human_readable(params)} ({params:,})")
        if params_total != params:
            print(f"    Params (total incl. buffers): {human_readable(params_total)}")

        # Compute MACs
        print(f"    Dang tinh MACs...", end=" ")
        macs = compute_macs(model, dummy_input.clone())
        if macs > 0:
            print(f"{human_readable(macs)} MACs")
        else:
            print("Khong co thop - bo qua MACs")

        # Measure inference time
        print(f"    Dang do Inference Time ({MEASURE_RUNS} runs)...", end=" ")
        infer_ms = measure_inference_time_gpu(model, device)
        print(f"{infer_ms:.2f} ms/anh")

        # Relative to baseline
        if baseline_params is None:
            baseline_params = params
            baseline_time = infer_ms
            rel_params = 1.0
            rel_time = 1.0
        else:
            rel_params = params / baseline_params
            rel_time = infer_ms / baseline_time

        rows.append({
            "Model": vname,
            "Parameters": params,
            "Parameters (Human)": human_readable(params),
            "MACs": macs if macs > 0 else None,
            "MACs (Human)": human_readable(macs) if macs > 0 else "N/A",
            "Inference Time (ms)": round(infer_ms, 2),
            "Relative Params": round(rel_params, 3),
            "Relative Time": round(rel_time, 3),
        })

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Print Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" KET QUA TONG HOP")
    print("=" * 70)
    header = f"  {'Mo hinh':<22} | {'Params':>8} | {'MACs':>8} | {'Time (ms)':>10} | {'Rel.Params':>10} | {'Rel.Time':>9}"
    print(header)
    print("  " + "-" * 68)
    for r in rows:
        macs_str = r["MACs (Human)"]
        line = (f"  {r['Model']:<22} | "
                f"{r['Parameters (Human)']:>8} | "
                f"{macs_str:>8} | "
                f"{r['Inference Time (ms)']:>10.2f} | "
                f"{r['Relative Params']:>10.3f}x | "
                f"{r['Relative Time']:>8.3f}x")
        print(line)
    print("=" * 70)

    # ── Insight ───────────────────────────────────────────────────────────────
    if len(rows) >= 4:
        d = rows[-1]  # Variant D
        a = rows[0]   # Variant A (Baseline)
        param_inc = (d["Parameters"] - a["Parameters"]) / a["Parameters"] * 100
        time_inc = (d["Inference Time (ms)"] - a["Inference Time (ms)"]) / a["Inference Time (ms)"] * 100
        print(f"\n Nhan xet:")
        print(f"   EDR-REDNet tang {param_inc:.1f}% tham so va {time_inc:.1f}% thoi gian xu ly")
        print(f"   so voi RED-CNN baseline - hoan toan chap nhan duoc!")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_dir = os.path.join("results", "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_efficiency.csv")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n Ket qua da luu vao: {os.path.abspath(out_path)}")

    # ── Print paper-ready table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(" BANG DUNG TRUC TIEP CHO BAI BAO:")
    print("=" * 70)
    print(f"  {'Model':<30} | {'Params':>8} | {'Rel.':>6} | {'Time (ms)':>10} | {'Rel.':>6}")
    print("  " + "-" * 68)
    for r in rows:
        print(f"  {r['Model']:<30} | "
              f"{r['Parameters (Human)']:>8} | "
              f"{r['Relative Params']:>5.2f}x | "
              f"{r['Inference Time (ms)']:>10.2f} | "
              f"{r['Relative Time']:>5.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
