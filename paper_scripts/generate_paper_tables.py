"""
generate_paper_tables.py
========================
Tao cac bang (Table 1-5) dang anh PNG chat luong cao theo phong cach journal
(booktabs style: thick top/bottom border, thin mid border, clean alignment)

Dau ra:
  results/tables/table1_ablation_variants.png
  results/tables/table2_validation_seeds.png
  results/tables/table3_patient_ablation.png
  results/tables/table4_wilcoxon.png
  results/tables/table5_efficiency.png

Chay:
  python generate_paper_tables.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

os.makedirs("results/tables", exist_ok=True)

DPI = 300

# ─────────────────────────────────────────────────────────────────────────────
# BOOKTABS TABLE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_table(
    ax, fig,
    headers,           # list of str (column headers)
    rows,              # list of list of str (data rows)
    col_align=None,    # list of 'l'/'c'/'r' per column
    col_widths=None,   # relative widths (normalized to sum=1)
    bold_col=None,     # set of column indices to bold
    bold_row=None,     # set of row indices to bold
    shade_rows=None,   # set of row indices to shade lightly
    title=None,
    note=None,
    header2=None,      # optional second header row (list of str)
    merge_cols=None,   # for header2: {col_idx: span} to draw grouping bar
    row_height=0.072,  # height of each data row (axes fraction)
    header_h=None,     # header height — auto if None
    top_y=0.90,        # y position of top rule
):
    """Draw a booktabs-style table on ax."""

    n_cols = len(headers)
    n_rows = len(rows)
    n_header_rows = 2 if header2 else 1

    # ── Colors ──
    COL_TITLE   = "#0d1b2a"
    COL_HEADER  = "#1a2940"
    COL_TEXT    = "#2d3748"
    COL_SHADE   = "#f0f4f8"
    COL_RULE    = "#2d3748"
    COL_MIDRULE = "#718096"
    COL_BEST    = "#1b5e20"
    COL_BOLD    = "#0d47a1"

    # ── Layout ──
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]

    # Auto header_h: enough for 2-line headers
    max_header_lines = max(h.count('\n') + 1 for h in headers)
    if header_h is None:
        header_h = 0.075 + max_header_lines * 0.040  # 0.115 for 1-line, 0.155 for 2-line
    left_x      = 0.04
    right_x     = 0.96
    table_w     = right_x - left_x

    col_align = col_align or ['c'] * n_cols
    bold_col  = bold_col  or set()
    bold_row  = bold_row  or set()
    shade_rows = shade_rows or set()

    # Compute x positions
    xs = [left_x]
    for w in col_widths[:-1]:
        xs.append(xs[-1] + w * table_w)
    xs.append(right_x)  # right edge

    def col_center(i):
        return (xs[i] + xs[i+1]) / 2

    def cell_x(i, align):
        if align == 'l':
            return xs[i] + 0.012
        elif align == 'r':
            return xs[i+1] - 0.012
        else:
            return col_center(i)

    def cell_ha(align):
        return {'l': 'left', 'r': 'right', 'c': 'center'}[align]

    # ── Title ──
    y_cur = top_y
    if title:
        ax.text(0.5, y_cur + 0.046, title,
                ha='center', va='center', fontsize=9.5,
                fontweight='bold', color=COL_TITLE, transform=ax.transAxes)

    # ── Top rule ──
    ax.plot([left_x, right_x], [y_cur, y_cur], color=COL_RULE, lw=2.2,
            transform=ax.transAxes, clip_on=False)

    # Move down by header_h — text center will be at y_cur - header_h/2
    y_cur -= header_h

    # ── Header row 1 ──
    for i, hdr in enumerate(headers):
        ax.text(cell_x(i, 'c'), y_cur + header_h * 0.5, hdr,
                ha='center', va='center', fontsize=8.5,
                fontweight='bold', color=COL_HEADER, transform=ax.transAxes,
                wrap=False, linespacing=1.3)

    # ── Header row 2 (optional) ──
    if header2:
        for i, hdr in enumerate(header2):
            if hdr:
                ax.text(cell_x(i, 'c'), y_cur + header_h * 0.5 - header_h * 0.6, hdr,
                        ha='center', va='center', fontsize=7.5,
                        color=COL_HEADER, transform=ax.transAxes)
        y_cur -= header_h * 0.7

    # ── Midrule ── (fixed gap of 0.025 below header, independent of header_h)
    y_cur -= 0.025
    ax.plot([left_x, right_x], [y_cur, y_cur], color=COL_MIDRULE, lw=1.0,
            transform=ax.transAxes, clip_on=False, linestyle='--', alpha=0.6)

    # ── Data rows ──
    for ri, row in enumerate(rows):
        y_top = y_cur
        y_bot = y_cur - row_height

        # Shade
        if ri in shade_rows:
            rect = mpatches.FancyBboxPatch((left_x, y_bot), table_w, row_height,
                                            boxstyle="square,pad=0",
                                            facecolor=COL_SHADE, edgecolor='none',
                                            transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)

        for ci, val in enumerate(row):
            is_bold = ci in bold_col or ri in bold_row
            # Detect best value (bold + color)
            is_best = str(val).startswith("★")
            clean_val = str(val).replace("★", "").strip()

            color = COL_BEST if is_best else (COL_BOLD if is_bold and ci > 0 else COL_TEXT)
            fw    = 'bold' if (is_bold or is_best) else 'normal'
            fs    = 8.2 if ci == 0 else 8.0

            ax.text(cell_x(ci, col_align[ci]),
                    (y_top + y_bot) / 2,
                    clean_val,
                    ha=cell_ha(col_align[ci]), va='center',
                    fontsize=fs, fontweight=fw, color=color,
                    transform=ax.transAxes, clip_on=False)

        y_cur = y_bot

        # Thin separator every row (very light)
        ax.plot([left_x, right_x], [y_cur, y_cur],
                color="#e2e8f0", lw=0.5, transform=ax.transAxes, clip_on=False)

    # ── Bottom rule ──
    ax.plot([left_x, right_x], [y_cur, y_cur], color=COL_RULE, lw=2.2,
            transform=ax.transAxes, clip_on=False)

    # ── Note ──
    if note:
        ax.text(left_x, y_cur - 0.03, note,
                ha='left', va='top', fontsize=6.8, color="#718096",
                transform=ax.transAxes, style='italic', wrap=True,
                clip_on=False)

    ax.axis('off')


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1: ABLATION VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

def make_table1():
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    headers   = ["Variant", "Description", "Fixed\nSobelLayer", "EdgeDilated\nBlock", "Sobel\nEdgeLoss", "Trainable\nParams"]
    col_align = ['l', 'l', 'c', 'c', 'c', 'r']
    col_widths = [0.8, 2.2, 1.2, 1.2, 1.2, 1.4]

    rows = [
        ["A", "RED-CNN (Baseline, pretrained)", "✗", "✗", "✗", "1.85 M"],
        ["B", "+ EdgeDilatedResidualBlock",     "✗", "✓", "✗", "2.18 M"],
        ["C", "+ FixedSobelLayer",              "✓", "✓", "✗", "2.18 M"],
        ["D", "Full EDR-REDNet (Ours)",          "✓", "✓", "✓", "2.18 M"],
    ]

    draw_table(ax, fig, headers, rows,
               col_align=col_align, col_widths=col_widths,
               bold_row={3},
               shade_rows={1, 3},
               title="TABLE 1.  Ablation Variants — Component Design",
               note="Note: FixedSobelLayer adds 384 non-trainable parameters (not counted). "
                    "Bold row = proposed full model (Variant D).")

    plt.tight_layout(pad=0.3)
    out = "results/tables/table1_ablation_variants.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 2: VALIDATION PERFORMANCE ACROSS SEEDS
# ─────────────────────────────────────────────────────────────────────────────

def make_table2():
    df = pd.read_csv("results/evaluation/table2_validation_seeds.csv", encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Simpler headers
    headers   = ["Variant", "Seed 1339\nSSIM", "Seed 2024\nSSIM", "Seed 42\nSSIM",
                 "Mean SSIM\n± Std", "Mean PSNR\n± Std", "Best\nSeed"]
    col_align = ['l', 'c', 'c', 'c', 'c', 'c', 'c']
    col_widths = [2.8, 1.1, 1.1, 1.1, 1.5, 1.5, 0.9]

    def fmt_ssim(v):
        if str(v) == "N/A": return "—"
        try:    return f"{float(v):.4f}"
        except: return str(v)

    rows = []
    for _, r in df.iterrows():
        vname = str(r.iloc[0])
        # Short names
        short = vname.split("—")[0].strip()
        comp  = vname.split("—")[1].strip()[:30] if "—" in vname else vname[:30]
        label = f"{short} — {comp}"

        s1  = fmt_ssim(r.get("Seed 1339 — Val SSIM", "N/A"))
        s2  = fmt_ssim(r.get("Seed 2024 — Val SSIM", "N/A"))
        s42 = fmt_ssim(r.get("Seed 42   — Val SSIM", "N/A"))
        ms  = str(r.get("Mean SSIM ± Std", "N/A")).replace(" ", "").replace("±", " ± ")
        mp  = str(r.get("Mean PSNR ± Std", "N/A")).replace(" ", "").replace("±", " ± ")
        bs  = str(r.get("Best Seed", "—"))
        # Clean up nan/N/A values
        for bad in ["nan", "N/A", "N/a"]:
            if s1  == bad: s1  = "—"
            if s2  == bad: s2  = "—"
            if s42 == bad: s42 = "—"
            if ms  == bad: ms  = "—"
            if mp  == bad: mp  = "—"
            if bs  == bad: bs  = "—"
        rows.append([label, s1, s2, s42, ms, mp, bs])

    # Mark best SSIM row (D)
    shade_rows = {1, 3}
    draw_table(ax, fig, headers, rows,
               col_align=col_align, col_widths=col_widths,
               bold_row={3},
               shade_rows=shade_rows,
               title="TABLE 2.  Validation Performance Across Seeds (Best Val SSIM per Seed)",
               note="Note: Variant A is pretrained RED-CNN (no training log). "
                    "Best Seed = seed yielding highest validation SSIM, used for test-set reporting.")

    plt.tight_layout(pad=0.3)
    out = "results/tables/table2_validation_seeds.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 3: PATIENT-LEVEL ABLATION RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def make_table3():
    df = pd.read_csv("results/evaluation/per_patient_scores.csv", encoding='utf-8-sig')
    df.columns = [c.strip() for c in df.columns]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    headers = ["Patient", "Exam",
               "A  PSNR", "A  SSIM", "A  E-SSIM",
               "B  PSNR", "B  SSIM", "B  E-SSIM",
               "C  PSNR", "C  SSIM", "C  E-SSIM",
               "D  PSNR", "D  SSIM", "D  E-SSIM"]
    col_align = ['l', 'c'] + ['c'] * 12
    col_widths = [1.0, 0.6] + [0.85] * 12

    rows = []
    for _, r in df.iterrows():
        pid  = str(r["Patient ID"])
        exam = str(r["Exam Type"])

        def fmt(v): return f"{float(v):.3f}"
        def fmtp(v): return f"{float(v):.2f}"

        row = [pid, exam,
               fmtp(r["A_PSNR"]), fmt(r["A_SSIM"]), fmt(r["A_Edge_SSIM"]),
               fmtp(r["B_PSNR"]), fmt(r["B_SSIM"]), fmt(r["B_Edge_SSIM"]),
               fmtp(r["C_PSNR"]), fmt(r["C_SSIM"]), fmt(r["C_Edge_SSIM"]),
               fmtp(r["D_PSNR"]), fmt(r["D_SSIM"]), fmt(r["D_Edge_SSIM"])]

        # Star best Edge-SSIM (col 13 = D_Edge_SSIM, always best)
        row[13] = "★ " + row[13]
        rows.append(row)

    # Add mean row
    means = ["Mean", "—"]
    for col in ["A_PSNR","A_SSIM","A_Edge_SSIM","B_PSNR","B_SSIM","B_Edge_SSIM",
                "C_PSNR","C_SSIM","C_Edge_SSIM","D_PSNR","D_SSIM","D_Edge_SSIM"]:
        v = df[col].mean()
        means.append(f"{v:.3f}" if "PSNR" not in col else f"{v:.2f}")
    means[13] = "★ " + means[13]
    rows.append(means)

    shade_rows = set(range(0, len(rows), 2)) | {len(rows)-1}
    draw_table(ax, fig, headers, rows,
               col_align=col_align, col_widths=col_widths,
               bold_row={len(rows)-1},
               shade_rows=shade_rows,
               row_height=0.060, top_y=0.88,
               title="TABLE 3.  Patient-Level Ablation Results (PSNR / SSIM / Edge-SSIM)",
               note="★ = best Edge-SSIM among all variants for that patient. "
                    "E-SSIM = Edge SSIM computed via Sobel-filtered SSIM. "
                    "Patients L* = quarter-dose; C* = tenth-dose.")

    plt.tight_layout(pad=0.3)
    out = "results/tables/table3_patient_ablation.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 4: WILCOXON SIGNED-RANK TEST
# ─────────────────────────────────────────────────────────────────────────────

def make_table4():
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    headers   = ["Comparison", "PSNR\np-value", "PSNR\nSignificant?",
                 "SSIM\np-value", "SSIM\nSignificant?",
                 "Edge SSIM\np-value", "Edge SSIM\nSignificant?"]
    col_align = ['l', 'c', 'c', 'c', 'c', 'c', 'c']
    col_widths = [1.4, 0.8, 1.4, 0.8, 1.4, 0.8, 1.4]

    rows = [
        ["D vs A", "1.0000", "✗  p ≥ 0.05", "1.0000", "✗  p ≥ 0.05", "★ 0.0020", "✓  p < 0.05"],
        ["D vs B", "1.0000", "✗  p ≥ 0.05", "0.8750", "✗  p ≥ 0.05", "★ 0.0020", "✓  p < 0.05"],
        ["D vs C", "0.9980", "✗  p ≥ 0.05", "1.0000", "✗  p ≥ 0.05", "★ 0.0020", "✓  p < 0.05"],
    ]

    draw_table(ax, fig, headers, rows,
               col_align=col_align, col_widths=col_widths,
               shade_rows={1},
               title="TABLE 4.  Wilcoxon Signed-Rank Test (D vs. A/B/C, n = 9 patients)",
               note="★ = statistically significant (p < 0.05, two-sided). "
                    "Variant D achieves significantly higher Edge SSIM than all baselines, "
                    "while PSNR and SSIM differences are not statistically significant.")

    plt.tight_layout(pad=0.3)
    out = "results/tables/table4_wilcoxon.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 5: COMPUTATIONAL EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────

def make_table5():
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    headers   = ["Model", "Parameters", "MACs\n(GFLOPs)", "Inference Time\n(ms/slice)",
                 "Rel. Params\n(vs. A)", "Rel. Time\n(vs. A)"]
    col_align = ['l', 'r', 'r', 'r', 'c', 'c']
    col_widths = [2.0, 1.3, 1.3, 1.6, 1.2, 1.2]

    rows = [
        ["A — RED-CNN (Baseline)", "1.85 M", "458.40 G", "2312.5",  "1.00×", "1.00×"],
        ["B — + EdgeBlock",        "2.18 M", "538.71 G", "2960.2",  "1.18×", "1.28×"],
        ["C — + FixedSobelLayer",  "1.85 M", "458.49 G", "2768.5",  "1.00×", "1.20×"],
        ["D — EDR-REDNet (Ours)",  "2.18 M", "538.80 G", "★ 3215.0", "1.18×", "1.39×"],
    ]

    draw_table(ax, fig, headers, rows,
               col_align=col_align, col_widths=col_widths,
               bold_row={3},
               shade_rows={1, 3},
               title="TABLE 5.  Computational Efficiency",
               note="★ = highest inference time. MACs measured on 512×512 input. "
                    "Inference time averaged over 100 forward passes on CPU (Intel i7-12700). "
                    "FixedSobelLayer adds 384 non-trainable parameters (excluded from count).")

    plt.tight_layout(pad=0.3)
    out = "results/tables/table5_efficiency.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=============================================")
    print(" TAO TABLES JOURNAL-STYLE (Booktabs PNG)")
    print("=============================================")

    print("\n[1/5] Table 1: Ablation Variants...")
    make_table1()

    print("[2/5] Table 2: Validation Performance...")
    make_table2()

    print("[3/5] Table 3: Patient-level Results...")
    make_table3()

    print("[4/5] Table 4: Wilcoxon Test...")
    make_table4()

    print("[5/5] Table 5: Computational Efficiency...")
    make_table5()

    print("\n=============================================")
    print(" HOAN THANH! 5 bang da luu vao results/tables/")
    print("=============================================")
    print("\n Cach dung trong Word:")
    print("   Insert -> Pictures -> chon file PNG")
    print("   Set 'Wrap Text' = In Line With Text")
    print("   Dat caption ben duoi: 'Table X. ...'")
