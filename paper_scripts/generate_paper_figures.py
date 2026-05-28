"""
generate_paper_figures.py
=========================
Tao Figure 1 va Figure 2 cho bai bao khoa hoc EDR-REDNet.

Figure 1: Kien truc EDR-REDNet  (2-row U-shape layout)
Figure 2: Ablation Design A-D

Chay:
    python generate_paper_figures.py

Dau ra:
    results/fig1_architecture.png          (nen trang)
    results/fig2_ablation_design.png       (nen trang)
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.linewidth'] = 0.8

os.makedirs("results", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE  (white-background, print-friendly)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    # Figure background & text
    "bg":          "white",
    "title":       "#0d1b2a",
    "subtitle":    "#4a5568",
    "label":       "#2d3748",
    "italic":      "#718096",

    # Main flow
    "input_fc":    "#1565c0",   # LDCT Input / Output  (dark blue)
    "input_ec":    "#0d47a1",
    "enc_fc":      "#0288d1",   # Encoder Conv          (mid blue)
    "enc_ec":      "#01579b",
    "dec_fc":      "#b71c1c",   # Decoder TConv         (dark red)
    "dec_ec":      "#7f0000",
    "edge_fc":     "#2e7d32",   # EdgeDilatedResBlock   (dark green)
    "edge_ec":     "#1b5e20",
    "sobel_fc":    "#6a1b9a",   # FixedSobelLayer       (dark purple)
    "sobel_ec":    "#4a148c",

    # Loss section
    "charb_fc":    "#e65100",   # Charbonnier           (deep orange)
    "charb_ec":    "#bf360c",
    "sobeloss_fc": "#4527a0",   # SobelEdgeLoss         (deep indigo)
    "sobeloss_ec": "#311b92",
    "ltotal_fc":   "#37474f",   # L_total               (blue-gray)
    "ltotal_ec":   "#263238",
    "ndct_fc":     "#546e7a",   # NDCT Target           (slate)
    "ndct_ec":     "#37474f",

    # Arrows
    "arr_main":    "#455a64",
    "arr_edge":    "#2e7d32",
    "arr_sobel":   "#6a1b9a",
    "arr_skip":    "#e65100",
    "arr_loss":    "#90a4ae",

    # Legend / misc
    "legend_bg":   "#f5f7fa",
    "legend_ec":   "#b0bec5",
    "sep_line":    "#b0bec5",
}


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, label,
             facecolor="#1565c0", edgecolor="#0d47a1",
             fontsize=8, radius=0.05, alpha=1.0, textcolor="white"):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}",
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=1.6, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=textcolor, zorder=4)


def arrow(ax, x1, y1, x2, y2, color="#455a64", lw=1.5,
          arrowstyle='->', mutation_scale=12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle=arrowstyle,
                                 color=color, lw=lw,
                                 mutation_scale=mutation_scale),
                 zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: ARCHITECTURE  (white bg, 2-row U-shape)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure1():
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(6.5, 9.65, "EDR-REDNet Architecture",
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=C["title"])
    ax.text(6.5, 9.3, "Edge-Dilated Residual RED-CNN for Low-Dose CT Denoising",
            ha='center', va='center', fontsize=11, color=C["subtitle"])

    # ── Layout constants ────────────────────────────────────────────────────
    ENC_Y  = 7.2
    DEC_Y  = 4.5
    BOX_W  = 1.35
    BOX_H  = 0.82
    ENC_XS = [1.0, 2.7, 4.4, 6.1, 7.8, 9.5]   # Input + Conv1..5
    DEC_XS = [9.5, 7.8, 6.1, 4.4, 2.7, 1.0]   # TConv1..5 + Output
    EDGE_X = 11.4
    EDGE_Y1 = 7.2
    EDGE_Y2 = 5.85

    # ── FIXED SOBEL LAYER ──────────────────────────────────────────────────
    sobel_x, sobel_y = 1.0, 8.6
    draw_box(ax, sobel_x, sobel_y, 1.55, 0.68, "Fixed\nSobel Layer",
             facecolor=C["sobel_fc"], edgecolor=C["sobel_ec"], fontsize=10)
    ax.text(1.9, 8.6, "non-trainable · 4-direction edge maps\n(H, V, Diag45°, Diag135°)",
            ha='left', va='center', fontsize=8.5, color=C["italic"], style='italic')

    arrow(ax, ENC_XS[0], ENC_Y + BOX_H/2, ENC_XS[0], sobel_y - 0.34,
          color=C["arr_sobel"])
    ax.annotate("",
                xy=(EDGE_X - BOX_W/2 - 0.05, (EDGE_Y1 + EDGE_Y2)/2),
                xytext=(sobel_x + 0.78, sobel_y),
                arrowprops=dict(arrowstyle='->', color=C["arr_sobel"],
                                lw=1.5, connectionstyle='arc3,rad=-0.15',
                                mutation_scale=12),
                zorder=5)
    ax.text(6.9, 8.88, "+ edge features  (additive fusion at bottleneck)",
            ha='center', va='center', fontsize=8.5, color=C["italic"], style='italic')

    # ── ENCODER ROW ────────────────────────────────────────────────────────
    enc_labels = ["LDCT\nInput", "Conv1\n64f, 5×5", "Conv2\n64f, 5×5",
                  "Conv3\n64f, 5×5", "Conv4\n64f, 5×5", "Conv5\n64f, 5×5"]
    enc_fc = [C["input_fc"]] + [C["enc_fc"]] * 5
    enc_ec = [C["input_ec"]] + [C["enc_ec"]] * 5

    for i, (x, lbl, fc, ec) in enumerate(zip(ENC_XS, enc_labels, enc_fc, enc_ec)):
        draw_box(ax, x, ENC_Y, BOX_W, BOX_H, lbl, facecolor=fc, edgecolor=ec, fontsize=9.5)
        if i > 0:
            arrow(ax, ENC_XS[i-1] + BOX_W/2, ENC_Y, x - BOX_W/2, ENC_Y,
                  color=C["arr_main"])

    ax.text(5.25, ENC_Y - BOX_H/2 - 0.26,
            "─── Encoder  (RED-CNN Backbone: Conv 5×5 ×5) ───",
            ha='center', fontsize=9, fontweight='bold', color=C["enc_ec"], style='italic')

    # ── BOTTLENECK: EDGE-DILATED BLOCKS ────────────────────────────────────
    draw_box(ax, EDGE_X, EDGE_Y1, 1.45, BOX_H, "EdgeBlock\nd = 2",
             facecolor=C["edge_fc"], edgecolor=C["edge_ec"], fontsize=11)
    draw_box(ax, EDGE_X, EDGE_Y2, 1.45, BOX_H, "EdgeBlock\nd = 3",
             facecolor=C["edge_fc"], edgecolor=C["edge_ec"], fontsize=11)
    ax.text(EDGE_X, EDGE_Y2 - BOX_H/2 - 0.26,
            "Edge-Dilated\nResidual Blocks",
            ha='center', fontsize=8.5, fontweight='bold', color=C["edge_ec"],
            style='italic', linespacing=1.3)

    arrow(ax, ENC_XS[-1] + BOX_W/2, ENC_Y, EDGE_X - BOX_W/2 - 0.02, EDGE_Y1,
          color=C["arr_main"])
    arrow(ax, EDGE_X, EDGE_Y1 - BOX_H/2, EDGE_X, EDGE_Y2 + BOX_H/2,
          color=C["arr_edge"])

    # ── DECODER ROW ────────────────────────────────────────────────────────
    dec_labels = ["TConv1\n64f, 5×5", "TConv2\n64f, 5×5", "TConv3\n64f, 5×5",
                  "TConv4\n64f, 5×5", "TConv5\n1f, 5×5", "Output\n(Denoised)"]
    dec_fc = [C["dec_fc"]] * 5 + [C["input_fc"]]
    dec_ec = [C["dec_ec"]] * 5 + [C["input_ec"]]

    for i, (x, lbl, fc, ec) in enumerate(zip(DEC_XS, dec_labels, dec_fc, dec_ec)):
        draw_box(ax, x, DEC_Y, BOX_W, BOX_H, lbl, facecolor=fc, edgecolor=ec, fontsize=9.5)
        if i > 0:
            arrow(ax, DEC_XS[i-1] - BOX_W/2, DEC_Y, x + BOX_W/2, DEC_Y,
                  color=C["arr_main"])

    ax.text(5.25, DEC_Y - BOX_H/2 - 0.26,
            "─── Decoder  (RED-CNN Backbone: TConv 5×5 ×5) ───",
            ha='center', fontsize=9, fontweight='bold', color=C["dec_ec"], style='italic')

    arrow(ax, EDGE_X, EDGE_Y2 - BOX_H/2, EDGE_X, DEC_Y + BOX_H/2,
          color=C["arr_edge"])
    arrow(ax, EDGE_X - BOX_W/2 - 0.02, DEC_Y, DEC_XS[0] + BOX_W/2, DEC_Y,
          color=C["arr_edge"])

    # ── GLOBAL RESIDUAL SKIP ───────────────────────────────────────────────
    ax.annotate("",
                xy=(DEC_XS[-1], DEC_Y - BOX_H/2 - 0.02),
                xytext=(ENC_XS[0], ENC_Y - BOX_H/2 - 0.02),
                arrowprops=dict(arrowstyle='->', color=C["arr_skip"],
                                lw=1.8, connectionstyle='arc3,rad=0.20',
                                mutation_scale=13),
                zorder=5)
    ax.text(5.25, 3.4, "Global Residual Skip   (Input ⊕ Output)",
            ha='center', va='center', fontsize=9.5, color=C["arr_skip"],
            style='italic', fontweight='bold')

    # ── LOSS SECTION ───────────────────────────────────────────────────────
    LOSS_Y = 2.0
    draw_box(ax, 1.0,  LOSS_Y, 1.65, 0.68, "L_total\n= L_c + α·L_s",
             facecolor=C["ltotal_fc"], edgecolor=C["ltotal_ec"], fontsize=9.5)
    draw_box(ax, 3.6,  LOSS_Y, 1.65, 0.68, "Sobel\nEdge Loss",
             facecolor=C["sobeloss_fc"], edgecolor=C["sobeloss_ec"], fontsize=9.5)
    draw_box(ax, 6.2,  LOSS_Y, 1.65, 0.68, "Charbonnier\nLoss",
             facecolor=C["charb_fc"], edgecolor=C["charb_ec"], fontsize=9.5)
    draw_box(ax, 9.0,  LOSS_Y, 1.65, 0.68, "NDCT\nTarget (GT)",
             facecolor=C["ndct_fc"], edgecolor=C["ndct_ec"], fontsize=9.5)

    arrow(ax, DEC_XS[-1], DEC_Y - BOX_H/2, DEC_XS[-1], LOSS_Y + 0.34,
          color=C["input_ec"])
    arrow(ax, 8.17, LOSS_Y, 7.03, LOSS_Y, color=C["arr_loss"])
    arrow(ax, 5.38, LOSS_Y, 4.43, LOSS_Y, color=C["charb_ec"])
    arrow(ax, 2.78, LOSS_Y, 1.83, LOSS_Y, color=C["sobeloss_ec"])
    arrow(ax, 1.0, LOSS_Y - 0.34, 1.0, 1.28, color=C["ltotal_ec"])
    ax.text(1.0, 1.1, "Backprop", ha='center', fontsize=8.5,
            color=C["italic"], style='italic')

    # Formula box
    ax.text(5.2, LOSS_Y - 0.74,
            "L_total  =  L_Charbonnier(pred, target)  +  α × L_Sobel(pred, target),   α = 0.1",
            ha='center', va='center', fontsize=9.5, color=C["label"],
            bbox=dict(facecolor='#f0f4f8', edgecolor=C["sep_line"],
                      boxstyle='round,pad=0.35', alpha=1.0))

    # ── SECTION BARS (left margin) ─────────────────────────────────────────
    for ybot, ytop, color in [
        (ENC_Y - BOX_H/2, ENC_Y + BOX_H/2, C["enc_fc"]),
        (DEC_Y - BOX_H/2, DEC_Y + BOX_H/2, C["dec_fc"]),
    ]:
        ax.plot([-0.08, -0.08], [ybot, ytop], color=color, lw=3.5, zorder=4,
                solid_capstyle='round')

    # ── LEGEND ─────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(facecolor=C["input_fc"],    edgecolor=C["input_ec"],    label='Input / Output'),
        mpatches.Patch(facecolor=C["enc_fc"],      edgecolor=C["enc_ec"],      label='Encoder Conv (RED-CNN)'),
        mpatches.Patch(facecolor=C["dec_fc"],      edgecolor=C["dec_ec"],      label='Decoder TConv (RED-CNN)'),
        mpatches.Patch(facecolor=C["edge_fc"],     edgecolor=C["edge_ec"],     label='EdgeDilatedResidualBlock  ★ NEW'),
        mpatches.Patch(facecolor=C["sobel_fc"],    edgecolor=C["sobel_ec"],    label='FixedSobelLayer  ★ NEW (non-trainable)'),
        mpatches.Patch(facecolor=C["sobeloss_fc"], edgecolor=C["sobeloss_ec"], label='SobelEdgeLoss  ★ NEW'),
        mpatches.Patch(facecolor=C["charb_fc"],    edgecolor=C["charb_ec"],    label='Charbonnier Loss'),
    ]
    ax.legend(handles=legend_items, loc='lower right',
              bbox_to_anchor=(1.01, 0.0), ncol=2,
              fontsize=9, facecolor=C["legend_bg"], edgecolor=C["legend_ec"],
              labelcolor=C["label"], framealpha=1.0)

    plt.tight_layout(pad=0.5)
    out_path = os.path.join("results", "fig1_architecture.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=C["bg"])
    plt.close(fig)
    print(f"  [OK] Figure 1 da luu: {os.path.abspath(out_path)}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: ABLATION DESIGN  (white bg)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure2():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    ax.text(6, 5.72, "Ablation Study Design: Variants A – D",
            ha='center', va='center', fontsize=16, fontweight='bold',
            color=C["title"])
    ax.text(6, 5.38, "Each variant adds one component to the previous, isolating individual contributions",
            ha='center', va='center', fontsize=10.5, color=C["subtitle"])

    # ── Column headers ──────────────────────────────────────────────────────
    headers = ["Variant", "FixedSobelLayer\n(non-trainable)",
               "EdgeDilated\nResidualBlock", "SobelEdgeLoss\n(α·L_s)", "Description"]
    col_x   = [1.1, 3.4, 5.7, 7.9, 10.3]
    col_w   = [1.4, 1.8, 1.8, 1.8, 2.8]

    for hdr, cx in zip(headers, col_x):
        ax.text(cx, 4.78, hdr, ha='center', va='center',
                fontsize=10.5, fontweight='bold', color=C["title"])
    ax.plot([0.2, 11.8], [4.46, 4.46], color=C["sep_line"], lw=1.5)

    # ── Variant rows ────────────────────────────────────────────────────────
    VARIANTS = [
        {"name": "A", "title": "RED-CNN\n(Baseline)",
         "row_fc": "#f0f4f8", "row_ec": "#9aabb8", "text_vc": C["title"],
         "sobel": False, "edgeblock": False, "sobeloss": False,
         "desc": "Original RED-CNN\npretrained on Mayo LDCT\n(hub checkpoint)", "y": 3.70},
        {"name": "B", "title": "+ EdgeBlock",
         "row_fc": "#e3f2fd", "row_ec": "#1565c0", "text_vc": "#0d47a1",
         "sobel": False, "edgeblock": True, "sobeloss": False,
         "desc": "RED-CNN + EdgeDilated\nResidualBlocks (d=2,3)\nat bottleneck", "y": 2.75},
        {"name": "C", "title": "+ Sobel Input",
         "row_fc": "#f3e5f5", "row_ec": "#6a1b9a", "text_vc": "#4a148c",
         "sobel": True, "edgeblock": True, "sobeloss": False,
         "desc": "Variant B +\nFixedSobelLayer input\n(zero param cost)", "y": 1.80},
        {"name": "D", "title": "Full EDR-REDNet\n(Ours)",
         "row_fc": "#e8f5e9", "row_ec": "#2e7d32", "text_vc": "#1b5e20",
         "sobel": True, "edgeblock": True, "sobeloss": True,
         "desc": "Variant C +\nSobelEdgeLoss\n(full model)", "y": 0.85},
    ]
    ROW_H = 0.70

    for v in VARIANTS:
        y = v["y"]
        is_ours = v["name"] == "D"

        # Row background
        lw_bg = 2.2 if is_ours else 1.0
        bg = FancyBboxPatch((0.2, y - ROW_H/2 - 0.02), 11.6, ROW_H + 0.04,
                             boxstyle="round,pad=0.04",
                             facecolor=v["row_fc"], edgecolor=v["row_ec"],
                             linewidth=lw_bg, alpha=1.0, zorder=2)
        ax.add_patch(bg)

        # Variant letter
        ax.text(col_x[0], y + 0.06, v["name"], ha='center', va='center',
                fontsize=20, fontweight='bold', color=v["text_vc"], zorder=3)
        ax.text(col_x[0], y - 0.19, v["title"], ha='center', va='top',
                fontsize=8.5, color=v["text_vc"], zorder=3, linespacing=1.2,
                fontweight='bold')

        # Component cells
        for cx, comp in zip(col_x[1:4], [v["sobel"], v["edgeblock"], v["sobeloss"]]):
            if comp:
                cell = FancyBboxPatch((cx - 0.65, y - 0.27), 1.3, 0.54,
                                       boxstyle="round,pad=0.06",
                                       facecolor="#c8e6c9", edgecolor="#388e3c",
                                       linewidth=1.5, alpha=1.0, zorder=3)
                ax.add_patch(cell)
                ax.text(cx, y, "✓", ha='center', va='center',
                        fontsize=22, color="#1b5e20", fontweight='bold', zorder=4)
            else:
                cell = FancyBboxPatch((cx - 0.65, y - 0.27), 1.3, 0.54,
                                       boxstyle="round,pad=0.06",
                                       facecolor="#fafafa", edgecolor="#cfd8dc",
                                       linewidth=1.0, alpha=1.0, zorder=3)
                ax.add_patch(cell)
                ax.text(cx, y, "✗", ha='center', va='center',
                        fontsize=20, color="#b0bec5", fontweight='bold', zorder=4)

        # Description
        ax.text(col_x[4], y, v["desc"], ha='center', va='center',
                fontsize=9, color=C["label"], zorder=3, linespacing=1.4,
                fontweight='bold')

        # "OURS" badge
        if is_ours:
            badge = FancyBboxPatch((10.85, y + 0.14), 0.80, 0.30,
                                    boxstyle="round,pad=0.05",
                                    facecolor=C["edge_fc"], edgecolor=C["edge_ec"],
                                    linewidth=1.3, zorder=5)
            ax.add_patch(badge)
            ax.text(11.25, y + 0.29, "OURS", ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white', zorder=6)

    # ── Cumulative arrows ───────────────────────────────────────────────────
    arrow_x = 0.42
    ys = [v["y"] for v in VARIANTS]
    for i in range(len(ys) - 1):
        ax.annotate("",
                    xy=(arrow_x, ys[i+1] + ROW_H/2 + 0.05),
                    xytext=(arrow_x, ys[i] - ROW_H/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', color=C["arr_skip"],
                                    lw=1.8, mutation_scale=12),
                    zorder=6)
        ax.text(arrow_x + 0.13, (ys[i] + ys[i+1]) / 2,
                "+1\ncomp.", ha='left', va='center',
                fontsize=8, color=C["arr_skip"], style='italic', fontweight='bold')

    # ── Column separator lines ──────────────────────────────────────────────
    for cx, cw in zip(col_x[1:4], col_w[1:4]):
        ax.plot([cx - cw/2 - 0.3, cx - cw/2 - 0.3], [0.3, 4.46],
                color=C["sep_line"], lw=0.8, zorder=1, linestyle='--')

    # ── Bottom note ─────────────────────────────────────────────────────────
    ax.text(6, 0.2,
            "Each variant trained with 3 random seeds (1339, 2024, 42). "
            "Best model selected per variant based on validation SSIM.",
            ha='center', va='center', fontsize=9, color=C["subtitle"],
            style='italic')

    plt.tight_layout(pad=0.3)
    out_path = os.path.join("results", "fig2_ablation_design.png")
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=C["bg"])
    plt.close(fig)
    print(f"  [OK] Figure 2 da luu: {os.path.abspath(out_path)}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n========================================")
    print(" TAO FIGURE CHO BAI BAO EDR-REDNet")
    print(" (nen trang / white background)")
    print("========================================")

    print("\n[1/2] Dang tao Figure 1 (Kien truc)...")
    p1 = make_figure1()

    print("\n[2/2] Dang tao Figure 2 (Ablation Design)...")
    p2 = make_figure2()

    print("\n========================================")
    print(" HOAN THANH!")
    print("========================================")
    print(f"  Figure 1: {p1}")
    print(f"  Figure 2: {p2}")
    print("\n  Copy 2 file nay vao Word / LaTeX la xong.")
