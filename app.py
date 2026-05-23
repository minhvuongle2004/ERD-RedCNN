import os
import sys
import shutil
import yaml
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pydicom
import pandas as pd
from skimage import filters, metrics
from scipy import ndimage

st.set_page_config(page_title="EDR-REDNet Visualizer", layout="wide")

# ==========================================
# 1. SETUP ENVIRONMENT & PATCHES
# ==========================================
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return original_load(*args, **kwargs)
torch.load = safe_load

def safe_load_yaml(path: str):
    with open(path, encoding='utf-8') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

import ldctbench.evaluate.utils
import ldctbench.utils
ldctbench.evaluate.utils.torch.load = safe_load
ldctbench.evaluate.utils.load_yaml = safe_load_yaml
ldctbench.utils.load_yaml = safe_load_yaml

from ldctbench.data import TestData
from ldctbench.evaluate import setup_trained_model
from ldctbench.hub import load_model

# ==========================================
# 2. SETUP CACHING & MODELS
# ==========================================
def _setup_variant_wandb(run_name, ckpt_path, cfg_path, overrides=None):
    """Copy checkpoint + config vào fake wandb dir để setup_trained_model dùng được."""
    d = os.path.join("wandb", run_name, "files")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(ckpt_path) and os.path.exists(cfg_path):
        if overrides:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg.update(overrides)
            with open(os.path.join(d, "args.yaml"), 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f)
        else:
            shutil.copy(cfg_path, os.path.join(d, "args.yaml"))
        shutil.copy(ckpt_path, os.path.join(d, "best_SSIM.pt"))
        return True
    return False

@st.cache_resource
def setup_environment():
    # Variant D
    checkpoint_path = r"results\training\seed2024\lan1\seed2024_best_SSIM.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = r"results\training\VariantD\seed2024\seed2024_best_SSIM.pt"
    cfg = r"configs\edrrednet.yaml"
    _setup_variant_wandb("edr_redcnn_seed2024", checkpoint_path, cfg)
    # Variant B
    _setup_variant_wandb("edr_variant_b",
        r"results\training\VariantB\Seed1339\variantB_seed1339_best_SSIM.pt", cfg,
        overrides={"use_sobel_input": False})
    # Variant C
    _setup_variant_wandb("edr_variant_c",
        r"results\training\VariantC\Seed1339\variantC_seed1339_best_SSIM.pt", cfg,
        overrides={"use_sobel_input": True})
    return checkpoint_path

@st.cache_resource
def load_dataset():
    return TestData("data", "meanstd")

@st.cache_resource
def load_networks():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    networks = {}
    # Variant A — RED-CNN pretrained
    networks["redcnn"] = load_model("redcnn", eval=True).to(dev)
    networks["variant_a"] = networks["redcnn"]
    # Variant D — Full EDR-REDNet
    net_d = setup_trained_model(run_name="edr_redcnn_seed2024", device=dev,
        network_name="Model", state_dict="best_SSIM", eval=True)
    networks["edr_redcnn"] = net_d
    networks["variant_d"] = net_d
    # Variant B
    try:
        networks["variant_b"] = setup_trained_model(run_name="edr_variant_b", device=dev,
            network_name="Model", state_dict="best_SSIM", eval=True)
    except Exception:
        networks["variant_b"] = None
    # Variant C
    try:
        networks["variant_c"] = setup_trained_model(run_name="edr_variant_c", device=dev,
            network_name="Model", state_dict="best_SSIM", eval=True)
    except Exception:
        networks["variant_c"] = None
    return networks, dev

ckpt_path = setup_environment()
if not os.path.exists(ckpt_path):
    st.error(f"❌ Không tìm thấy file trọng số tại {ckpt_path}. Vui lòng kiểm tra lại.")
    st.stop()

dataset = load_dataset()
networks, device = load_networks()

# ==========================================
# 3. MAIN LAYOUT — TABS
# ==========================================
st.title("🔬 EDR-REDNet: Interactive Evaluation")
tab_infer, tab_ablation = st.tabs(["🖼️ So sánh Mô hình", "🧪 Ablation Study"])

# ==========================================
# TAB 2: ABLATION STUDY
# ==========================================
# =========================================================
# GLOBAL UTILITIES (dùng ở cả 2 tab)
# =========================================================
def window_image(img, vmin, vmax):
    return (np.clip(img, vmin, vmax) - vmin) / (vmax - vmin)

def get_edge_map(img):
    return np.hypot(ndimage.sobel(img, axis=0), ndimage.sobel(img, axis=1))

def calc_metrics(pred, target, roi_target=None, roi_bg=None):
    from sewar.full_ref import vifp
    vmin, vmax = -1024.0, 3000.0
    p_n = (np.clip(pred, vmin, vmax) - vmin) / (vmax - vmin)
    t_n = (np.clip(target, vmin, vmax) - vmin) / (vmax - vmin)
    ssim_v = metrics.structural_similarity(t_n, p_n, data_range=1.0)
    psnr_v = metrics.peak_signal_noise_ratio(t_n, p_n, data_range=1.0)
    vif_v  = vifp(t_n, p_n)
    ep = get_edge_map(p_n); et = get_edge_map(t_n)
    ep = (ep - ep.min()) / (ep.max() - ep.min() + 1e-8)
    et = (et - et.min()) / (et.max() - et.min() + 1e-8)
    edge_ssim = metrics.structural_similarity(et, ep, data_range=1.0)
    
    res = {"SSIM": ssim_v, "PSNR": psnr_v, "VIF": vif_v, "Edge SSIM": edge_ssim}
    
    # Tính CNR và HU Deviation nếu có ROI
    if roi_target and roi_bg:
        tx, ty, tw, th = roi_target
        bx, by, bw, bh = roi_bg
        
        pred_roi_t = pred[ty:ty+th, tx:tx+tw]
        pred_roi_b = pred[by:by+bh, bx:bx+bw]
        
        mean_t = np.mean(pred_roi_t)
        mean_b = np.mean(pred_roi_b)
        std_b = np.std(pred_roi_b) + 1e-8
        
        cnr = abs(mean_t - mean_b) / std_b
        res["CNR"] = cnr
        res["HU Dev (Bg)"] = std_b
        
    return res

def to_numpy_hu(tensor, ds):
    img_np = ds.denormalize(tensor.cpu().squeeze()).numpy()
    return ds._convert_hu(img_np, to_hu=True)

# =========================================================
with tab_ablation:
    st.header("🧪 Ablation Study — Phân tích đóng góp từng thành phần")
    st.markdown("So sánh 4 biến thể kiến trúc để chứng minh vai trò từng thành phần trong EDR-REDNet.")

    # ---- Live Inference Section ----
    st.markdown("---")
    st.subheader("🔬 So sánh Trực tiếp trên Lát cắt (Live Inference)")

    VARIANT_META = {
        "A — RED-CNN (Baseline)":     {"key": "variant_a", "color": "#888888"},
        "B — + EdgeBlock":            {"key": "variant_b", "color": "#4e9af1"},
        "C — + Sobel Input":          {"key": "variant_c", "color": "#f1a74e"},
        "D — Full EDR-REDNet (Ours)": {"key": "variant_d", "color": "#2ecc71"},
    }

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        patient_names_abl = [p["info"]["id"] for p in dataset.samples]
        abl_pat = st.selectbox("Chọn bệnh nhân", range(len(patient_names_abl)),
                               format_func=lambda i: patient_names_abl[i], key="abl_pat")
    abl_batch = dataset[abl_pat]
    n_sl = abl_batch["info"]["n_slices"]
    with col_ctrl2:
        abl_sl = st.slider("Chọn lát cắt", 0, n_sl - 1, n_sl // 2, key="abl_sl")

    abl_hu_min = st.sidebar.slider("Ablation Min HU", -1024, 1024, -160, key="abl_hmin")
    abl_hu_max = st.sidebar.slider("Ablation Max HU", -1024, 3000, 245, key="abl_hmax")

    x_abl = abl_batch["x"][abl_sl].unsqueeze(0).unsqueeze(0).to(device)
    y_abl_np = abl_batch["y"][abl_sl].numpy()

    # ---- ROI Selection UI ----
    st.markdown("### 🎯 Chọn Vùng Quan Tâm (ROI) để tính CNR")
    st.markdown("Kéo các thanh trượt bên dưới để chọn 2 vùng: **Vùng Mục tiêu** (nốt phổi, mạch máu) và **Vùng Nền** (mô mềm đồng nhất, không khí).")
    
    col_roi_img, col_roi_sliders = st.columns([1, 2])
    
    with col_roi_sliders:
        st.write("🔴 **Vùng Mục tiêu (Target)** - Dùng để lấy Tín hiệu (Signal)")
        t_x = st.slider("Target X", 0, 512, 200)
        t_y = st.slider("Target Y", 0, 512, 250)
        t_s = st.slider("Target Size", 5, 100, 20)
        
        st.write("🔵 **Vùng Nền (Background)** - Dùng để đo Nhiễu (Noise)")
        b_x = st.slider("Bg X", 0, 512, 100)
        b_y = st.slider("Bg Y", 0, 512, 250)
        b_s = st.slider("Bg Size", 5, 100, 30)

    roi_target = (t_x, t_y, t_s, t_s)
    roi_bg = (b_x, b_y, b_s, b_s)

    with st.spinner("Đang chạy inference trên 4 Variant..."):
        abl_imgs = {}
        with torch.no_grad():
            abl_imgs["LDCT (Input)"] = to_numpy_hu(x_abl, dataset)
            abl_imgs["NDCT (Ref)"]   = to_numpy_hu(torch.tensor(y_abl_np), dataset)
            for vname, vmeta in VARIANT_META.items():
                net = networks.get(vmeta["key"])
                if net is not None:
                    abl_imgs[vname] = to_numpy_hu(net(x_abl), dataset)
                else:
                    abl_imgs[vname] = None
                    
    with col_roi_img:
        import matplotlib.patches as patches
        fig_roi, ax_roi = plt.subplots(figsize=(4, 4))
        ax_roi.imshow(window_image(abl_imgs["NDCT (Ref)"], abl_hu_min, abl_hu_max), cmap="gray")
        
        # Vẽ ROI Target (Đỏ)
        rect_t = patches.Rectangle((t_x, t_y), t_s, t_s, linewidth=1.5, edgecolor='red', facecolor='none')
        ax_roi.add_patch(rect_t)
        
        # Vẽ ROI Background (Xanh)
        rect_b = patches.Rectangle((b_x, b_y), b_s, b_s, linewidth=1.5, edgecolor='blue', facecolor='none')
        ax_roi.add_patch(rect_b)
        
        ax_roi.axis("off")
        ax_roi.set_title("Bản đồ ROI trên NDCT")
        st.pyplot(fig_roi, use_container_width=True)

    img_ndct_abl = abl_imgs["NDCT (Ref)"]

    # Compute metrics for each variant vs NDCT
    abl_metrics = {}
    for vname in VARIANT_META:
        if abl_imgs[vname] is not None:
            abl_metrics[vname] = calc_metrics(abl_imgs[vname], img_ndct_abl, roi_target, roi_bg)

    # Find best value per metric
    metric_keys = ["SSIM", "PSNR", "VIF", "Edge SSIM", "CNR", "HU Dev (Bg)"]
    best_vals = {}
    for mk in metric_keys:
        vals = [abl_metrics[v][mk] for v in abl_metrics]
        if not vals:
            continue
        if mk == "HU Dev (Bg)":
            best_vals[mk] = min(vals) # HU dev nhỏ nhất là tốt nhất (ít nhiễu nhất)
        else:
            best_vals[mk] = max(vals)

    # --- Display: LDCT | A | B | C | D | NDCT ---
    col_labels = ["LDCT (Input)"] + list(VARIANT_META.keys()) + ["NDCT (Ref)"]
    cols = st.columns(len(col_labels))

    for col, label in zip(cols, col_labels):
        with col:
            img = abl_imgs.get(label)
            if img is None:
                st.markdown(f"**{label.split('—')[0].strip()}**")
                st.info("⏳ Chưa có model")
                continue

            # Image
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(window_image(img, abl_hu_min, abl_hu_max), cmap="gray")
            ax.axis("off")
            short = label.split("—")[0].strip() if "—" in label else label
            ax.set_title(short, fontsize=9, pad=3)
            plt.tight_layout(pad=0.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Metrics
            if label in abl_metrics:
                m = abl_metrics[label]
                for mk in metric_keys:
                    val = m[mk]
                    is_best = (best_vals.get(mk) is not None and abs(val - best_vals[mk]) < 1e-9)
                    delta_str = " 🏆" if is_best else ""
                    fmt = f"{val:.4f}" if mk != "PSNR" else f"{val:.2f} dB"
                    st.markdown(f"<small>**{mk}:** {fmt}{delta_str}</small>", unsafe_allow_html=True)

    # Metrics table
    st.markdown("---")
    st.subheader("📊 Bảng So sánh Metrics theo Lát cắt & ROI")
    if abl_metrics:
        rows = []
        for vname, m in abl_metrics.items():
            row = {"Biến thể": vname.split("—")[0].strip() + " " + (vname.split("—")[1].strip() if "—" in vname else "")}
            for mk in metric_keys:
                val = m[mk]
                is_best = best_vals.get(mk) is not None and abs(val - best_vals[mk]) < 1e-9
                row[mk] = f"{'⭐ ' if is_best else ''}{val:.4f}" if mk != "PSNR" else f"{'⭐ ' if is_best else ''}{val:.2f}"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Radar/bar chart
    if abl_metrics:
        st.subheader("📈 Biểu đồ So sánh Metrics")
        fig_bar, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        colors_list = [VARIANT_META[v]["color"] for v in abl_metrics]
        v_labels = [v.split("—")[0].strip() for v in abl_metrics]
        for ax, mk in zip(axes, metric_keys):
            vals = [abl_metrics[v][mk] for v in abl_metrics]
            bars = ax.bar(v_labels, vals, color=colors_list, width=0.6)
            ax.set_title(mk, fontsize=11, fontweight="bold")
            # Tự động set y limits để thấy rõ sự khác biệt
            min_val = min(vals)
            max_val = max(vals)
            ax.set_ylim(min_val - (max_val - min_val)*0.1, max_val + (max_val - min_val)*0.1)
            ax.tick_params(axis='x', labelsize=8, rotation=15)
            ax.grid(axis='y', alpha=0.3)
            # Highlight best
            best_idx = vals.index(min(vals)) if mk == "HU Dev (Bg)" else vals.index(max(vals))
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(3)
        plt.tight_layout()
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    st.markdown("---")

# ==========================================
# TAB 1: SO SÁNH MÔ HÌNH (inference)
# ==========================================
with tab_infer:
    st.sidebar.header("🕹️ Điều khiển")
    mode = st.sidebar.radio("Chế độ dữ liệu", ["Dữ liệu mẫu (Mayo)", "Tải lên file (.dcm)"])

    if mode == "Dữ liệu mẫu (Mayo)":
        patient_names = [p["info"]["id"] for p in dataset.samples]
        selected_patient_idx = st.sidebar.selectbox("1. Chọn Bệnh nhân (Patient ID)", range(len(patient_names)), format_func=lambda i: patient_names[i])
        patient_batch = dataset[selected_patient_idx]
        n_slices = patient_batch["info"]["n_slices"]
        selected_slice = st.sidebar.slider("2. Chọn Lát cắt (Slice)", 0, n_slices - 1, int(n_slices/2))
        x_raw = patient_batch["x"][selected_slice].unsqueeze(0).unsqueeze(0)
        y_raw = patient_batch["y"][selected_slice].numpy()
        target_available = True
    else:
        uploaded_file = st.sidebar.file_uploader("1. Chọn file LDCT (Đầu vào)", type=["dcm"])
        uploaded_target = st.sidebar.file_uploader("2. Chọn file NDCT (Đáp án - Tùy chọn)", type=["dcm"])

        if uploaded_file is not None:
            ds = pydicom.dcmread(uploaded_file)
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            x_raw_hu = ds.pixel_array.astype("float32") * slope + intercept
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                x_raw_hu = np.max(x_raw_hu) - x_raw_hu
            st.sidebar.info(f"📊 Thông số ảnh LDCT:\n- Min HU: {np.min(x_raw_hu):.1f}\n- Max HU: {np.max(x_raw_hu):.1f}")
            x_raw_np = x_raw_hu + 1024.0
            if x_raw_np.shape != (512, 512):
                import cv2
                x_raw_np = cv2.resize(x_raw_np, (512, 512))
            x_norm = dataset._normalize(x_raw_np)
            x_raw = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(0)
            selected_slice = 0
            if uploaded_target is not None:
                ds_t = pydicom.dcmread(uploaded_target)
                slope_t = float(getattr(ds_t, "RescaleSlope", 1))
                intercept_t = float(getattr(ds_t, "RescaleIntercept", 0))
                y_raw_hu = ds_t.pixel_array.astype("float32") * slope_t + intercept_t
                if getattr(ds_t, "PhotometricInterpretation", "") == "MONOCHROME1":
                    y_raw_hu = np.max(y_raw_hu) - y_raw_hu
                if y_raw_hu.shape != (512, 512):
                    import cv2
                    y_raw_hu = cv2.resize(y_raw_hu, (512, 512))
                img_ndct_raw = y_raw_hu
                target_available = True
                y_raw = y_raw_hu
            else:
                target_available = False
                img_ndct = None
        else:
            st.info("👆 Vui lòng tải lên ít nhất một file LDCT ở thanh bên trái để bắt đầu.")
            st.stop()

    show_diff = st.sidebar.checkbox("🔍 Hiển thị Bản đồ Lỗi (Difference Map)", value=False)
    show_edge = st.sidebar.checkbox("📐 Hiển thị Bản đồ Biên (Sobel Edge Map)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**HU Windowing**")
    hu_min = st.sidebar.slider("Min HU", -1024, 1024, -160)
    hu_max = st.sidebar.slider("Max HU", -1024, 3000, 245)

    # ==========================================
    # 4. INFERENCE & PROCESSING
    # ==========================================
    with st.spinner("Đang chạy Inference..."):
        x_tensor = x_raw.to(device)
        with torch.no_grad():
            pred_redcnn = networks["redcnn"](x_tensor)
            pred_edr = networks["edr_redcnn"](x_tensor)

        img_ld     = to_numpy_hu(x_tensor, dataset)
        img_redcnn = to_numpy_hu(pred_redcnn, dataset)
        img_edr    = to_numpy_hu(pred_edr, dataset)

        if target_available:
            if mode == "Dữ liệu mẫu (Mayo)":
                img_ndct = to_numpy_hu(torch.tensor(y_raw), dataset)
            else:
                img_ndct = y_raw
        else:
            img_ndct = None

    # ==========================================
    # 5. UI LAYOUT & VISUALIZATION
    # ==========================================
    st.subheader("Trực quan hóa Khử nhiễu")

    if not target_available and mode == "Tải lên file (.dcm)":
        st.info("💡 Mẹo: Bạn có thể tải lên file NDCT (Đáp án) ở thanh bên trái để xem bảng so sánh chỉ số.")

    if target_available and img_ndct is not None:
        m_red = calc_metrics(img_redcnn, img_ndct)
        m_edr = calc_metrics(img_edr, img_ndct)

        st.markdown("#### 📊 So sánh Chỉ số lát cắt (Slice-level Comparison)")
        df_data = []
        for k in ["SSIM", "PSNR", "VIF", "Edge SSIM"]:
            diff = m_edr[k] - m_red[k]
            df_data.append({
                "Chỉ số": k,
                "RED-CNN (Baseline)": round(m_red[k], 4),
                "EDR-REDNet (Ours)": round(m_edr[k], 4),
                "Δ (EDR − RED)": f"{'+' if diff>=0 else ''}{round(diff,4)}"
            })
        st.table(pd.DataFrame(df_data))

    if target_available:
        cols = st.columns(4)
        titles = ["LDCT (Input)", "RED-CNN (Baseline)", "EDR-REDNet (Ours)", "NDCT (Target)"]
        imgs = [img_ld, img_redcnn, img_edr, img_ndct]
    else:
        cols = st.columns(3)
        titles = ["LDCT (Input)", "RED-CNN (Baseline)", "EDR-REDNet (Ours)"]
        imgs = [img_ld, img_redcnn, img_edr]

    for col, title, img in zip(cols, titles, imgs):
        with col:
            fig, ax = plt.subplots()
            ax.imshow(window_image(img, hu_min, hu_max), cmap="gray")
            ax.axis("off")
            ax.set_title(title)
            st.pyplot(fig, use_container_width=True)

    if show_diff:
        st.markdown("---")
        st.subheader("Bản đồ Lỗi (So với NDCT)")
        if target_available:
            st.markdown("Màu càng đậm (đỏ/xanh) tức là chênh lệch với ảnh thực tế NDCT càng lớn.")
            diff_cols = st.columns(3)
            diff_titles = ["Error: LDCT (Input)", "Error: RED-CNN (Baseline)", "Error: EDR-REDNet (Ours)"]
            diff_imgs = [img_ld, img_redcnn, img_edr]
            for col, title, img in zip(diff_cols, diff_titles, diff_imgs):
                with col:
                    diff = img - img_ndct
                    fig, ax = plt.subplots()
                    im = ax.imshow(diff, cmap="seismic", vmin=-200, vmax=200)
                    ax.axis("off")
                    ax.set_title(title)
                    st.pyplot(fig, use_container_width=True)
        else:
            st.warning("⚠️ Bản đồ lỗi chỉ hiển thị khi có ảnh gốc NDCT để so sánh.")

    if show_edge:
        st.markdown("---")
        st.subheader("Bản đồ Biên (Sobel Edge Map)")
        st.markdown("So sánh khả năng giữ lại các chi tiết góc cạnh, viền mô mềm của các mô hình.")
        if target_available:
            edge_cols = st.columns(4)
            edge_titles = ["Edges: LDCT", "Edges: RED-CNN", "Edges: EDR-REDNet", "Edges: NDCT"]
            edge_imgs = [img_ld, img_redcnn, img_edr, img_ndct]
        else:
            edge_cols = st.columns(3)
            edge_titles = ["Edges: LDCT", "Edges: RED-CNN", "Edges: EDR-REDNet"]
            edge_imgs = [img_ld, img_redcnn, img_edr]
        for col, title, img in zip(edge_cols, edge_titles, edge_imgs):
            with col:
                edges = filters.sobel(img)
                fig, ax = plt.subplots()
                ax.imshow(edges, cmap="gray")
                ax.axis("off")
                ax.set_title(title)
                st.pyplot(fig, use_container_width=True)