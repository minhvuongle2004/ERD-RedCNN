import os
import sys
import shutil
import yaml
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pydicom
from skimage import filters, metrics

st.set_page_config(page_title="EDR-REDNet Visualizer", layout="wide")

# ==========================================
# 1. SETUP ENVIRONMENT & PATCHES
# ==========================================
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
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
@st.cache_resource
def setup_environment():
    checkpoint_path = r"results\training\seed2024\lan1\seed2024_best_SSIM.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = r"results\training\seed2024\seed2024_best_SSIM.pt"
    fake_run_dir = r"wandb\edr_redcnn_seed2024\files"
    os.makedirs(fake_run_dir, exist_ok=True)
    if os.path.exists(checkpoint_path):
        shutil.copy(r"configs\edrrednet.yaml", os.path.join(fake_run_dir, "args.yaml"))
        shutil.copy(checkpoint_path, os.path.join(fake_run_dir, "best_SSIM.pt"))
    return checkpoint_path

@st.cache_resource
def load_dataset():
    return TestData("data", "meanstd")

@st.cache_resource
def load_networks():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    networks = {}
    networks["redcnn"] = load_model("redcnn", eval=True).to(dev)
    net = setup_trained_model(
        run_name="edr_redcnn_seed2024",
        device=dev,
        network_name="Model",
        state_dict="best_SSIM",
        eval=True,
    )
    networks["edr_redcnn"] = net
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
with tab_ablation:
    st.header("🧪 Ablation Study — Phân tích đóng góp từng thành phần")
    st.markdown("""
    Bảng dưới đây so sánh hiệu năng của 4 biến thể kiến trúc để chứng minh vai trò
    của từng thành phần trong EDR-REDNet.
    """)

    import pandas as pd

    # --- Định nghĩa các Variant và đường dẫn ---
    VARIANTS = {
        "A — RED-CNN (Baseline)":    {"sobel_input": "❌", "edge_block": "❌", "sobel_loss": "❌", "folders": ["results/training/seed1339/lan1", "results/training/seed1339"]},
        "B — + EdgeBlock":            {"sobel_input": "❌", "edge_block": "✅", "sobel_loss": "❌", "folders": ["results/training/VariantB/Seed1339"]},
        "C — + Sobel Input":          {"sobel_input": "✅", "edge_block": "✅", "sobel_loss": "❌", "folders": ["results/training/VariantC/Seed1339"]},
        "D — Full EDR-REDNet (Ours)": {"sobel_input": "✅", "edge_block": "✅", "sobel_loss": "✅", "folders": ["results/training/seed2024/lan1", "results/training/seed2024"]},
    }

    def find_csv(folders, keyword):
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if keyword in f and f.endswith(".csv"):
                    return os.path.join(folder, f)
        return None

    # --- Bảng Kiến trúc ---
    st.subheader("🏗️ Cấu hình các Biến thể")
    arch_rows = []
    metrics_files = {}
    losses_files = {}

    for name, meta in VARIANTS.items():
        mf = find_csv(meta["folders"], "Metrics")
        lf = find_csv(meta["folders"], "Losses")
        metrics_files[name] = mf
        losses_files[name] = lf
        arch_rows.append({
            "Biến thể": name,
            "FixedSobelLayer (Input)": meta["sobel_input"],
            "EdgeBlock (Dilated)": meta["edge_block"],
            "Sobel Loss": meta["sobel_loss"],
            "Kết quả": "✅ Có" if mf else "⏳ Chưa có"
        })
    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True)

    # --- Bảng So sánh Số liệu ---
    st.subheader("📊 Bảng So sánh Số liệu (Best Validation)")
    summary_rows = []
    loaded_metrics_dfs = {}

    for name, meta in VARIANTS.items():
        mf = metrics_files[name]
        if mf and os.path.exists(mf):
            try:
                df = pd.read_csv(mf)
                ssim_col = next((c for c in df.columns if "ssim" in c.lower()), None)
                psnr_col = next((c for c in df.columns if "psnr" in c.lower()), None)
                loaded_metrics_dfs[name] = df
                best_ssim = float(df[ssim_col].max()) if ssim_col else None
                best_psnr = float(df[psnr_col].max()) if psnr_col else None
                summary_rows.append({
                    "Biến thể": name,
                    "Best SSIM ↑": f"{best_ssim:.5f}" if best_ssim else "N/A",
                    "Best PSNR ↑ (dB)": f"{best_psnr:.3f}" if best_psnr else "N/A",
                    "EdgeBlock": meta["edge_block"],
                    "Sobel Input": meta["sobel_input"],
                    "Sobel Loss": meta["sobel_loss"],
                })
            except Exception as e:
                summary_rows.append({
                    "Biến thể": name, "Best SSIM ↑": f"Lỗi: {e}",
                    "Best PSNR ↑ (dB)": "—", "EdgeBlock": meta["edge_block"],
                    "Sobel Input": meta["sobel_input"], "Sobel Loss": meta["sobel_loss"]
                })
        else:
            summary_rows.append({
                "Biến thể": name, "Best SSIM ↑": "⏳ Chưa có",
                "Best PSNR ↑ (dB)": "⏳ Chưa có", "EdgeBlock": meta["edge_block"],
                "Sobel Input": meta["sobel_input"], "Sobel Loss": meta["sobel_loss"]
            })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # --- Learning Curves ---
    st.subheader("📈 Learning Curves (SSIM theo Iterations)")
    if loaded_metrics_dfs:
        fig_lc, ax_lc = plt.subplots(figsize=(12, 5))
        colors = {
            "A — RED-CNN (Baseline)":    "#888888",
            "B — + EdgeBlock":            "#4e9af1",
            "C — + Sobel Input":          "#f1a74e",
            "D — Full EDR-REDNet (Ours)": "#2ecc71",
        }
        for name, df in loaded_metrics_dfs.items():
            ssim_col = next((c for c in df.columns if "ssim" in c.lower()), None)
            iter_col = next((c for c in df.columns if "iter" in c.lower() or "step" in c.lower()), None)
            if ssim_col:
                x = df[iter_col] if iter_col else range(len(df))
                ax_lc.plot(x, df[ssim_col], label=name, color=colors.get(name), linewidth=2)
        ax_lc.set_xlabel("Iterations")
        ax_lc.set_ylabel("SSIM (Validation)")
        ax_lc.set_title("So sánh tốc độ học và chất lượng cuối của 4 Variant")
        ax_lc.legend()
        ax_lc.grid(True, alpha=0.3)
        st.pyplot(fig_lc, use_container_width=True)
    else:
        st.info("⏳ Chưa có file kết quả. Hãy train các Variant trước.")

    # --- Loss Curves ---
    loaded_losses_dfs = {}
    for name, meta in VARIANTS.items():
        lf = losses_files[name]
        if lf and os.path.exists(lf):
            try:
                loaded_losses_dfs[name] = pd.read_csv(lf)
            except:
                pass

    if loaded_losses_dfs:
        st.subheader("📉 Loss Curves (Train Loss theo Iterations)")
        fig_ll, ax_ll = plt.subplots(figsize=(12, 5))
        colors = {
            "A — RED-CNN (Baseline)":    "#888888",
            "B — + EdgeBlock":            "#4e9af1",
            "C — + Sobel Input":          "#f1a74e",
            "D — Full EDR-REDNet (Ours)": "#2ecc71",
        }
        for name, df in loaded_losses_dfs.items():
            loss_col = next((c for c in df.columns if "loss" in c.lower()), None)
            iter_col = next((c for c in df.columns if "iter" in c.lower() or "step" in c.lower()), None)
            if loss_col:
                x = df[iter_col] if iter_col else range(len(df))
                ax_ll.plot(x, df[loss_col], label=name, color=colors.get(name), linewidth=2)
        ax_ll.set_xlabel("Iterations")
        ax_ll.set_ylabel("Loss")
        ax_ll.set_title("So sánh Train Loss của 4 Variant")
        ax_ll.legend()
        ax_ll.grid(True, alpha=0.3)
        st.pyplot(fig_ll, use_container_width=True)

    # --- Giải thích ---
    st.markdown("---")
    st.subheader("💡 Giải thích Kết quả")
    st.markdown("""
    | Bước | Câu hỏi được trả lời |
    |------|----------------------|
    | **A → B** | Thêm khối Dilated Residual vào giữa mạng có giúp tăng SSIM không? |
    | **B → C** | Nạp thêm bản đồ biên Sobel vào đầu vào có giúp mạng học tốt hơn không? |
    | **C → D** | Dùng thêm Sobel Loss trong hàm mất mát có phải là yếu tố đột phá nhất không? |
    """)

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

        def to_numpy_hu(tensor):
            img_np = dataset.denormalize(tensor.cpu().squeeze()).numpy()
            return dataset._convert_hu(img_np, to_hu=True)

        img_ld = to_numpy_hu(x_tensor)
        img_redcnn = to_numpy_hu(pred_redcnn)
        img_edr = to_numpy_hu(pred_edr)

        if target_available:
            if mode == "Dữ liệu mẫu (Mayo)":
                img_ndct = to_numpy_hu(torch.tensor(y_raw))
            else:
                img_ndct = y_raw
        else:
            img_ndct = None

    def window_image(img, vmin, vmax):
        img_clipped = np.clip(img, vmin, vmax)
        return (img_clipped - vmin) / (vmax - vmin)

    # ==========================================
    # 5. UI LAYOUT & VISUALIZATION
    # ==========================================
    st.subheader("Trực quan hóa Khử nhiễu")

    if not target_available and mode == "Tải lên file (.dcm)":
        st.info("💡 Mẹo: Bạn có thể tải lên file NDCT (Đáp án) ở thanh bên trái để xem bảng so sánh chỉ số.")

    if target_available and img_ndct is not None:
        import pandas as pd
        from sewar.full_ref import vifp
        from scipy import ndimage

        def get_edge_map(img):
            sx = ndimage.sobel(img, axis=0)
            sy = ndimage.sobel(img, axis=1)
            return np.hypot(sx, sy)

        def calc_metrics(pred, target):
            vmin, vmax = -1024.0, 3000.0
            p = np.clip(pred, vmin, vmax)
            t = np.clip(target, vmin, vmax)
            p_norm = (p - vmin) / (vmax - vmin)
            t_norm = (t - vmin) / (vmax - vmin)
            ssim_val = metrics.structural_similarity(t_norm, p_norm, data_range=1.0)
            psnr_val = metrics.peak_signal_noise_ratio(t_norm, p_norm, data_range=1.0)
            vif_val = vifp(t_norm, p_norm)
            edge_p = get_edge_map(p_norm)
            edge_t = get_edge_map(t_norm)
            ep_n = (edge_p - edge_p.min()) / (edge_p.max() - edge_p.min() + 1e-8)
            et_n = (edge_t - edge_t.min()) / (edge_t.max() - edge_t.min() + 1e-8)
            edge_ssim = metrics.structural_similarity(et_n, ep_n, data_range=1.0)
            grad_rmse = np.sqrt(np.mean((edge_p - edge_t)**2))
            hf_pres = np.sum(edge_p**2) / (np.sum(edge_t**2) + 1e-8)
            return {
                "SSIM": ssim_val, "PSNR": psnr_val, "VIF": vif_val,
                "Edge SSIM (Chỉ số biên)": edge_ssim,
                "Gradient RMSE (Sai số cạnh)": grad_rmse,
                "HF Preservation (Giữ chi tiết)": hf_pres
            }

        m_red = calc_metrics(img_redcnn, img_ndct)
        m_edr = calc_metrics(img_edr, img_ndct)

        st.markdown("#### 📊 So sánh Chỉ số lát cắt (Slice-level Comparison)")
        df_data = []
        for k in m_red.keys():
            diff = m_edr[k] - m_red[k]
            df_data.append({
                "Chỉ số": k,
                "RED-CNN (Baseline)": round(m_red[k], 4),
                "EDR-REDNet (Ours)": round(m_edr[k], 4),
                "Chênh lệch (Delta)": round(diff, 6)
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