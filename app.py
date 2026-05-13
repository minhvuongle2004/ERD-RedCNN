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
# Sửa lỗi weights_only=True trên PyTorch 2.6
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

# Sửa lỗi Unicode trên Windows khi đọc file YAML
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
    """Tạo thư mục ảo để ldctbench load được model custom"""
    checkpoint_path = r"results\training\seed2024\seed2024_best_SSIM.pt"
    fake_run_dir = r"wandb\edr_redcnn_seed2024\files"
    os.makedirs(fake_run_dir, exist_ok=True)
    
    if os.path.exists(checkpoint_path):
        shutil.copy(r"configs\edrrednet.yaml", os.path.join(fake_run_dir, "args.yaml"))
        shutil.copy(checkpoint_path, os.path.join(fake_run_dir, "best_SSIM.pt"))
    return checkpoint_path

@st.cache_resource
def load_dataset():
    """Tải Dataset Test"""
    # Dùng data folder local (data)
    return TestData("data", "meanstd")

@st.cache_resource
def load_networks():
    """Tải các mô hình vào VRAM"""
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    networks = {}
    # RED-CNN (Baseline)
    networks["redcnn"] = load_model("redcnn", eval=True).to(dev)
    # EDR-REDNet (Ours)
    net = setup_trained_model(
        run_name="edr_redcnn_seed2024",
        device=dev,
        network_name="Model",
        state_dict="best_SSIM",
        eval=True,
    )
    networks["edr_redcnn"] = net
    return networks, dev

# Khởi tạo
ckpt_path = setup_environment()
if not os.path.exists(ckpt_path):
    st.error(f"❌ Không tìm thấy file trọng số tại {ckpt_path}. Vui lòng kiểm tra lại.")
    st.stop()

dataset = load_dataset()
networks, device = load_networks()

# ==========================================
# 3. DATA LOADING & SELECTION
# ==========================================
st.title("🔬 EDR-REDNet: Interactive Evaluation")
st.sidebar.header("🕹️ Điều khiển")
mode = st.sidebar.radio("Chế độ dữ liệu", ["Dữ liệu mẫu (Mayo)", "Tải lên file (.dcm)"])

if mode == "Dữ liệu mẫu (Mayo)":
    patient_names = [p["info"]["id"] for p in dataset.samples]
    selected_patient_idx = st.sidebar.selectbox("1. Chọn Bệnh nhân (Patient ID)", range(len(patient_names)), format_func=lambda i: patient_names[i])

    # Tải dữ liệu bệnh nhân (Sử dụng Lazy Loading để tiết kiệm RAM)
    patient_batch = dataset[selected_patient_idx]
    n_slices = patient_batch["info"]["n_slices"]
    selected_slice = st.sidebar.slider("2. Chọn Lát cắt (Slice)", 0, n_slices - 1, int(n_slices/2))
    
    # Đảm bảo 4D: (1, 1, 512, 512)
    x_raw = patient_batch["x"][selected_slice].unsqueeze(0).unsqueeze(0)
    y_raw = patient_batch["y"][selected_slice].numpy()
    target_available = True
else:
    uploaded_file = st.sidebar.file_uploader("1. Chọn file LDCT (Đầu vào)", type=["dcm"])
    uploaded_target = st.sidebar.file_uploader("2. Chọn file NDCT (Đáp án - Tùy chọn)", type=["dcm"])
    
    if uploaded_file is not None:
        # Đọc file LDCT
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
        
        # Xử lý file NDCT nếu có
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
            y_raw = y_raw_hu # Để dùng đồng nhất ở dưới
        else:
            target_available = False
            img_ndct = None
    else:
        st.info("👆 Vui lòng tải lên ít nhất một file LDCT ở thanh bên trái để bắt đầu.")
        st.stop()

# Tùy chọn hiển thị
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
    # Lấy tensor của slice
    x_tensor = x_raw.to(device)
    
    # Inference
    with torch.no_grad():
        pred_redcnn = networks["redcnn"](x_tensor)
        pred_edr = networks["edr_redcnn"](x_tensor)

    # Denormalize về HU (Hounsfield Units)
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
            img_ndct = y_raw # Đã là HU sẵn từ lúc upload
    else:
        img_ndct = None

def window_image(img, vmin, vmax):
    """Cắt giá trị HU theo window để hiển thị đẹp hơn"""
    img_clipped = np.clip(img, vmin, vmax)
    return (img_clipped - vmin) / (vmax - vmin)

# ==========================================
# 5. UI LAYOUT & VISUALIZATION
# ==========================================
st.subheader("Trực quan hóa Khử nhiễu")

# Thông báo nếu thiếu Target
if not target_available and mode == "Tải lên file (.dcm)":
    st.info("💡 Mẹo: Bạn có thể tải lên file NDCT (Đáp án) ở thanh bên trái để xem bảng so sánh chỉ số SSIM/PSNR/Loss.")

# Tính toán Metrics nếu có ảnh Target
if target_available and img_ndct is not None:
    import pandas as pd
    from sewar.full_ref import vifp
    from scipy import ndimage
    
    def get_edge_map(img):
        # Tính Gradient theo hướng X và Y
        sx = ndimage.sobel(img, axis=0)
        sy = ndimage.sobel(img, axis=1)
        # Kết hợp thành biên tổng hợp (magnitude)
        return np.hypot(sx, sy)

    def calc_metrics(pred, target):
        vmin, vmax = -1024.0, 3000.0
        p = np.clip(pred, vmin, vmax)
        t = np.clip(target, vmin, vmax)
        
        # Chuẩn hóa về [0, 1] để tính SSIM/PSNR
        p_norm = (p - vmin) / (vmax - vmin)
        t_norm = (t - vmin) / (vmax - vmin)
        
        # 1. Chỉ số cơ bản
        ssim_val = metrics.structural_similarity(t_norm, p_norm, data_range=1.0)
        psnr_val = metrics.peak_signal_noise_ratio(t_norm, p_norm, data_range=1.0)
        vif_val = vifp(t_norm, p_norm)
        
        # 2. Chỉ số vùng BIÊN (Edge-centric)
        edge_p = get_edge_map(p_norm)
        edge_t = get_edge_map(t_norm)
        
        # Edge SSIM (SSIM tính trên bản đồ biên)
        ep_n = (edge_p - edge_p.min()) / (edge_p.max() - edge_p.min() + 1e-8)
        et_n = (edge_t - edge_t.min()) / (edge_t.max() - edge_t.min() + 1e-8)
        edge_ssim = metrics.structural_similarity(et_n, ep_n, data_range=1.0)
        
        # Gradient RMSE (Sai số trên cạnh - càng thấp càng tốt)
        grad_rmse = np.sqrt(np.mean((edge_p - edge_t)**2))
        
        # HF Preservation (Độ giữ chi tiết cao tần - Energy ratio)
        hf_pres = np.sum(edge_p**2) / (np.sum(edge_t**2) + 1e-8)

        return {
            "SSIM": ssim_val,
            "PSNR": psnr_val,
            "VIF": vif_val,
            "Edge SSIM (Chỉ số biên)": edge_ssim,
            "Gradient RMSE (Sai số cạnh)": grad_rmse,
            "HF Preservation (Giữ chi tiết)": hf_pres
        }

    # Tính toán cho cả 2 mô hình
    m_red = calc_metrics(img_redcnn, img_ndct)
    m_edr = calc_metrics(img_edr, img_ndct)

    # Hiển thị bảng so sánh mới
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

# Hiển thị ảnh
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

# Difference Map
if show_diff:
    st.markdown("---")
    st.subheader("Bản đồ Lỗi (So với NDCT)")
    if target_available:
        st.markdown("Màu càng đậm (đỏ/xanh) tức là chênh lệch với ảnh thực tế NDCT càng lớn. Màu trắng là giống hoàn toàn.")
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
        st.warning("⚠️ Bản đồ lỗi chỉ hiển thị khi có ảnh gốc NDCT để so sánh (Chế độ dữ liệu mẫu).")

# Hàng 3: Edge Map (Nếu bật)
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