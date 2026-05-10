import os
import sys
import shutil
import yaml
import torch
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

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
# 3. STREAMLIT UI
# ==========================================
st.title("🔬 EDR-REDNet: Interactive Evaluation")
st.markdown("Trình diễn khả năng khử nhiễu ảnh CT Liều Thấp (LDCT) bằng kiến trúc EDR-REDNet.")

# Sidebar Controls
st.sidebar.header("🕹️ Điều khiển")
patient_names = [p["info"]["id"] for p in dataset.samples]
selected_patient_idx = st.sidebar.selectbox("1. Chọn Bệnh nhân (Patient ID)", range(len(patient_names)), format_func=lambda i: patient_names[i])

patient_data = dataset[selected_patient_idx]
n_slices = patient_data["info"]["n_slices"]

selected_slice = st.sidebar.slider("2. Chọn Lát cắt (Slice)", 0, n_slices - 1, int(n_slices/2))

show_diff = st.sidebar.checkbox("🔍 Hiển thị Bản đồ Lỗi (Difference Map)", value=False)
show_edge = st.sidebar.checkbox("📐 Hiển thị Bản đồ Biên (Sobel Edge Map)", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown("**HU Windowing**")
hu_min = st.sidebar.slider("Min HU", -1024, 1024, -1000)
hu_max = st.sidebar.slider("Max HU", -1024, 3000, 1000)

# ==========================================
# 4. INFERENCE & PROCESSING
# ==========================================
with st.spinner("Đang chạy Inference..."):
    # Lấy tensor của slice
    # Shape: (H, W) -> Thêm batch & channel -> (1, 1, H, W)
    x_tensor = torch.unsqueeze(torch.unsqueeze(patient_data["x"][selected_slice], 0), 0).to(device)
    y_tensor = patient_data["y"][selected_slice].numpy() # Ground Truth NDCT
    
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
    img_ndct = to_numpy_hu(torch.tensor(y_tensor))

def window_image(img, vmin, vmax):
    """Cắt giá trị HU theo window để hiển thị đẹp hơn"""
    img_clipped = np.clip(img, vmin, vmax)
    return (img_clipped - vmin) / (vmax - vmin)

def get_sobel_edges(img):
    """Tính toán Sobel Edge Map"""
    dx = sobel(img, axis=0)
    dy = sobel(img, axis=1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag

# ==========================================
# 5. VISUALIZATION
# ==========================================
images = {
    "LDCT (Input)": img_ld,
    "RED-CNN (Baseline)": img_redcnn,
    "EDR-REDNet (Ours)": img_edr,
    "NDCT (Target)": img_ndct
}

# Hàng 1: Hình ảnh hiển thị bình thường
st.subheader("Trực quan hóa Khử nhiễu")
cols = st.columns(4)
for i, (title, img) in enumerate(images.items()):
    with cols[i]:
        st.markdown(f"**{title}**")
        fig, ax = plt.subplots()
        ax.imshow(window_image(img, hu_min, hu_max), cmap="gray")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)

# Hàng 2: Difference Map (Nếu bật)
if show_diff:
    st.subheader("Bản đồ Lỗi (So với NDCT)")
    st.markdown("Màu càng đậm (đỏ/xanh) tức là chênh lệch với ảnh thực tế NDCT càng lớn. Màu trắng là giống hoàn toàn.")
    diff_cols = st.columns(4)
    for i, (title, img) in enumerate(images.items()):
        with diff_cols[i]:
            if title == "NDCT (Target)":
                continue
            st.markdown(f"**Error: {title}**")
            diff = img - img_ndct
            fig, ax = plt.subplots()
            # Dùng colormap seismic để highlight sai số âm/dương
            im = ax.imshow(diff, cmap="seismic", vmin=-200, vmax=200)
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

# Hàng 3: Edge Map (Nếu bật)
if show_edge:
    st.subheader("Bản đồ Biên (Sobel Edge Map)")
    st.markdown("So sánh khả năng giữ lại các chi tiết góc cạnh, viền mô mềm của các mô hình.")
    edge_cols = st.columns(4)
    for i, (title, img) in enumerate(images.items()):
        with edge_cols[i]:
            st.markdown(f"**Edges: {title}**")
            edges = get_sobel_edges(img)
            fig, ax = plt.subplots()
            ax.imshow(edges, cmap="gray")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)