import streamlit as st
import torch
import numpy as np
import pydicom
import os
import tempfile
from PIL import Image, ImageDraw
from ldctbench.hub import load_model
import tkinter as tk
from tkinter import filedialog
from skimage import io as skio

# ============================================================================
# CẤU HÌNH VÀ HẰNG SỐ
# ============================================================================
st.set_page_config(page_title="LDCT Denoising App", layout="wide")

MEAN = 481.45419786099086
STD = 502.18507379395044

# ============================================================================
# CÁC HÀM HỖ TRỢ (Tái sử dụng từ script cũ)
# ============================================================================
def normalize(image):
    return (image - MEAN) / STD

def denormalize(image):
    return (image * STD) + MEAN

def to_uint8(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        scaled = (image - min_val) / (max_val - min_val)
    else:
        scaled = np.zeros_like(image)
    return (scaled * 255).astype(np.uint8)

def load_image(file_path_or_obj, is_file_obj=False):
    """Load ảnh từ nhiều định dạng: DICOM, PNG, JPG, TIFF"""
    try:
        if is_file_obj:
            # Xử lý file upload từ Streamlit
            file_ext = os.path.splitext(file_path_or_obj.name)[1].lower()
        else:
            # Xử lý file từ đường dẫn
            file_ext = os.path.splitext(file_path_or_obj)[1].lower()
        
        # DICOM files
        if file_ext in ['.dcm', '.dicom']:
            if is_file_obj:
                dcm = pydicom.dcmread(file_path_or_obj, force=True)
            else:
                dcm = pydicom.dcmread(file_path_or_obj, force=True)
            image = dcm.pixel_array.astype(np.float32)
            if image.ndim > 2:
                image = image[0]
            return image
        
        # TIFF files (có thể là 16-bit hoặc float)
        elif file_ext in ['.tiff', '.tif']:
            if is_file_obj:
                # Lưu file tạm để đọc
                with tempfile.NamedTemporaryFile(delete=False, suffix='.tiff') as tmp:
                    tmp.write(file_path_or_obj.read())
                    tmp_path = tmp.name
                image = skio.imread(tmp_path).astype(np.float32)
                os.unlink(tmp_path)
            else:
                image = skio.imread(file_path_or_obj).astype(np.float32)
            # Nếu là ảnh grayscale, đảm bảo là 2D
            if image.ndim > 2:
                image = image[:, :, 0] if image.shape[2] == 1 else np.mean(image, axis=2)
            return image
        
        # PNG, JPG, JPEG files
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            if is_file_obj:
                img = Image.open(file_path_or_obj)
            else:
                img = Image.open(file_path_or_obj)
            
            # Chuyển sang grayscale nếu là RGB
            if img.mode != 'L':
                img = img.convert('L')
            
            image = np.array(img).astype(np.float32)
            return image
        
        else:
            raise ValueError(f"Định dạng không được hỗ trợ: {file_ext}")
    
    except Exception as e:
        raise ValueError(f"Không thể đọc file: {e}")

@st.cache_resource
def get_model(model_name):
    """Load model và cache lại để không phải load lại mỗi lần thao tác"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, eval=True).to(device)
    return model, device

# ============================================================================
# GIAO DIỆN CHÍNH
# ============================================================================
def main():
    st.title("🏥 Ứng dụng Khử nhiễu ảnh CT (LDCT Denoising)")

    # --- Sidebar: Cấu hình ---
    st.sidebar.header("⚙️ Cấu hình")
    model_name = st.sidebar.selectbox(
        "Chọn mô hình", 
        ["redcnn", "cnn10", "wganvgg", "resnet", "qae", "dugan", "transct", "bilateral"], 
        index=0
    )
    
    # Load model
    try:
        model, device = get_model(model_name)
        st.sidebar.success(f"✅ Đã tải mô hình: {model_name}\n\n🖥️ Thiết bị: {device}")
    except Exception as e:
        st.sidebar.error(f"Lỗi tải mô hình: {e}")
        return

    # Chọn chế độ
    mode = st.sidebar.radio("Chọn chế độ làm việc:", ["📂 Xử lý cả thư mục (Local)", "⬆️ Upload file lẻ"])

    # Khởi tạo session state cho đường dẫn thư mục nếu chưa có
    if 'input_dir' not in st.session_state:
        st.session_state.input_dir = ""

    # Khởi tạo state cho viewer thư mục
    if 'viewer_active' not in st.session_state:
        st.session_state.viewer_active = False
    if 'viewer_files' not in st.session_state:
        st.session_state.viewer_files = []
    if 'viewer_dirs' not in st.session_state:
        st.session_state.viewer_dirs = {}

    # --- Chế độ 1: Xử lý thư mục ---
    if mode == "📂 Xử lý cả thư mục (Local)":
        st.header("Xử lý hàng loạt thư mục trên máy")
        
        col1, col2 = st.columns(2)
        with col1:
            # Nút chọn thư mục sử dụng Tkinter
            if st.button("📂 Chọn thư mục Input"):
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1) # Đẩy cửa sổ lên trên cùng
                folder_path = filedialog.askdirectory(master=root)
                root.destroy()
                if folder_path:
                    st.session_state.input_dir = folder_path
            
            input_dir = st.text_input("Đường dẫn thư mục đầu vào:", st.session_state.input_dir)

        with col2:
            # Output mặc định là result/dcm
            default_out = os.path.join(os.getcwd(), "result", "dcm")
            output_dir = st.text_input("Thư mục đầu ra (Output)", default_out)

        if st.button("🚀 Bắt đầu khử nhiễu", type="primary"):
            if not os.path.exists(input_dir):
                st.error("❌ Thư mục đầu vào không tồn tại!")
            else:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Lọc các file ảnh được hỗ trợ
                supported_extensions = ['.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.tiff', '.tif']
                files = sorted([f for f in os.listdir(input_dir) 
                                if os.path.isfile(os.path.join(input_dir, f)) 
                                and any(f.lower().endswith(ext) for ext in supported_extensions)])
                
                if not files:
                    st.warning("Không tìm thấy file ảnh nào được hỗ trợ (.dcm, .png, .jpg, .tiff) trong thư mục này!")
                    return
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Placeholder để hiển thị ảnh preview
                image_spot = st.empty()
                
                for i, filename in enumerate(files):
                    input_path = os.path.join(input_dir, filename)
                    try:
                        # Load ảnh (hỗ trợ nhiều định dạng)
                        input_image = load_image(input_path, is_file_obj=False)
                        
                        # Xử lý khử nhiễu
                        norm_img = normalize(input_image)
                        input_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            output_tensor = model(input_tensor)
                        
                        output_array = output_tensor.squeeze().cpu().numpy()
                        denoised_image = denormalize(output_array)
                        
                        # Lưu file
                        file_ext = os.path.splitext(filename)[1].lower()
                        output_filename = f"{os.path.splitext(filename)[0]}_denoised{file_ext}"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Lưu theo định dạng gốc
                        if file_ext in ['.dcm', '.dicom']:
                            # DICOM: giữ nguyên metadata
                            dcm = pydicom.dcmread(input_path, force=True)
                            dtype_min = np.iinfo(dcm.pixel_array.dtype).min
                            dtype_max = np.iinfo(dcm.pixel_array.dtype).max
                            clipped_image = np.clip(denoised_image, dtype_min, dtype_max)
                            dcm.PixelData = clipped_image.astype(dcm.pixel_array.dtype).tobytes()
                            dcm.save_as(output_path)
                        else:
                            # PNG, JPG, TIFF: lưu dưới dạng PNG
                            clipped_image = np.clip(denoised_image, 0, 65535)  # 16-bit range
                            Image.fromarray(to_uint8(clipped_image)).save(output_path.replace(file_ext, '.png'))

                        # Preview mỗi 5 ảnh
                        if i % 5 == 0:
                            col_a, col_b = st.columns(2)
                            image_spot.image(to_uint8(clipped_image), caption=f"Đang xử lý: {filename}", width=300)

                    except Exception as e:
                        print(f"Lỗi {filename}: {e}")
                    
                    # Cập nhật tiến độ
                    progress_bar.progress((i + 1) / len(files))
                    status_text.text(f"Đang xử lý {i+1}/{len(files)}: {filename}")
                
                # Cập nhật state để hiển thị viewer
                st.session_state.viewer_active = True
                st.session_state.viewer_files = files
                st.session_state.viewer_dirs = {"input": input_dir, "output": output_dir}
                
                st.success("✅ Hoàn thành! Kéo xuống dưới để xem kết quả.")

        # --- Phần hiển thị kết quả dạng cuộn (Viewer) ---
        if st.session_state.viewer_active and st.session_state.viewer_files:
            st.markdown("---")
            st.header("🎞️ Xem kết quả: Cuộn để xem các lát cắt")
            
            # Thanh trượt để chọn lát cắt (giả lập hiệu ứng cuộn ảnh động)
            slice_idx = st.slider(
                "Kéo thanh trượt để xem các lát cắt (Slice Index)", 
                min_value=0, 
                max_value=len(st.session_state.viewer_files) - 1, 
                value=0,
                key="slice_slider"
            )
            
            # Lấy tên file hiện tại dựa trên thanh trượt
            current_file = st.session_state.viewer_files[slice_idx]
            
            # Đường dẫn file
            in_path = os.path.join(st.session_state.viewer_dirs["input"], current_file)
            out_path = os.path.join(st.session_state.viewer_dirs["output"], current_file)
            
            try:
                # Load ảnh Input và Output (hỗ trợ nhiều định dạng)
                img_in = load_image(in_path, is_file_obj=False)
                img_out = load_image(out_path, is_file_obj=False)
                
                # Hiển thị side-by-side
                c1, c2 = st.columns(2)
                with c1:
                    st.image(to_uint8(img_in), caption=f"Input: {current_file}", use_column_width=True)
                with c2:
                    st.image(to_uint8(img_out), caption=f"Denoised: {current_file}", use_column_width=True)
            except Exception as e:
                st.error(f"Không thể đọc file {current_file}: {e}")

    # --- Chế độ 2: Upload file ---
    elif mode == "⬆️ Upload file lẻ":
        st.header("Upload và khử nhiễu nhanh")
        uploaded_file = st.file_uploader(
            "Chọn file ảnh", 
            type=["dcm", "dicom", "png", "jpg", "jpeg", "tiff", "tif"],
            help="Hỗ trợ: DICOM (.dcm), PNG, JPG, TIFF"
        )
        
        if uploaded_file is not None:
            # Đọc file từ bộ nhớ (hỗ trợ nhiều định dạng)
            input_image = load_image(uploaded_file, is_file_obj=True)

            # Hiển thị ảnh gốc
            col1, col2 = st.columns(2)
            with col1:
                st.image(to_uint8(input_image), caption="Ảnh gốc (Nhiễu)", use_column_width=True)

            if st.button("✨ Khử nhiễu ngay"):
                # Tạo thư mục result/img nếu chưa có
                save_dir = os.path.join(os.getcwd(), "result", "img")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                with st.spinner("Đang xử lý..."):
                    norm_img = normalize(input_image)
                    input_tensor = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output_tensor = model(input_tensor)
                    
                    output_array = output_tensor.squeeze().cpu().numpy()
                    denoised_image = denormalize(output_array)
                    
                    # Lưu kết quả
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    if file_ext in ['.dcm', '.dicom']:
                        # DICOM: giữ nguyên định dạng
                        dcm = pydicom.dcmread(uploaded_file, force=True)
                        dtype_min = np.iinfo(dcm.pixel_array.dtype).min
                        dtype_max = np.iinfo(dcm.pixel_array.dtype).max
                        clipped_image = np.clip(denoised_image, dtype_min, dtype_max)
                        output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_denoised.dcm"
                        output_path = os.path.join(save_dir, output_filename)
                        dcm.PixelData = clipped_image.astype(dcm.pixel_array.dtype).tobytes()
                        dcm.save_as(output_path)
                    else:
                        # PNG, JPG, TIFF: lưu dưới dạng PNG
                        clipped_image = np.clip(denoised_image, 0, 65535)
                        output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_denoised.png"
                        output_path = os.path.join(save_dir, output_filename)
                        Image.fromarray(to_uint8(clipped_image)).save(output_path)

                # Hiển thị kết quả
                with col2:
                    st.image(to_uint8(clipped_image), caption="Ảnh sau khi khử nhiễu", use_column_width=True)
                
                st.success(f"Xử lý xong! Ảnh đã được lưu tại: {output_path}")

if __name__ == "__main__":
    main()