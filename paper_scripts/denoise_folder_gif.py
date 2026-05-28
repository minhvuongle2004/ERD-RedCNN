import argparse  # Thư viện để xử lý tham số dòng lệnh
import torch  # Thư viện PyTorch cho deep learning
import numpy as np  # Thư viện xử lý mảng và tính toán số học
import pydicom  # Thư viện đọc/ghi file DICOM (định dạng ảnh y tế)
from ldctbench.hub import load_model  # Import hàm load model từ thư viện ldctbench
import os  # Thư viện xử lý đường dẫn và thao tác với hệ thống file
from PIL import Image, ImageDraw  # Thư viện xử lý ảnh và vẽ text lên ảnh

# ============================================================================
# CÁC HẰNG SỐ CHUẨN HÓA
# ============================================================================
# Giá trị trung bình và độ lệch chuẩn từ tập dữ liệu ldctbench
# Dùng để chuẩn hóa dữ liệu đầu vào trước khi đưa vào mô hình
MEAN = 481.45419786099086  # Giá trị trung bình của pixel trong dataset
STD = 502.18507379395044   # Độ lệch chuẩn của pixel trong dataset

# ============================================================================
# HÀM CHUẨN HÓA DỮ LIỆU
# ============================================================================
def normalize(image):
    """
    Chuẩn hóa ảnh về dạng zero-mean, unit-variance
    Công thức: (x - mean) / std
    """
    return (image - MEAN) / STD

# ============================================================================
# HÀM KHÔI PHỤC DỮ LIỆU VỀ THANG GỐC
# ============================================================================
def denormalize(image):
    """
    Đảo ngược quá trình chuẩn hóa
    Công thức: (x * std) + mean
    """
    return (image * STD) + MEAN

# ============================================================================
# HÀM CHUYỂN ĐỔI ẢNH SANG ĐỊNH DẠNG 8-BIT (0-255)
# ============================================================================
def to_uint8(image):
    """
    Chuyển đổi ảnh sang dạng uint8 (0-255) để hiển thị và tạo GIF
    Scale dựa trên giá trị min/max của ảnh
    """
    # Tìm giá trị nhỏ nhất trong ảnh
    min_val = np.min(image)
    # Tìm giá trị lớn nhất trong ảnh
    max_val = np.max(image)
    
    # Nếu có sự khác biệt giữa min và max, thực hiện scale
    if max_val > min_val:
        # Scale về khoảng [0, 1]
        scaled = (image - min_val) / (max_val - min_val)
    else:
        # Nếu tất cả pixel có cùng giá trị, tạo ảnh đen
        scaled = np.zeros_like(image)
    
    # Nhân 255 và chuyển sang kiểu uint8 (số nguyên 8-bit)
    return (scaled * 255).astype(np.uint8)

# ============================================================================
# HÀM CHÍNH: XỬ LÝ THƯ MỤC ẢNH CT
# ============================================================================
def denoise_directory(input_dir: str, output_dir: str, model_name: str, gif_output_path: str):
    """
    Xử lý khử nhiễu cho toàn bộ thư mục ảnh DICOM
    
    Args:
        input_dir: Đường dẫn thư mục chứa ảnh DICOM gốc
        output_dir: Đường dẫn thư mục lưu ảnh đã khử nhiễu
        model_name: Tên mô hình khử nhiễu (ví dụ: "redcnn")
        gif_output_path: Đường dẫn file GIF so sánh trước/sau
    """
    # In thông báo đang load model
    print(f"Đang tải mô hình '{model_name}'...")
    
    # Xác định thiết bị tính toán (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load mô hình và chuyển sang chế độ đánh giá (evaluation mode)
    # eval=True tắt dropout và batch normalization training mode
    model = load_model(model_name, eval=True).to(device)

    # Kiểm tra và tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Tạo thư mục (bao gồm cả thư mục cha nếu cần)
        print(f"Đã tạo thư mục đầu ra: {output_dir}")

    # Lấy danh sách file trong thư mục input và sắp xếp theo thứ tự
    # Chỉ lấy các file, không lấy thư mục
    files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
    
    # In số lượng file tìm thấy
    print(f"Tìm thấy {len(files)} tệp trong thư mục đầu vào.")
    
    # Danh sách để lưu các frame của GIF
    frames = []

    # Duyệt qua từng file trong danh sách
    for i, filename in enumerate(files):
        # Tạo đường dẫn đầy đủ đến file input
        input_path = os.path.join(input_dir, filename)
        
        try:
            # ================================================================
            # BƯỚC 1: ĐỌC FILE DICOM
            # ================================================================
            # Đọc file DICOM, force=True bỏ qua một số lỗi format nhỏ
            dcm = pydicom.dcmread(input_path, force=True)
            # Lấy mảng pixel và chuyển sang kiểu float32 để tính toán
            input_image = dcm.pixel_array.astype(np.float32)
        except Exception as e:
            # Nếu có lỗi khi đọc file, in thông báo và bỏ qua file này
            print(f"Lỗi khi đọc tệp DICOM {filename}: {e}. Bỏ qua.")
            continue

        # ================================================================
        # BƯỚC 2: XỬ LÝ KÍCH THƯỚC ẢNH
        # ================================================================
        # Đảm bảo ảnh là 2D (một số DICOM có thể có nhiều chiều)
        if input_image.ndim > 2:
            input_image = input_image[0]  # Lấy slice đầu tiên

        # ================================================================
        # BƯỚC 3: TIỀN XỬ LÝ - CHUẨN HÓA DỮ LIỆU
        # ================================================================
        # Chuẩn hóa ảnh theo mean và std của dataset
        normalized_image = normalize(input_image)
        
        # Chuyển từ numpy array sang PyTorch tensor
        # unsqueeze(0): Thêm chiều batch (batch_size=1)
        # unsqueeze(0) lần 2: Thêm chiều channel (channels=1 cho ảnh grayscale)
        # Kết quả: shape [1, 1, height, width]
        input_tensor = torch.from_numpy(normalized_image).unsqueeze(0).unsqueeze(0).to(device)

        # ================================================================
        # BƯỚC 4: CHẠY MÔ HÌNH (INFERENCE)
        # ================================================================
        # torch.no_grad(): Tắt tính gradient để tiết kiệm bộ nhớ và tăng tốc
        with torch.no_grad():
            # Đưa ảnh qua mô hình để khử nhiễu
            output_tensor = model(input_tensor)

        # ================================================================
        # BƯỚC 5: HẬU XỬ LÝ - CHUYỂN ĐỔI KẾT QUẢ
        # ================================================================
        # Loại bỏ các chiều thừa và chuyển từ GPU về CPU, sau đó về numpy
        output_array = output_tensor.squeeze().cpu().numpy()
        
        # Đảo ngược quá trình chuẩn hóa về thang giá trị gốc
        denoised_image = denormalize(output_array)
        
        # ================================================================
        # BƯỚC 6: GIỚI HẠN GIÁ TRỊ PIXEL THEO KIỂU DỮ LIỆU
        # ================================================================
        # Lấy giá trị min/max của kiểu dữ liệu gốc (ví dụ: int16)
        dtype_min = np.iinfo(dcm.pixel_array.dtype).min
        dtype_max = np.iinfo(dcm.pixel_array.dtype).max
        
        # Cắt bỏ các giá trị ngoài phạm vi hợp lệ
        clipped_image = np.clip(denoised_image, dtype_min, dtype_max)
        
        # ================================================================
        # BƯỚC 7: CẬP NHẬT VÀ LƯU FILE DICOM
        # ================================================================
        # Chuyển mảng pixel về kiểu dữ liệu gốc và cập nhật vào DICOM
        dcm.PixelData = clipped_image.astype(dcm.pixel_array.dtype).tobytes()
        
        # Tạo tên file output (giữ tên gốc, đổi extension thành .dcm)
        output_filename = f"{os.path.splitext(filename)[0]}.dcm"
        output_file_path = os.path.join(output_dir, output_filename)
        
        # Lưu file DICOM đã khử nhiễu
        dcm.save_as(output_file_path)

        # ================================================================
        # BƯỚC 8: TẠO FRAME CHO GIF SO SÁNH
        # ================================================================
        # Chuyển cả ảnh output và input sang dạng uint8 (0-255)
        output_uint8 = to_uint8(clipped_image)
        input_uint8 = to_uint8(input_image)
        
        # Chuyển numpy array sang PIL Image
        pil_input = Image.fromarray(input_uint8)
        pil_output = Image.fromarray(output_uint8)
        
        # Tính kích thước ảnh kết hợp (ghép ngang)
        total_width = pil_input.width + pil_output.width  # Tổng chiều rộng
        max_height = max(pil_input.height, pil_output.height)  # Chiều cao lớn nhất
        
        # Tạo ảnh trắng mới với kích thước tổng hợp
        combined_image = Image.new('RGB', (total_width, max_height))
        
        # Dán ảnh gốc vào bên trái
        combined_image.paste(pil_input, (0, 0))
        # Dán ảnh đã khử nhiễu vào bên phải
        combined_image.paste(pil_output, (pil_input.width, 0))
        
        # Thêm nhãn text lên ảnh
        draw = ImageDraw.Draw(combined_image)
        draw.text((10, 10), "Original", fill=(255, 255, 255))  # Nhãn "Original" ở góc trái
        draw.text((pil_input.width + 10, 10), "Denoised", fill=(255, 255, 255))  # Nhãn "Denoised" ở góc phải

        # Thêm frame vào danh sách
        frames.append(combined_image)

        # In tiến độ mỗi 10 ảnh
        if (i + 1) % 10 == 0:
            print(f"Đã xử lý {i + 1}/{len(files)} ảnh...")

    # In thông báo hoàn thành
    print("Hoàn thành khử nhiễu.")

    # ================================================================
    # BƯỚC 9: TẠO VÀ LƯU FILE GIF
    # ================================================================
    if frames:  # Kiểm tra có frame nào được tạo không
        print(f"Đang tạo tệp GIF so sánh tại: {gif_output_path}")
        
        # Lưu GIF với tất cả các frame
        frames[0].save(
            gif_output_path,
            save_all=True,  # Lưu tất cả frame
            append_images=frames[1:],  # Các frame từ thứ 2 trở đi
            duration=100,  # Thời gian mỗi frame (ms), 100ms = 10fps
            loop=0  # Lặp vô hạn (0 = infinite loop)
        )
        print("Đã tạo xong GIF!")
    else:
        # Không có ảnh nào được xử lý thành công
        print("Không có ảnh nào được xử lý để tạo GIF.")

# ============================================================================
# CHƯƠNG TRÌNH CHÍNH
# ============================================================================
if __name__ == "__main__":
    # Tạo parser để xử lý các tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Denoise a directory of CT images and create a comparison GIF.")
    
    # ================================================================
    # CÁC ĐƯỜNG DẪN MẶC ĐỊNH
    # ================================================================
    default_input = r"D:\tesss\ldct-benchmark\ldct-data\LDCT-and-Projection-data\C002\12-23-2021-NA-NA-62464\1.000000-Low Dose Images-39882"
    default_output = r"D:\tesss\ldct-benchmark\ldct-data\LDCT-and-Projection-data\C002\12-23-2021-NA-NA-62464\1.000000-Denoised"
    default_gif = r"D:\tesss\ldct-benchmark\ldct-data\LDCT-and-Projection-data\C002\12-23-2021-NA-NA-62464\comparison.gif"

    # ================================================================
    # ĐỊNH NGHĨA CÁC THAM SỐ DÒNG LỆNH
    # ================================================================
    # Thêm tham số --input: đường dẫn thư mục chứa ảnh đầu vào
    parser.add_argument("--input", type=str, default=default_input, help="Path to input directory.")
    
    # Thêm tham số --output: đường dẫn thư mục lưu ảnh đầu ra
    parser.add_argument("--output", type=str, default=default_output, help="Path to output directory.")
    
    # Thêm tham số --gif: đường dẫn file GIF so sánh
    parser.add_argument("--gif", type=str, default=default_gif, help="Path to output GIF file.")
    
    # Thêm tham số --model: tên mô hình khử nhiễu
    parser.add_argument("--model", type=str, default="redcnn", help="Model name.")

    # Parse các tham số từ dòng lệnh
    args = parser.parse_args()

    # Gọi hàm chính để thực hiện khử nhiễu
    denoise_directory(args.input, args.output, args.model, args.gif)