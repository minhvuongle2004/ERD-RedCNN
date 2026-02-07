# Câu 17: Cần chuẩn bị gì để train RED-CNN?

## 🎯 Mục tiêu
Biết được những gì cần chuẩn bị trước khi bắt đầu train RED-CNN

---

## 📖 Trả lời

### Chuẩn bị = Sẵn sàng mọi thứ trước khi bắt đầu

**Giống như nấu ăn**:
```
Trước khi nấu, cần:
✅ Nguyên liệu (dữ liệu)
✅ Dụng cụ (phần mềm, code)
✅ Bếp (GPU/máy tính)
✅ Công thức (hướng dẫn)
```

**Áp dụng vào train RED-CNN**:
```
Trước khi train, cần:
✅ Dataset (dữ liệu ảnh CT)
✅ Code RED-CNN
✅ GPU hoặc máy tính mạnh
✅ Môi trường Python
✅ Thư viện cần thiết
```

---

## 🛠️ 5 thứ cần chuẩn bị

### 1. Dataset (Dữ liệu)

**Cần gì?**:
```
Hàng ngàn cặp ảnh:
- Ảnh nhiễu (input)
- Ảnh sạch tương ứng (ground truth)

Ví dụ:
10,000 cặp ảnh = Tốt ✅
1,000 cặp ảnh = Tạm được ⚠️
100 cặp ảnh = Không đủ ❌
```

**Dataset phổ biến**:
- ✅ **Mayo Clinic Dataset**: Dataset công khai cho CT denoising
- ✅ **Dữ liệu riêng**: Nếu có ảnh CT từ bệnh viện

**Lưu ý**:
- ✅ Ảnh phải có kích thước phù hợp (thường 128×128 hoặc 512×512)
- ✅ Ảnh nhiễu và ảnh sạch phải tương ứng nhau
- ✅ Cần chia train/validation/test

---

### 2. Môi trường Python

**Cần cài đặt**:
```
Python 3.7+ (hoặc 3.8, 3.9)
```

**Cách kiểm tra**:
```bash
python --version
# Hoặc
python3 --version
```

**Nếu chưa có**: Tải từ python.org

---

### 3. Thư viện cần thiết

**Các thư viện quan trọng**:

**PyTorch hoặc TensorFlow**:
```
PyTorch:  Thư viện deep learning phổ biến
TensorFlow: Thư viện deep learning khác

→ RED-CNN thường dùng PyTorch
```

**Các thư viện khác**:
```
- NumPy: Xử lý mảng số
- PIL/Pillow: Xử lý ảnh
- Matplotlib: Vẽ biểu đồ
- tqdm: Hiển thị progress bar
```

**Cách cài đặt**:
```bash
pip install torch torchvision
pip install numpy pillow matplotlib tqdm
```

**Hoặc dùng file requirements.txt**:
```
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
Pillow>=8.0.0
matplotlib>=3.3.0
tqdm>=4.60.0
```

---

### 4. GPU (Quan trọng!)

**Tại sao cần GPU?**:
```
CPU:  Train 1 ảnh mất 10 giây
      → Train 10,000 ảnh = 27 giờ! ❌

GPU:  Train 1 ảnh mất 0.1 giây
      → Train 10,000 ảnh = 17 phút! ✅
```

**GPU phổ biến**:
```
NVIDIA GPU:
- RTX 3060, 3070, 3080, 3090
- GTX 1660, 1080 Ti
- Hoặc GPU trên cloud (Google Colab, AWS)
```

**Nếu không có GPU?**:
```
Có thể train trên CPU, nhưng:
- Rất chậm (hàng ngày)
- Không khuyến khích

Giải pháp:
✅ Dùng Google Colab (miễn phí GPU)
✅ Thuê GPU trên cloud
✅ Dùng máy tính có GPU
```

**Cách kiểm tra GPU**:
```python
import torch
print(torch.cuda.is_available())  # True = có GPU
print(torch.cuda.get_device_name(0))  # Tên GPU
```

---

### 5. Code RED-CNN

**Cần gì?**:
```
1. Code model RED-CNN (architecture)
2. Code training script
3. Code data loader (đọc dữ liệu)
4. Code utilities (helper functions)
```

**Có thể**:
- ✅ Tải từ GitHub (nếu có code công khai)
- ✅ Tự viết theo paper
- ✅ Dùng code có sẵn trong dự án

**Cấu trúc thư mục thường thấy**:
```
red-cnn/
├── model/
│   └── red_cnn.py          # Model architecture
├── data/
│   ├── train/              # Ảnh training
│   └── val/               # Ảnh validation
├── train.py                # Script training
├── utils.py                # Helper functions
└── requirements.txt        # Thư viện cần thiết
```

---

## 📋 Checklist chuẩn bị

### Trước khi bắt đầu train

```
□ 1. Có dataset (ít nhất 1,000 cặp ảnh)
□ 2. Cài Python 3.7+
□ 3. Cài PyTorch/TensorFlow
□ 4. Cài các thư viện cần thiết
□ 5. Có GPU (hoặc dùng Colab)
□ 6. Có code RED-CNN
□ 7. Kiểm tra GPU hoạt động
□ 8. Test đọc được dữ liệu
□ 9. Test code chạy được (không lỗi)
□ 10. Có đủ dung lượng ổ cứng (ít nhất 10GB)
```

---

## 💻 Môi trường đề xuất

### Option 1: Local (Máy tính của bạn)

**Yêu cầu**:
```
- CPU: Intel i5 trở lên
- RAM: 16GB trở lên
- GPU: NVIDIA RTX 3060 trở lên (quan trọng!)
- Ổ cứng: 50GB trống
- OS: Windows/Linux/Mac
```

**Ưu điểm**:
- ✅ Nhanh (không phụ thuộc internet)
- ✅ Bảo mật (dữ liệu không gửi đi)
- ✅ Kiểm soát hoàn toàn

**Nhược điểm**:
- ❌ Cần đầu tư phần cứng
- ❌ Tốn điện

---

### Option 2: Google Colab (Miễn phí)

**Yêu cầu**:
```
- Tài khoản Google
- Kết nối internet
- Không cần GPU riêng
```

**Ưu điểm**:
- ✅ Miễn phí GPU
- ✅ Không cần cài đặt phức tạp
- ✅ Dễ dùng cho người mới

**Nhược điểm**:
- ❌ Giới hạn thời gian (12 giờ/phiên)
- ❌ Cần upload dữ liệu lên
- ❌ Chậm hơn GPU riêng

**Cách dùng**:
1. Vào colab.research.google.com
2. Tạo notebook mới
3. Chọn GPU: Runtime → Change runtime type → GPU
4. Upload code và dữ liệu

---

### Option 3: Cloud (AWS, Azure, GCP)

**Yêu cầu**:
```
- Tài khoản cloud
- Trả phí theo giờ
```

**Ưu điểm**:
- ✅ GPU mạnh
- ✅ Linh hoạt
- ✅ Không cần máy tính mạnh

**Nhược điểm**:
- ❌ Tốn phí
- ❌ Cần kiến thức cloud

---

## 🔍 Kiểm tra sẵn sàng

### Test 1: Kiểm tra Python

```bash
python --version
# Kết quả: Python 3.8.10 (hoặc tương tự)
```

### Test 2: Kiểm tra PyTorch

```python
import torch
print(torch.__version__)  # Ví dụ: 1.10.0
```

### Test 3: Kiểm tra GPU

```python
import torch
print(torch.cuda.is_available())  # True = OK
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # Tên GPU
```

### Test 4: Kiểm tra đọc dữ liệu

```python
from PIL import Image
import numpy as np

# Test đọc ảnh
img = Image.open("path/to/image.png")
print(img.size)  # Kích thước ảnh
```

---

## 💡 Ví dụ thực tế

### Scenario: Bạn mới bắt đầu

```
Bước 1: Tải dataset Mayo Clinic
        → Có 10,000 cặp ảnh ✅

Bước 2: Cài Python 3.8
        → Cài xong ✅

Bước 3: Cài PyTorch
        → pip install torch ✅

Bước 4: Kiểm tra GPU
        → Có RTX 3060 ✅

Bước 5: Tải code RED-CNN
        → Clone từ GitHub ✅

Bước 6: Test chạy được
        → Chạy thử, không lỗi ✅

→ Sẵn sàng train! 🚀
```

---

## 🎯 Tóm tắt

**5 thứ cần chuẩn bị**:

```
1. ✅ Dataset (hàng ngàn cặp ảnh)
2. ✅ Môi trường Python (3.7+)
3. ✅ Thư viện (PyTorch, NumPy, ...)
4. ✅ GPU (quan trọng!)
5. ✅ Code RED-CNN
```

**Môi trường đề xuất**:
- ✅ Local: Nếu có GPU mạnh
- ✅ Google Colab: Nếu mới bắt đầu, không có GPU
- ✅ Cloud: Nếu cần GPU mạnh, có ngân sách

**Checklist**:
- ✅ Kiểm tra tất cả trước khi train
- ✅ Test code chạy được
- ✅ Test đọc được dữ liệu
- ✅ Test GPU hoạt động

**Kết quả**: Sẵn sàng train RED-CNN! 🎉

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Cần những gì để train RED-CNN?
2. Tại sao cần GPU?
3. Dataset cần như thế nào?
4. Làm sao kiểm tra đã sẵn sàng chưa?
5. Nên dùng môi trường nào nếu mới bắt đầu?

**Hoạt động**: 
- Kiểm tra máy tính có đủ yêu cầu không
- Hướng dẫn cài đặt Python và PyTorch
- Test GPU nếu có

---

➡️ **Tiếp theo**: `CAU_HOI_18.md` - Dataset là gì và cần như thế nào?

