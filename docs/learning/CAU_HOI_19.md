# Câu 19: Cách chuẩn bị dữ liệu training?

## 🎯 Mục tiêu
Biết được các bước xử lý dữ liệu trước khi train RED-CNN

---

## 📖 Trả lời

### Chuẩn bị dữ liệu = Xử lý ảnh trước khi train

**Giống như chuẩn bị nguyên liệu nấu ăn**:
```
Rau củ mua về:
→ Rửa sạch
→ Cắt nhỏ
→ Sắp xếp
→ Sẵn sàng nấu! ✅
```

**Áp dụng vào ảnh CT**:
```
Ảnh CT thô:
→ Chuyển đổi định dạng
→ Resize về cùng kích thước
→ Normalize (chuẩn hóa)
→ Sắp xếp vào thư mục
→ Sẵn sàng train! ✅
```

---

## 🔄 Quy trình chuẩn bị

### Bước 1: Thu thập ảnh

**Có thể từ**:
```
1. Dataset công khai (Mayo Clinic)
2. Dữ liệu từ bệnh viện
3. Tạo ảnh nhiễu từ ảnh sạch (synthetic)
```

**Lưu ý**:
- ✅ Ảnh phải có cặp (nhiễu + sạch)
- ✅ Ảnh phải tương ứng nhau
- ✅ Kiểm tra chất lượng ảnh

---

### Bước 2: Chuyển đổi định dạng

**Ảnh CT thường ở định dạng DICOM**:
```
DICOM (.dcm) → PNG (.png) hoặc TIFF (.tiff)

Tại sao?
→ DICOM phức tạp, khó xử lý
→ PNG/TIFF đơn giản, dễ xử lý
```

**Cách chuyển đổi**:
```
Dùng thư viện:
- pydicom (đọc DICOM)
- PIL/Pillow (lưu PNG)

Hoặc dùng tool:
- ImageJ
- DICOM viewer
```

**Ví dụ code (minh họa)**:
```python
import pydicom
from PIL import Image

# Đọc DICOM
dcm = pydicom.dcmread("image.dcm")
img_array = dcm.pixel_array

# Chuyển sang PNG
img = Image.fromarray(img_array)
img.save("image.png")
```

---

### Bước 3: Resize về cùng kích thước

**Vấn đề**: Ảnh có thể có kích thước khác nhau

```
Ảnh 1: 512×512
Ảnh 2: 256×256
Ảnh 3: 1024×1024

→ Cần resize về cùng kích thước!
```

**Kích thước phổ biến**:
```
- 128×128 (nhỏ, train nhanh)
- 256×256 (vừa, cân bằng)
- 512×512 (lớn, chi tiết tốt)
```

**Cách resize**:
```python
from PIL import Image

# Đọc ảnh
img = Image.open("image.png")

# Resize về 256×256
img_resized = img.resize((256, 256))

# Lưu lại
img_resized.save("image_256.png")
```

**Lưu ý**:
- ✅ Dùng interpolation tốt (LANCZOS)
- ✅ Giữ tỷ lệ nếu có thể
- ✅ Resize cả ảnh nhiễu và sạch cùng cách

---

### Bước 4: Normalize (Chuẩn hóa)

**Tại sao cần normalize?**:
```
Ảnh CT có giá trị pixel:
- Có thể từ 0-255 (8-bit)
- Có thể từ 0-4095 (12-bit)
- Có thể từ -1000 đến 3000 (HU values)

→ Model cần giá trị trong khoảng nhất định
→ Thường normalize về 0-1 hoặc -1 đến 1
```

**Cách normalize**:
```python
import numpy as np

# Đọc ảnh
img_array = np.array(Image.open("image.png"))

# Normalize về 0-1
img_normalized = img_array / 255.0

# Hoặc normalize về -1 đến 1
img_normalized = (img_array / 255.0) * 2 - 1
```

**Quan trọng**:
- ✅ Normalize cả ảnh nhiễu và sạch cùng cách
- ✅ Lưu lại giá trị min/max để denormalize sau

---

### Bước 5: Tạo cặp ảnh

**Đảm bảo ảnh nhiễu và sạch tương ứng**:
```
Ảnh nhiễu: img_001_noisy.png
Ảnh sạch:  img_001_clean.png

→ Tên file phải khớp!
→ Cùng bệnh nhân, cùng vị trí
```

**Cách tổ chức**:
```
dataset/
├── train/
│   ├── noisy/
│   │   ├── 001.png
│   │   ├── 002.png
│   │   └── ...
│   └── clean/
│       ├── 001.png  ← Phải khớp với 001.png ở noisy
│       ├── 002.png
│       └── ...
```

---

### Bước 6: Chia train/val/test

**Tại sao cần chia?**:
```
Train:    Model học từ đây
Val:      Kiểm tra khi train
Test:     Đánh giá cuối cùng
```

**Tỷ lệ phổ biến**:
```
80/10/10:  Train 80%, Val 10%, Test 10%
70/15/15:  Train 70%, Val 15%, Test 15%
```

**Cách chia**:
```python
import os
import shutil
import random

# Danh sách tất cả ảnh
all_images = os.listdir("all_images/")
random.shuffle(all_images)

# Chia
train = all_images[:8000]  # 80%
val = all_images[8000:9000]  # 10%
test = all_images[9000:]  # 10%

# Copy vào thư mục tương ứng
for img in train:
    shutil.copy(f"all_images/{img}", f"train/{img}")
# ... tương tự cho val và test
```

---

## 🛠️ Data Augmentation (Tăng dữ liệu)

### Tại sao cần?

```
Có 1,000 ảnh:
→ Train → Kết quả không tốt ❌

Có 1,000 ảnh + Augmentation:
→ Tạo thêm 4,000 ảnh nữa
→ Tổng 5,000 ảnh
→ Train → Kết quả tốt hơn ✅
```

### Các kỹ thuật Augmentation

**1. Xoay ảnh**:
```
Ảnh gốc → Xoay 90°, 180°, 270°
→ Tạo thêm 3 ảnh
```

**2. Lật ảnh**:
```
Ảnh gốc → Lật ngang, lật dọc
→ Tạo thêm 2 ảnh
```

**3. Thay đổi độ sáng**:
```
Ảnh gốc → Sáng hơn, tối hơn
→ Tạo thêm 2 ảnh
```

**Lưu ý**:
- ✅ Chỉ augment ảnh nhiễu và sạch CÙNG cách
- ✅ Không làm biến dạng quá nhiều
- ✅ Phù hợp với ảnh CT (không xoay lung tung)

---

## 📋 Checklist chuẩn bị

### Trước khi train

```
□ 1. Thu thập đủ ảnh (1,000+ cặp)
□ 2. Chuyển DICOM → PNG (nếu cần)
□ 3. Resize về cùng kích thước
□ 4. Normalize giá trị pixel
□ 5. Tạo cặp ảnh (nhiễu + sạch)
□ 6. Đảm bảo tên file khớp
□ 7. Chia train/val/test
□ 8. Data augmentation (nếu cần)
□ 9. Kiểm tra lại tất cả ảnh
□ 10. Lưu metadata (kích thước, min/max values)
```

---

## 💻 Code mẫu (Minh họa)

### Script chuẩn bị dữ liệu

```python
import os
from PIL import Image
import numpy as np

def prepare_data(input_dir, output_dir, target_size=(256, 256)):
    """
    Chuẩn bị dữ liệu:
    - Resize về cùng kích thước
    - Normalize về 0-1
    - Lưu vào thư mục mới
    """
    
    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)
    
    # Xử lý từng ảnh
    for filename in os.listdir(input_dir):
        if filename.endswith('.png'):
            # Đọc ảnh
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)
            
            # Resize
            img_resized = img.resize(target_size, Image.LANCZOS)
            
            # Chuyển sang array và normalize
            img_array = np.array(img_resized)
            img_normalized = img_array / 255.0
            
            # Lưu lại
            output_path = os.path.join(output_dir, filename)
            np.save(output_path.replace('.png', '.npy'), img_normalized)

# Sử dụng
prepare_data("raw_images/", "processed_images/")
```

---

## 💡 Ví dụ thực tế

### Scenario: Chuẩn bị dataset Mayo Clinic

```
Bước 1: Tải dataset
        → Có file DICOM

Bước 2: Convert DICOM → PNG
        → Dùng pydicom + PIL
        → Có 10,000 ảnh PNG

Bước 3: Resize về 256×256
        → Tất cả ảnh cùng kích thước

Bước 4: Normalize
        → Giá trị về 0-1

Bước 5: Tạo cặp
        → Đảm bảo nhiễu-sạch khớp

Bước 6: Chia train/val/test
        → 8,000 / 1,000 / 1,000

Bước 7: Data augmentation
        → Tăng train lên 32,000 ảnh

→ Sẵn sàng train! ✅
```

---

## 🎯 Tóm tắt

**Quy trình chuẩn bị**:

```
1. Thu thập ảnh
2. Chuyển đổi định dạng (DICOM → PNG)
3. Resize về cùng kích thước
4. Normalize giá trị pixel
5. Tạo cặp ảnh (nhiễu + sạch)
6. Chia train/val/test
7. Data augmentation (nếu cần)
```

**Yêu cầu**:
- ✅ Tất cả ảnh cùng kích thước
- ✅ Giá trị pixel đã normalize
- ✅ Ảnh nhiễu và sạch tương ứng
- ✅ Đã chia train/val/test

**Kết quả**: Dữ liệu sẵn sàng để train! 🎉

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Các bước chuẩn bị dữ liệu là gì?
2. Tại sao cần resize ảnh?
3. Normalize là gì? Tại sao cần?
4. Tại sao cần chia train/val/test?
5. Data augmentation là gì? Khi nào dùng?

**Hoạt động**: 
- Thực hành resize ảnh
- Thực hành normalize
- Tạo cặp ảnh và chia dataset

---

➡️ **Tiếp theo**: `CAU_HOI_20.md` - Các tham số training quan trọng là gì?

