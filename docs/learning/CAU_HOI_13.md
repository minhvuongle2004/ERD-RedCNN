# Câu 13: Training xong, dùng model như thế nào?

## 🎯 Mục tiêu
Hiểu được:
- Inference là gì
- Quy trình sử dụng model đã train
- Deployment trong thực tế

---

## 📖 Trả lời

### Training xong rồi, làm gì tiếp?

**Giống như học xong lái xe**:
```
Training:  Học lái xe (hàng trăm giờ)
     ↓
Đã học xong: Có bằng lái
     ↓
Inference:  Lái xe thật trên đường! 🚗
```

**Áp dụng vào RED-CNN**:
```
Training:  Dạy model khử nhiễu (hàng ngàn ảnh)
     ↓
Đã train xong: Model đã học được
     ↓
Inference:  Dùng model khử nhiễu ảnh mới! 🖼️
```

---

## 🔄 Training vs Inference

### Training (Huấn luyện)

```
Input: Ảnh nhiễu + Ảnh sạch (đáp án)
  ↓
Model học cách khử nhiễu
  ↓
Điều chỉnh Model (update weights)
  ↓
Lặp lại hàng ngàn lần
```

**Đặc điểm**:
- ✅ Cần đáp án (ground truth)
- ✅ Điều chỉnh Model
- ✅ Mất nhiều thời gian
- ✅ Cần GPU mạnh

### Inference (Sử dụng)

```
Input: Ảnh nhiễu (KHÔNG có đáp án)
  ↓
Model xử lý (KHÔNG điều chỉnh)
  ↓
Output: Ảnh sạch
```

**Đặc điểm**:
- ✅ KHÔNG cần đáp án
- ✅ KHÔNG điều chỉnh Model
- ✅ Nhanh (vài giây)
- ✅ Có thể chạy trên CPU/GPU

---

## 🚀 Quy trình Inference

### Bước 1: Load Model đã train

```
Model đã train = File .pth hoặc .h5
(chứa tất cả "kiến thức" model đã học)

Load vào bộ nhớ:
┌─────────────┐
│ Model.pth     │ → Load → Model sẵn sàng
└─────────────┘
```

**Giống như**: Mở app trên điện thoại (đã cài sẵn)

### Bước 2: Chuẩn bị ảnh input

```
Ảnh CT mới (nhiễu):
┌──────────────┐
│ 🖼️ ▓▒░ Nhiễu │
│   🫁 🫁      │
│     ❤️       │
└──────────────┘

Chuẩn bị:
- Resize về 128×128 (nếu cần)
- Chuyển sang grayscale
- Normalize (chuẩn hóa)
```

### Bước 3: Cho vào Model

```
Ảnh nhiễu → [RED-CNN Model] → Ảnh sạch
🖼️ ▓▒░                      🖼️ ✨
```

**Quá trình bên trong**:
```
Input (128×128, nhiễu)
  ↓
[Encoder] → Nén, loại nhiễu
  ↓
[Bottleneck] → Thông tin sạch
  ↓
[Decoder] → Khôi phục chi tiết
  ↓
Output (128×128, sạch) ✨
```

### Bước 4: Lấy kết quả

```
Output: Ảnh sạch
┌──────────────┐
│ 🖼️ ✨ Sạch   │
│   🫁 🫁      │
│     ❤️       │
└──────────────┘

→ Lưu file hoặc hiển thị
```

---

## 💻 Code đơn giản (Minh họa)

### Pseudocode (Không phải code thật)

```
1. Load Model:
   model = load_model("red_cnn_trained.pth")

2. Load ảnh:
   image = load_image("ct_scan_noisy.png")

3. Chuẩn bị:
   image = resize(image, 128x128)
   image = normalize(image)

4. Inference:
   clean_image = model.predict(image)

5. Lưu kết quả:
   save_image(clean_image, "ct_scan_clean.png")
```

**Thời gian**: Vài giây cho 1 ảnh (tùy GPU/CPU)

---

## 🏥 Ứng dụng thực tế

### Scenario 1: Bệnh viện

```
Bước 1: Bệnh nhân chụp CT (liều thấp, nhiễu)
        ↓
Bước 2: Upload ảnh lên hệ thống
        ↓
Bước 3: RED-CNN xử lý (tự động)
        ↓
Bước 4: Bác sĩ xem ảnh sạch
        ↓
Bước 5: Chẩn đoán chính xác hơn ✅
```

**Thời gian**: 5-10 giây/ảnh

### Scenario 2: Batch Processing

```
Có 100 ảnh CT cần xử lý:

Ảnh 1 → Model → Ảnh sạch 1
Ảnh 2 → Model → Ảnh sạch 2
...
Ảnh 100 → Model → Ảnh sạch 100

Tổng thời gian: ~10 phút (cho 100 ảnh)
```

---

## 🔧 Deployment (Triển khai)

### Deployment = Đưa Model vào sử dụng thật

**3 cách phổ biến**:

### 1. Local Deployment (Trên máy tính)

```
Máy tính bác sĩ:
┌─────────────────┐
│ RED-CNN Model    │
│ (đã cài sẵn)     │
│                  │
│ Ảnh CT → Model → │
│ Ảnh sạch         │
└─────────────────┘
```

**Ưu điểm**:
- ✅ Nhanh (không cần internet)
- ✅ Bảo mật (dữ liệu không gửi đi)

**Nhược điểm**:
- ❌ Cần cài đặt trên mỗi máy
- ❌ Khó cập nhật

### 2. Server Deployment (Trên server)

```
Bác sĩ gửi ảnh → Server (có Model) → Trả về ảnh sạch
     📱              🖥️                    📱
```

**Ưu điểm**:
- ✅ Dễ cập nhật (chỉ cần update server)
- ✅ Nhiều người dùng cùng lúc
- ✅ Quản lý tập trung

**Nhược điểm**:
- ❌ Cần internet
- ❌ Có thể chậm nếu nhiều người dùng

### 3. Cloud Deployment (Trên đám mây)

```
Bác sĩ → Cloud (AWS/Azure/GCP) → Ảnh sạch
  📱         ☁️                        📱
```

**Ưu điểm**:
- ✅ Không cần server riêng
- ✅ Tự động scale (xử lý nhiều ảnh)
- ✅ Dễ bảo trì

**Nhược điểm**:
- ❌ Cần internet
- ❌ Có thể tốn phí

---

## ⚡ Tối ưu hóa Inference

### 1. Batch Processing

```
Xử lý 1 ảnh:  5 giây
Xử lý 10 ảnh: 10 giây (nhanh hơn 10×5=50 giây!)

→ Xử lý nhiều ảnh cùng lúc = Nhanh hơn
```

### 2. GPU Acceleration

```
CPU:  10 giây/ảnh
GPU:  0.5 giây/ảnh  ← Nhanh hơn 20 lần!

→ Dùng GPU = Nhanh hơn rất nhiều
```

### 3. Model Quantization

```
Model gốc:  100 MB, chạy chậm
Model nhỏ:  25 MB, chạy nhanh hơn 4 lần

→ Giảm kích thước model = Nhanh hơn
```

---

## 🎯 So sánh Training vs Inference

```
┌─────────────────────────────────────┐
│  TRAINING        vs    INFERENCE    │
├─────────────────────────────────────┤
│ Cần đáp án      Không cần đáp án   │
│ Điều chỉnh Model Không điều chỉnh   │
│ Chậm (giờ)      Nhanh (giây)       │
│ Cần GPU mạnh    CPU/GPU đều được    │
│ 1 lần duy nhất  Dùng nhiều lần     │
│ Học kiến thức   Áp dụng kiến thức  │
└─────────────────────────────────────┘
```

---

## 💡 Ví dụ thực tế dễ hiểu

### Giống như học nấu ăn

```
TRAINING:
- Học công thức (hàng trăm lần)
- Nếm thử, điều chỉnh
- Cuối cùng: Biết nấu ngon

INFERENCE:
- Có khách đến
- Nấu món ăn (dùng công thức đã học)
- Không cần học lại!
```

### Giống như học lái xe

```
TRAINING:
- Học lái xe (hàng trăm giờ)
- Có thầy sửa lỗi
- Cuối cùng: Có bằng lái

INFERENCE:
- Lái xe thật trên đường
- Không cần thầy nữa
- Tự lái được!
```

---

## 🔒 Lưu ý quan trọng

### 1. Model phải được train tốt

```
Model train kém → Inference kém
Model train tốt → Inference tốt ✅
```

### 2. Input phải giống training data

```
Training: Ảnh CT phổi 128×128
Inference: Ảnh CT phổi 128×128 ✅

Training: Ảnh CT phổi
Inference: Ảnh X-quang ❌ (Sai loại!)
```

### 3. Không cần train lại

```
Inference = Chỉ dùng model
KHÔNG điều chỉnh model!

→ Model giữ nguyên sau khi train xong
```

---

## 📊 Tóm tắt

**Inference = Dùng Model đã train**

**Quy trình**:
```
1. Load Model đã train
2. Chuẩn bị ảnh input
3. Cho vào Model
4. Lấy ảnh sạch
```

**Deployment**:
- ✅ Local (máy tính)
- ✅ Server (máy chủ)
- ✅ Cloud (đám mây)

**Tối ưu**:
- ✅ Batch processing
- ✅ GPU acceleration
- ✅ Model quantization

**Kết quả**: 
- ✅ Xử lý nhanh (vài giây/ảnh)
- ✅ Không cần đáp án
- ✅ Dùng được nhiều lần
- ✅ Áp dụng vào thực tế! 🎉

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Inference là gì? Khác Training như thế nào?
2. Quy trình inference có mấy bước?
3. Deployment là gì? Có mấy cách?
4. Tại sao inference nhanh hơn training?
5. Khi nào cần train lại model?

**Hoạt động**: 
- Vẽ sơ đồ quy trình inference
- So sánh training vs inference
- Nghĩ ra scenario sử dụng model trong thực tế

---

➡️ **Tiếp theo**: `CAU_HOI_14.md` - Tại sao RED-CNN hiệu quả cho CT denoising?

