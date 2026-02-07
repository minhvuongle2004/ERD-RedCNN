# Câu 18: Dataset là gì và cần như thế nào?

## 🎯 Mục tiêu
Hiểu được:
- Dataset là gì
- Dataset cần như thế nào cho RED-CNN
- Cách tìm và sử dụng dataset

---

## 📖 Trả lời

### Dataset = Tập dữ liệu

**Đơn giản**: Dataset là **tập hợp nhiều dữ liệu** để train model

**Giống như**:
```
Học sinh học toán:
→ Cần sách bài tập (dataset)
→ Làm nhiều bài (train)
→ Cuối cùng: Giỏi toán! ✅
```

**Áp dụng vào RED-CNN**:
```
Model học khử nhiễu:
→ Cần nhiều cặp ảnh (dataset)
→ Train trên các cặp ảnh
→ Cuối cùng: Biết khử nhiễu! ✅
```

---

## 🖼️ Dataset cho RED-CNN

### Dataset cần gì?

**Cần CẶP ẢNH**:
```
Cặp 1:
  Ảnh nhiễu (Input)    +    Ảnh sạch (Ground truth)
  🖼️ ▓▒░ Nhiễu              🖼️ ✨ Sạch

Cặp 2:
  Ảnh nhiễu (Input)    +    Ảnh sạch (Ground truth)
  🖼️ ▓▒░ Nhiễu              🖼️ ✨ Sạch

...

Cặp 10,000:
  Ảnh nhiễu (Input)    +    Ảnh sạch (Ground truth)
  🖼️ ▓▒░ Nhiễu              🖼️ ✨ Sạch
```

**Quan trọng**:
- ✅ Ảnh nhiễu và ảnh sạch phải **TƯƠNG ỨNG** nhau
- ✅ Cùng bệnh nhân, cùng vị trí, cùng thời điểm
- ✅ Chỉ khác nhau về mức độ nhiễu

---

## 📊 Yêu cầu về số lượng

### Cần bao nhiêu ảnh?

```
Ít:     100-500 cặp    → Model học không tốt ❌
Vừa:    1,000-5,000   → Model học tạm được ⚠️
Nhiều:  10,000+        → Model học tốt ✅
Rất nhiều: 50,000+    → Model học rất tốt ✅✅
```

**Tại sao cần nhiều?**:
```
Model phức tạp (nhiều layers)
→ Cần nhiều ví dụ để học
→ Ít dữ liệu = Không học được pattern
→ Nhiều dữ liệu = Học được pattern tốt
```

**Ví dụ**:
```
10 cặp ảnh:
→ Model chỉ nhớ 10 ảnh này
→ Ảnh mới → Xử lý kém ❌

10,000 cặp ảnh:
→ Model học được pattern tổng quát
→ Ảnh mới → Xử lý tốt ✅
```

---

## 🏥 Dataset phổ biến

### 1. Mayo Clinic Dataset

**Thông tin**:
```
- Dataset công khai cho CT denoising
- Có sẵn ảnh CT thật
- Đã được dùng trong nhiều nghiên cứu
- Có thể tải về miễn phí (sau khi đăng ký)
```

**Bao gồm**:
- ✅ Ảnh CT phổi
- ✅ Cặp ảnh nhiễu-sạch
- ✅ Đã được xử lý sẵn

**Cách tải**:
- Tìm "Mayo Clinic LDCT Dataset" trên Google
- Đăng ký và tải về

---

### 2. Dữ liệu riêng

**Nếu có dữ liệu từ bệnh viện**:
```
Ưu điểm:
✅ Phù hợp với mục đích cụ thể
✅ Có thể có nhiều loại ảnh
✅ Kiểm soát chất lượng

Nhược điểm:
❌ Cần xin phép
❌ Cần xử lý (anonymize)
❌ Cần đánh dấu ảnh sạch
```

**Cách chuẩn bị**:
1. Thu thập ảnh CT (có nhiễu)
2. Tạo ảnh sạch tương ứng (chụp lại với liều cao, hoặc xử lý)
3. Ghép thành cặp
4. Lưu vào thư mục

---

## 📁 Cấu trúc Dataset

### Cấu trúc thư mục thường dùng

```
dataset/
├── train/
│   ├── noisy/          # Ảnh nhiễu training
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── clean/          # Ảnh sạch training
│       ├── img_001.png
│       ├── img_002.png
│       └── ...
├── val/
│   ├── noisy/          # Ảnh nhiễu validation
│   └── clean/           # Ảnh sạch validation
└── test/
    ├── noisy/           # Ảnh nhiễu test
    └── clean/           # Ảnh sạch test
```

**Quan trọng**:
- ✅ Tên file phải khớp nhau (img_001.png trong noisy và clean)
- ✅ Chia train/val/test (80/10/10 hoặc 70/15/15)

---

## 🎯 Chia Dataset

### Tại sao cần chia?

```
Train set:    Dùng để train model
              → Model học từ đây

Validation set: Dùng để kiểm tra khi train
                → Xem model có học tốt không

Test set:     Dùng để đánh giá cuối cùng
              → Xem model tốt thật sự không
```

**Ví dụ với 10,000 ảnh**:
```
Train:        8,000 ảnh (80%)
Validation:    1,000 ảnh (10%)
Test:          1,000 ảnh (10%)
```

**Tại sao không dùng hết để train?**:
```
Dùng hết để train:
→ Model có thể "học thuộc lòng"
→ Test trên ảnh mới → Kém ❌

Chia train/test:
→ Model học tổng quát
→ Test trên ảnh mới → Tốt ✅
```

---

## 📏 Yêu cầu về chất lượng

### 1. Kích thước ảnh

```
Phổ biến:
- 128×128 pixels
- 256×256 pixels
- 512×512 pixels

Quan trọng:
✅ Tất cả ảnh phải cùng kích thước
✅ Kích thước phải là số chẵn (128, 256, 512)
```

**Tại sao?**:
```
Model RED-CNN được thiết kế cho kích thước cụ thể
→ Ảnh khác kích thước → Phải resize
→ Resize có thể làm mất chi tiết
```

---

### 2. Định dạng ảnh

```
Phổ biến:
- PNG (không mất chất lượng) ✅
- TIFF (không mất chất lượng) ✅
- JPEG (có mất chất lượng) ⚠️

Khuyến khích: PNG hoặc TIFF
```

---

### 3. Chất lượng ảnh

```
Ảnh nhiễu:
✅ Phải thật sự có nhiễu (không quá sạch)
✅ Nhiễu phải tự nhiên (từ CT thật)

Ảnh sạch:
✅ Phải thật sự sạch (ground truth)
✅ Phải tương ứng với ảnh nhiễu
✅ Không được có nhiễu
```

---

## 🔍 Kiểm tra Dataset

### Checklist

```
□ 1. Có đủ số lượng (ít nhất 1,000 cặp)
□ 2. Ảnh nhiễu và sạch tương ứng nhau
□ 3. Tất cả ảnh cùng kích thước
□ 4. Đã chia train/val/test
□ 5. Tên file khớp nhau
□ 6. Ảnh không bị lỗi (có thể đọc được)
□ 7. Ảnh sạch thật sự sạch
□ 8. Ảnh nhiễu thật sự có nhiễu
```

---

## 💡 Ví dụ cụ thể

### Ví dụ 1: Dataset Mayo Clinic

```
Có sẵn:
- 10,000 cặp ảnh CT phổi
- Kích thước: 512×512
- Định dạng: DICOM (cần convert sang PNG)

Sau khi xử lý:
dataset/
├── train/
│   ├── noisy/ (8,000 ảnh)
│   └── clean/ (8,000 ảnh)
├── val/
│   ├── noisy/ (1,000 ảnh)
│   └── clean/ (1,000 ảnh)
└── test/
    ├── noisy/ (1,000 ảnh)
    └── clean/ (1,000 ảnh)

→ Sẵn sàng train! ✅
```

---

### Ví dụ 2: Dataset nhỏ (1,000 ảnh)

```
Chia:
- Train: 700 ảnh (70%)
- Val:   150 ảnh (15%)
- Test:  150 ảnh (15%)

→ Có thể train, nhưng kết quả không tốt bằng dataset lớn
```

---

## 🎯 Tóm tắt

**Dataset là gì?**:
- Tập hợp cặp ảnh (nhiễu + sạch)
- Dùng để train model

**Yêu cầu**:
- ✅ Ít nhất 1,000 cặp (tốt nhất 10,000+)
- ✅ Ảnh nhiễu và sạch tương ứng nhau
- ✅ Cùng kích thước (128×128, 256×256, 512×512)
- ✅ Chia train/val/test (80/10/10)

**Dataset phổ biến**:
- ✅ Mayo Clinic Dataset (công khai)
- ✅ Dữ liệu riêng (từ bệnh viện)

**Cấu trúc**:
```
dataset/
├── train/ (noisy + clean)
├── val/ (noisy + clean)
└── test/ (noisy + clean)
```

**Kết quả**: Có dataset tốt = Train model tốt! 🎉

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Dataset là gì? Cần như thế nào?
2. Tại sao cần nhiều ảnh?
3. Tại sao cần chia train/val/test?
4. Dataset cần đáp ứng yêu cầu gì?
5. Làm sao kiểm tra dataset đã đúng chưa?

**Hoạt động**: 
- Xem cấu trúc dataset thật
- Kiểm tra số lượng ảnh
- Kiểm tra ảnh nhiễu và sạch có tương ứng không

---

➡️ **Tiếp theo**: `CAU_HOI_19.md` - Cách chuẩn bị dữ liệu training?

