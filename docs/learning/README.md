# 📚 Tài liệu học RED-CNN theo phương pháp Hỏi-Đáp

## 🎯 Phương pháp này dành cho ai?

- ✅ Người dạy KHÔNG phải chuyên gia
- ✅ Sinh viên mới bắt đầu, chưa biết code
- ✅ Muốn hiểu **CÁCH HOẠT ĐỘNG**, chưa cần hiểu code

---

## 📖 Cách sử dụng

### Cho người dạy:

1. **Mở file `DANH_SACH_CAU_HOI.md`**
   - Xem danh sách 16 câu hỏi theo thứ tự
   - Mỗi câu có mục tiêu rõ ràng

2. **Dạy theo thứ tự từng câu**
   - Mở file `CAU_HOI_XX.md` tương ứng
   - Đọc và giải thích cho sinh viên
   - Không skip câu, dạy tuần tự

3. **Check understanding**
   - Cuối mỗi câu có phần "Kiểm tra hiểu bài"
   - Hỏi lại sinh viên xem đã hiểu chưa

### Cho sinh viên:

1. **Đọc theo thứ tự** từ Câu 1 đến Câu 16
2. **Không vội**, mỗi câu 10-15 phút
3. **Vẽ sơ đồ** trên giấy để hiểu rõ hơn
4. **Tự hỏi lại** các câu hỏi ở cuối mỗi file

---

## 📋 Danh sách câu hỏi (16 câu)

### 📘 Phần 1: Hiểu vấn đề (3 câu)

| # | Câu hỏi | File | Thời gian |
|---|---------|------|-----------|
| 1 | Ảnh CT là gì và tại sao cần nó? | `CAU_HOI_01.md` | 10 phút |
| 2 | Vấn đề gì khi giảm liều bức xạ? | `CAU_HOI_02.md` | 10 phút |
| 3 | AI giúp được gì? | `CAU_HOI_03.md` | 15 phút |

**Sau phần này, sinh viên biết**: Vấn đề LDCT denoising là gì

---

### 📗 Phần 2: Hiểu RED-CNN làm gì (3 câu)

| # | Câu hỏi | File | Thời gian |
|---|---------|------|-----------|
| 4 | RED-CNN nhận input/output gì? | `CAU_HOI_04.md` | 10 phút |
| 5 | RED-CNN hoạt động như thế nào? | `CAU_HOI_05.md` | 15 phút |
| 6 | Tại sao gọi là "Encoder-Decoder"? | `CAU_HOI_06.md` | 10 phút |

**Sau phần này, sinh viên biết**: RED-CNN làm gì và flow cơ bản

---

### 📙 Phần 3: Hiểu các bước xử lý (4 câu)

| # | Câu hỏi | File | Thời gian |
|---|---------|------|-----------|
| 7 | Encoder làm gì với ảnh? | `CAU_HOI_07.md` | 10 phút |
| 8 | "Bottleneck" là gì? | `CAU_HOI_08.md` | 10 phút |
| 9 | Decoder làm gì? | `CAU_HOI_09.md` | 10 phút |
| 10 | Skip connections là gì? | `CAU_HOI_10.md` | 15 phút |

**Sau phần này, sinh viên biết**: Chi tiết từng bước xử lý của RED-CNN

---

### 📕 Phần 4: Hiểu training (3 câu)

| # | Câu hỏi | File | Thời gian |
|---|---------|------|-----------|
| 11 | Model học như thế nào? | `CAU_HOI_11.md` | 15 phút |
| 12 | Biết model học tốt bằng cách nào? | `CAU_HOI_12.md` | 10 phút |
| 13 | Training xong, dùng thế nào? | `CAU_HOI_13.md` | 10 phút |

**Sau phần này, sinh viên biết**: Cơ bản về training và sử dụng model

---

### 📓 Phần 5: Hiểu tại sao (3 câu)

| # | Câu hỏi | File | Thời gian |
|---|---------|------|-----------|
| 14 | Tại sao RED-CNN hiệu quả? | `CAU_HOI_14.md` | 10 phút |
| 15 | RED-CNN có hạn chế gì? | `CAU_HOI_15.md` | 10 phút |
| 16 | So với phương pháp khác thì sao? | `CAU_HOI_16.md` | 10 phút |

**Sau phần này, sinh viên biết**: Ưu/nhược điểm và bối cảnh của RED-CNN

---

## ⏱️ Thời gian học

**Tổng thời gian**: 3-4 giờ

**Khuyến nghị**:
- **Buổi 1** (1 giờ): Câu 1-3 (Hiểu vấn đề)
- **Buổi 2** (1 giờ): Câu 4-6 (Hiểu RED-CNN làm gì)
- **Buổi 3** (1.5 giờ): Câu 7-10 (Chi tiết xử lý)
- **Buổi 4** (1 giờ): Câu 11-13 (Training)
- **Buổi 5** (0.5 giờ): Câu 14-16 (Tại sao)

---

## 📊 Checklist sau khi học xong

Sinh viên cần trả lời được:

### ✅ Về vấn đề:
- [ ] CT scan là gì?
- [ ] Vấn đề của bức xạ trong CT?
- [ ] AI giúp được gì?

### ✅ Về RED-CNN:
- [ ] RED-CNN nhận input gì, cho output gì?
- [ ] 3 bước xử lý chính là gì?
- [ ] Encoder làm gì? Decoder làm gì?
- [ ] Bottleneck là gì?
- [ ] Skip connections là gì và tại sao cần?

### ✅ Về Training:
- [ ] Training là gì?
- [ ] Cần gì để training?
- [ ] LOSS là gì?
- [ ] Sau training, dùng model như thế nào?

### ✅ Về ưu/nhược điểm:
- [ ] Tại sao RED-CNN hiệu quả?
- [ ] RED-CNN có hạn chế gì?

---

## 🎨 Tips cho việc dạy

### 1. Dùng hình vẽ

Mỗi câu hỏi đã có ASCII art, nhưng nên:
- Vẽ lại trên bảng/giấy
- Cho sinh viên tự vẽ
- Dùng màu sắc khác nhau

### 2. Ví dụ thực tế

So sánh với:
- Chỉnh ảnh trên điện thoại
- Nén file ZIP
- Học lái xe
- ...

### 3. Check understanding liên tục

Sau mỗi 2-3 câu, hỏi lại:
- "Vậy Encoder làm gì?"
- "Tại sao cần skip connections?"
- "Cho ví dụ thực tế?"

### 4. Không vội vàng

- 1 buổi chỉ dạy 3-4 câu
- Cho thời gian suy nghĩ
- Khuyến khích hỏi

### 5. Tạo không khí thoải mái

- "Không có câu hỏi ngớ ngẩn"
- "Mình cũng đang học cùng các bạn"
- "Hỏi bất cứ lúc nào"

---

## 🚫 Chưa cần trong giai đoạn này

- ❌ Hiểu code Python
- ❌ Tính toán dimensions
- ❌ Công thức toán phức tạp
- ❌ Chạy training
- ❌ Debug errors

**Chỉ cần**: Hiểu KHÁI NIỆM và CÁCH HOẠT ĐỘNG

---

## 📁 Cấu trúc thư mục

```
docs/learning/
├── README.md                 ← File này
├── DANH_SACH_CAU_HOI.md     ← Danh sách 16 câu
├── CAU_HOI_01.md            ← Trả lời câu 1
├── CAU_HOI_02.md            ← Trả lời câu 2
├── ...
└── CAU_HOI_16.md            ← Trả lời câu 16
```

---

## 🎯 Mục tiêu cuối cùng

Sau khi học xong 16 câu, sinh viên có thể:

1. ✅ **Giải thích** cho người khác về RED-CNN
2. ✅ **Vẽ sơ đồ** cách RED-CNN hoạt động
3. ✅ **Hiểu** tại sao RED-CNN hiệu quả
4. ✅ **Sẵn sàng** học code (nếu muốn)

---

## ➡️ Bước tiếp theo (Optional)

Nếu sinh viên muốn học sâu hơn:

1. **Xem code**: `../RED-CNN_TUTORIAL_VI.md`
2. **Chạy visualization**: `../../scripts/visualize_redcnn.py`
3. **Đọc paper gốc**: Chen et al., 2017

Nhưng **không bắt buộc**!

---

**Chúc bạn dạy/học thành công! 🎓**

*Có thắc mắc? Xem lại `DANH_SACH_CAU_HOI.md` và file câu hỏi tương ứng.*
