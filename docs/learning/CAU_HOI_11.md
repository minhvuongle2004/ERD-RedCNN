# Câu 11: Model học như thế nào?

## 🎯 Mục tiêu
Hiểu cơ bản về quá trình training (huấn luyện) model

---

## 📖 Trả lời

### Training = Dạy AI học

**Giống như dạy trẻ con**:
```
1. Cho xem ví dụ: "Đây là mèo 🐱"
2. Hỏi: "Đây là con gì?" → Trả lời sai
3. Sửa: "Sai rồi, đây là mèo"
4. Lặp lại nhiều lần...
5. Cuối cùng: Nhận biết được mèo!
```

**Áp dụng cho RED-CNN**:
```
1. Cho xem ví dụ: "Ảnh nhiễu này → ảnh sạch này"
2. Hỏi: "Khử nhiễu ảnh này?" → Kết quả chưa tốt
3. Sửa: "Sai rồi, phải thế này mới đúng"
4. Lặp lại hàng ngàn lần...
5. Cuối cùng: Khử nhiễu giỏi!
```

---

## 📚 Quá trình Training chi tiết

### Bước 1: Chuẩn bị dữ liệu

Cần **NHIỀU cặp ảnh**:
```
Cặp 1:  Ảnh nhiễu (X)  +  Ảnh sạch (Y)
        🖼️ ▓▒░            🖼️ ✨

Cặp 2:  Ảnh nhiễu (X)  +  Ảnh sạch (Y)
        🖼️ ▓▒░            🖼️ ✨

...

Cặp 10,000:  Ảnh nhiễu (X)  +  Ảnh sạch (Y)
```

**Trong dự án này**: Dùng dataset Mayo Clinic (hàng ngàn cặp ảnh CT thật)

### Bước 2: Cho Model dự đoán

```
Iteration 1:
    Ảnh nhiễu  →  [RED-CNN]  →  Ảnh dự đoán
    🖼️ ▓▒░                       🖼️ ?? (chưa tốt)
```

### Bước 3: So sánh với đáp án đúng

```
Ảnh dự đoán  vs  Ảnh sạch thật (Ground truth)
   🖼️ ??           🖼️ ✨

    ↓ Tính độ khác biệt
    
LOSS (Sai số) = Cao
"Model đoán sai nhiều!"
```

### Bước 4: Điều chỉnh Model

```
LOSS cao  →  Điều chỉnh Model  →  Thử lại

(Quá trình toán học phức tạp, không cần hiểu chi tiết)
```

### Bước 5: Lặp lại

```
Iteration 1:  LOSS = Cao      (Kém)
Iteration 100:  LOSS = Giảm   (Khá hơn)
Iteration 1000: LOSS = Thấp   (Tốt)
Iteration 10000: LOSS = Rất thấp (Rất tốt!)
```

---

## 🔁 Vòng lặp Training

```
┌──────────────────────────────────────┐
│ 1. Cho ảnh nhiễu vào Model            │
│         ↓                             │
│ 2. Model dự đoán ảnh sạch             │
│         ↓                             │
│ 3. So sánh với ảnh sạch thật          │
│         ↓                             │
│ 4. Tính LOSS (sai số)                 │
│         ↓                             │
│ 5. Điều chỉnh Model để giảm LOSS     │
│         ↓                             │
└─────────┼────────────────────────────┘
          │
          └──→ Lặp lại hàng ngàn lần
```

**Mỗi vòng lặp = 1 iteration**

---

## 📈 Quá trình học qua thời gian

### Ban đầu (Iteration 0-100):

```
Input:               Output:              Target:
🖼️ ▓▒░ Nhiễu  →    🖼️ ▓░ Vẫn nhiều  ≠   🖼️ ✨ Sạch
                    
LOSS = Cao (Model chưa biết làm gì)
```

### Giữa chừng (Iteration 1000-5000):

```
Input:               Output:              Target:
🖼️ ▓▒░ Nhiễu  →    🖼️ ░ Ít nhiễu   ≈   🖼️ ✨ Sạch
                    
LOSS = Trung bình (Model đang học)
```

### Cuối cùng (Iteration 90000+):

```
Input:               Output:              Target:
🖼️ ▓▒░ Nhiễu  →    🖼️ ✨ Gần sạch  ≈≈  🖼️ ✨ Sạch
                    
LOSS = Thấp (Model học tốt rồi!)
```

---

## 🎯 Mục tiêu của Training

**Minimize LOSS** = Giảm sai số xuống thấp nhất có thể

```
LOSS

Cao ^                *
    |              *
    |            *
    |          *        ← Model đang học
    |        *
    |      *
Thấp|    *____________  ← Đạt mục tiêu
    └──────────────────────> Iterations
    0     10K    90K
```

---

## 💡 Ví dụ thực tế dễ hiểu

### Giống như học lái xe:

```
Lần 1: Lái xe đâm vào tường  ← Sai số lớn
        ↓ Học
Lần 10: Lái xe còn vượt làn  ← Sai số trung bình
        ↓ Học tiếp
Lần 100: Lái xe đúng làn     ← Sai số nhỏ
        ↓ Luyện tập
Lần 1000: Lái xe thành thạo  ← Gần như không sai
```

**RED-CNN cũng vậy**: Càng train nhiều, càng giỏi khử nhiễu

---

## ⚙️ Các yếu tố quan trọng trong Training

### 1. Dataset (Dữ liệu)

```
✅ Nhiều ảnh (hàng ngàn)
✅ Đa dạng (nhiều loại ca bệnh)
✅ Chất lượng tốt (ảnh sạch thật sự sạch)
```

### 2. Iterations (Số lần lặp)

```
Ít:     1,000 iterations   → Model chưa học đủ
Vừa:    10,000 iterations  → Model học tạm được
Nhiều:  90,000 iterations  → Model học tốt
```

### 3. Learning Rate (Tốc độ học)

```
Nhanh:  Học nhanh nhưng có thể bỏ sót
Chậm:   Học chậm nhưng kỹ càng
Vừa:    Cân bằng ⚖️
```

---

## 📊 Tóm tắt

**Training = Dạy Model học**

**Quy trình**:
```
Dữ liệu (nhiều cặp ảnh)
         ↓
┌────────────────────┐
│ Vòng lặp Training:  │
│ 1. Dự đoán          │
│ 2. So sánh          │
│ 3. Tính LOSS       │
│ 4. Điều chỉnh      │
└────────────────────┘
         ↓ (lặp lại hàng ngàn lần)
Model đã học xong
```

**Kết quả**: Model biết cách khử nhiễu tốt!

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Training là gì?
2. Cần gì để training model?
3. Vòng lặp training có mấy bước chính?
4. LOSS là gì? Mục tiêu là gì?
5. Tại sao cần train nhiều iterations?

**Ví dụ**: Cho sinh viên tự nghĩ ra ví dụ khác về quá trình học (học nhạc, học thể thao...)

---

➡️ **Tiếp theo**: `CAU_HOI_12.md` - Làm sao biết model đang học tốt?
