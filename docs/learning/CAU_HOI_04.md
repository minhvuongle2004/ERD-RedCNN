# Câu 4: RED-CNN nhận input gì và cho output gì?

## 🎯 Mục tiêu
Hiểu được:
- Input của RED-CNN là gì
- Output của RED-CNN là gì
- Quá trình chuyển đổi cơ bản

---

## 📖 Trả lời

### Input (Đầu vào)

**RED-CNN nhận vào**: Ảnh CT liều thấp (có nhiễu)

```
┌──────────────────────┐
│   🖼️ Ảnh CT liều thấp │
│                       │
│   Đặc điểm:          │
│   • Có nhiều nhiễu ▓▒░│
│   • Mờ, không rõ ràng│
│   • Grayscale (xám)  │
│   • Kích thước cố định│
└──────────────────────┘
```

**Ví dụ**: Ảnh phổi với nhiễu
```
Input:
┌───────────────┐
│  ▓🫁▒🫁░       │  ← Nhiều hạt nhiễu
│  ░▒ ❤️ ▓       │  ← Biên giới không rõ
│  ▒░🦴▓        │  ← Khó nhìn chi tiết
└───────────────┘
```

### Output (Đầu ra)

**RED-CNN cho ra**: Ảnh CT đã khử nhiễu (sạch)

```
┌──────────────────────┐
│   🖼️ Ảnh CT sạch      │
│                       │
│   Đặc điểm:          │
│   • Ít nhiễu ✨       │
│   • Rõ nét, sắc nét   │
│   • Vẫn là Grayscale │
│   • Cùng kích thước   │
└──────────────────────┘
```

**Ví dụ**: Cùng ảnh phổi nhưng đã sạch
```
Output:
┌───────────────┐
│   🫁  🫁       │  ← Không còn nhiễu
│     ❤️         │  ← Biên giới rõ ràng
│   🦴          │  ← Chi tiết rõ nét
└───────────────┘
```

### Quá trình chuyển đổi

```
       INPUT                    OUTPUT
(Ảnh CT liều thấp)         (Ảnh CT sạch)

┌──────────────┐          ┌──────────────┐
│  🫁▓▒🫁░      │          │  🫁  🫁       │
│  ░ ❤️▒▓      │   RED    │    ❤️         │
│  ▓🦴░         │  ─────→  │  🦴          │
│   ▓▒░        │   CNN    │              │
└──────────────┘          └──────────────┘

     Nhiễu                    Sạch
```

### Điểm quan trọng

**1. Kích thước giống nhau**
```
Input:  128 × 128 pixels
        ↓
    [RED-CNN]
        ↓
Output: 128 × 128 pixels  ← Same size!
```

**2. Nội dung không đổi**
```
✅ Vị trí phổi vẫn như cũ
✅ Kích thước tim không thay đổi
✅ Chỉ LOẠI BỎ NHIỄU, không thêm/bớt nội dung
```

**3. Chỉ xử lý 1 ảnh (slice) mỗi lần**
```
CT scan đầy đủ = Nhiều slices (lát cắt):
┌────┬────┬────┬────┐
│ 📄 │ 📄 │ 📄 │ 📄 │  ← Nhiều lát cắt
└────┴────┴────┴────┘

RED-CNN xử lý từng lát một:
    📄  →  [RED-CNN]  →  📄✨
    📄  →  [RED-CNN]  →  📄✨
    📄  →  [RED-CNN]  →  📄✨
```

---

## 🔍 Chi tiết hơn về Input

### Dữ liệu thô

```
Ảnh CT = Ma trận số

Ví dụ 4×4 pixels (thực tế 128×128):

Input (có nhiễu):
┌──────────────────┐
│ -120  -95  -130  │
│ -115 -100  -105  │  ← Số âm/dương
│  -90  -85  -125  │     (HU - Hounsfield Units)
│ -110 -100   -95  │
└──────────────────┘
     ↓
  [RED-CNN]
     ↓
Output (sạch):
┌──────────────────┐
│ -118  -100  -118 │
│ -100  -100  -100 │  ← Mịn hơn, ít biến động
│  -95  -100  -118 │
│ -100  -100  -100 │
└──────────────────┘
```

### Ý nghĩa của số

```
Hounsfield Units (HU):

-1000  →  Không khí (đen)
 -500  →  Phổi
    0  →  Nước
  +40  →  Mô mềm
 +400  →  Xương (trắng)
```

---

## 💡 Tóm tắt

### Input → RED-CNN → Output

```
┌─────────────────┐
│  INPUT          │  • Ảnh CT liều thấp
│  🖼️ ▓▒░         │  • Nhiều nhiễu
│                 │  • 128×128 pixels
└────────┬────────┘
         ↓
    [RED-CNN]  ← AI Model
         ↓
┌────────┴────────┐
│  OUTPUT         │  • Ảnh CT sạch
│  🖼️ ✨          │  • Ít nhiễu
│                 │  • 128×128 pixels
└─────────────────┘
```

**Nhiệm vụ duy nhất**: LOẠI BỎ NHIỄU, giữ nguyên nội dung

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. RED-CNN nhận input là gì?
2. Output của RED-CNN có gì khác so với input?
3. Kích thước ảnh có thay đổi không?
4. RED-CNN có thêm/bớt nội dung không?

---

➡️ **Tiếp theo**: `CAU_HOI_05.md` - RED-CNN hoạt động như thế nào?
