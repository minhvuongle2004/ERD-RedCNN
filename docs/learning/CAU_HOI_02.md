# Câu 2: Vấn đề gì xảy ra khi giảm liều bức xạ?

## 🎯 Mục tiêu
Hiểu được:
- CT có bức xạ và tại sao đó là vấn đề
- Trade-off giữa an toàn và chất lượng ảnh

---

## 📖 Trả lời

### Vấn đề 1: CT có bức xạ

**CT dùng tia X-ray** → Có bức xạ ion hóa

```
Máy CT                Người bệnh
  📷 ⚡                    🧑
   ↓                      ↓
Phát tia X-ray  →   Nhận bức xạ
```

**Bức xạ nhiều** = Tăng nguy cơ:
- Tổn thương tế bào
- Ung thư (nếu chụp nhiều lần)
- Đặc biệt nguy hiểm với trẻ em

### Vấn đề 2: Trade-off

```
           🎚️ Liều bức xạ
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
Liều CAO               Liều THẤP
    ↓                       ↓
✅ Ảnh RẤT RÕ           ✅ AN TOÀN hơn
❌ NGUY HIỂM           ❌ Ảnh BỊ NHIỄU
```

**Giống như chụp ảnh trong phòng tối**:
- Nhiều ánh sáng (flash mạnh) → Ảnh đẹp nhưng chói mắt ☀️😵
- Ít ánh sáng → An toàn nhưng ảnh mờ 🌙😕

### So sánh trực quan

**Ảnh CT liều cao (High-dose)**:
```
┌──────────────┐
│  🫁 🫁       │
│   Rõ nét     │  ← Chi tiết rõ ràng
│   ❤️         │  ← Biên giới sắc nét
│   🦴         │
└──────────────┘
```

**Ảnh CT liều thấp (Low-dose)**:
```
┌──────────────┐
│  🫁?🫁?      │
│   ▓▒░ Nhiễu  │  ← Có hạt nhiễu (noise)
│   ❤️??       │  ← Biên giới mờ
│   🦴?        │
└──────────────┘
```

### Ví dụ cụ thể

**Case 1: Phát hiện khối u nhỏ**
```
Ảnh liều cao:              Ảnh liều thấp:
┌───────┐                  ┌───────┐
│ 🫁   │                  │ 🫁 ▓▒│
│   ⚫ │ ← Thấy rõ khối u   │   ??│ ← Khối u bị nhiễu che
└───────┘                  └───────┘
   ✅ Chẩn đoán đúng          ❌ Có thể bỏ sót
```

### Thực tế trong bệnh viện

**Bác sĩ phải chọn**:

```
Tình huống 1: Nghi ung thư phổi
→ Dùng liều CAO (cần ảnh rõ để chẩn đoán chính xác)

Tình huống 2: Theo dõi định kỳ
→ Dùng liều THẤP (chụp nhiều lần, cần an toàn)

Tình huống 3: Trẻ em
→ Ưu tiên liều THẤP NHẤT có thể
```

---

## 🤔 Vậy giải pháp là gì?

### Ý tưởng: Dùng AI để cải thiện ảnh liều thấp!

```
Ảnh liều thấp (nhiễu)  →  [AI/RED-CNN]  →  Ảnh sạch
      INPUT                                  OUTPUT
```

**Lợi ích**:
✅ Bệnh nhân an toàn hơn (ít bức xạ)  
✅ Ảnh vẫn đủ rõ để chẩn đoán  
✅ Có thể chụp thường xuyên hơn  

**Đây chính là nhiệm vụ của RED-CNN!**

---

## 💡 Tóm tắt

**Trade-off cơ bản**:
```
Liều bức xạ cao  →  Ảnh rõ   nhưng  Nguy hiểm
Liều bức xạ thấp →  Ảnh nhiễu  nhưng  An toàn
```

**Mục tiêu RED-CNN**: 
> Khử nhiễu ảnh liều thấp để có cả **an toàn** lẫn **chất lượng**

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Tại sao bức xạ trong CT là vấn đề?
2. Điều gì xảy ra khi giảm liều bức xạ?
3. AI có thể giúp gì trong trường hợp này?

---

➡️ **Tiếp theo**: `CAU_HOI_03.md` - AI/Deep Learning giúp thế nào?
