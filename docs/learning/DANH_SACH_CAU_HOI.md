# 📝 Danh sách câu hỏi - Lộ trình học RED-CNN

> **Mục đích**: Dạy sinh viên hiểu **CÁCH HOẠT ĐỘNG** của RED-CNN  
> **Không cần**: Hiểu code, toán phức tạp

---

## 🎯 Cách sử dụng

Mỗi câu hỏi có:
- **Mục tiêu**: Sinh viên sẽ biết được gì sau khi học
- **File trả lời**: `CAU_HOI_XX.md` tương ứng

**Thứ tự**: Dạy theo thứ tự từ trên xuống dưới

---

## Phần 1: Hiểu vấn đề (3 câu hỏi)

### ❓ Câu 1: Ảnh CT là gì và tại sao cần nó?
**Mục tiêu**: Sinh viên hiểu CT scan dùng để làm gì

📄 **Đáp án**: `CAU_HOI_01.md`

---

### ❓ Câu 2: Vấn đề gì xảy ra khi giảm liều bức xạ?
**Mục tiêu**: Sinh viên hiểu trade-off giữa an toàn và chất lượng ảnh

📄 **Đáp án**: `CAU_HOI_02.md`

---

### ❓ Câu 3: AI/Deep Learning giúp được gì trong trường hợp này?
**Mục tiêu**: Sinh viên hiểu cơ bản về image denoising bằng AI

📄 **Đáp án**: `CAU_HOI_03.md`

---

## Phần 2: Hiểu RED-CNN làm gì (3 câu hỏi)

### ❓ Câu 4: RED-CNN nhận input gì và cho output gì?
**Mục tiêu**: Sinh viên biết input/output của model

📄 **Đáp án**: `CAU_HOI_04.md`

---

### ❓ Câu 5: RED-CNN hoạt động như thế nào? (Tổng quan)
**Mục tiêu**: Sinh viên hiểu flow: Input → Nén → Giải nén → Output

📄 **Đáp án**: `CAU_HOI_05.md`

---

### ❓ Câu 6: Tại sao gọi là "Encoder-Decoder"?
**Mục tiêu**: Sinh viên hiểu 2 phần chính của model

📄 **Đáp án**: `CAU_HOI_06.md`

---

## Phần 3: Hiểu các bước xử lý (4 câu hỏi)

### ❓ Câu 7: Encoder làm gì với ảnh?
**Mục tiêu**: Sinh viên hiểu quá trình "nén thông tin"

📄 **Đáp án**: `CAU_HOI_07.md`

---

### ❓ Câu 8: "Bottleneck" là gì?
**Mục tiêu**: Sinh viên hiểu điểm nén nhất của model

📄 **Đáp án**: `CAU_HOI_08.md`

---

### ❓ Câu 9: Decoder làm gì?
**Mục tiêu**: Sinh viên hiểu quá trình "khôi phục ảnh"

📄 **Đáp án**: `CAU_HOI_09.md`

---

### ❓ Câu 10: Skp iconnections là gì và tại sao cần nó?
**Mục tiêu**: Sinh viên hiểu "đường tắt" trong model

📄 **Đáp án**: `CAU_HOI_10.md`

---

## Phần 4: Hiểu training (3 câu hỏi)

### ❓ Câu 11: Model học như thế nào?
**Mục tiêu**: Sinh viên hiểu cơ bản về training process

📄 **Đáp án**: `CAU_HOI_11.md`

---

### ❓ Câu 12: Làm sao biết model đang học tốt hay không?
**Mục tiêu**: Sinh viên hiểu về loss và metrics

📄 **Đáp án**: `CAU_HOI_12.md`

---

### ❓ Câu 13: Training xong, dùng model như thế nào?
**Mục tiêu**: Sinh viên hiểu về inference/deployment

📄 **Đáp án**: `CAU_HOI_13.md`

---

## Phần 5: Hiểu tại sao (3 câu hỏi)

### ❓ Câu 14: Tại sao RED-CNN hiệu quả cho CT denoising?
**Mục tiêu**: Sinh viên hiểu điểm mạnh của RED-CNN

📄 **Đáp án**: `CAU_HOI_14.md`

---

### ❓ Câu 15: RED-CNN có hạn chế gì?
**Mục tiêu**: Sinh viên hiểu giới hạn của model

📄 **Đáp án**: `CAU_HOI_15.md`

---

### ❓ Câu 16: So với các phương pháp khác thì sao?
**Mục tiêu**: Sinh viên có cái nhìn tổng quan về các approaches

📄 **Đáp án**: `CAU_HOI_16.md`

---

## Phần 6: Cách train RED-CNN (7 câu hỏi)

### ❓ Câu 17: Cần chuẩn bị gì để train RED-CNN?
**Mục tiêu**: Sinh viên biết cần những gì trước khi bắt đầu train

📄 **Đáp án**: `CAU_HOI_17.md`

---

### ❓ Câu 18: Dataset là gì và cần như thế nào?
**Mục tiêu**: Sinh viên hiểu về dataset và yêu cầu dữ liệu

📄 **Đáp án**: `CAU_HOI_18.md`

---

### ❓ Câu 19: Cách chuẩn bị dữ liệu training?
**Mục tiêu**: Sinh viên biết các bước xử lý dữ liệu trước khi train

📄 **Đáp án**: `CAU_HOI_19.md`

---

### ❓ Câu 20: Các tham số training quan trọng là gì?
**Mục tiêu**: Sinh viên hiểu về learning rate, batch size, epochs...

📄 **Đáp án**: `CAU_HOI_20.md`

---

### ❓ Câu 21: Cách chạy training như thế nào?
**Mục tiêu**: Sinh viên biết quy trình chạy training từ đầu đến cuối

📄 **Đáp án**: `CAU_HOI_21.md`

---

### ❓ Câu 22: Làm sao biết training đang chạy tốt?
**Mục tiêu**: Sinh viên biết cách theo dõi và đánh giá quá trình training

📄 **Đáp án**: `CAU_HOI_22.md`

---

### ❓ Câu 23: Xử lý lỗi khi training như thế nào?
**Mục tiêu**: Sinh viên biết cách xử lý các lỗi thường gặp

📄 **Đáp án**: `CAU_HOI_23.md`

---

## 📊 Tổng kết

**Tổng cộng**: 23 câu hỏi  
**Thời gian ước tính**: 10-15 phút/câu = ~4-6 giờ

**Sau khi học xong**, sinh viên sẽ:
- ✅ Hiểu vấn đề LDCT denoising
- ✅ Hiểu cách RED-CNN hoạt động
- ✅ Hiểu encoder, decoder, skip connections
- ✅ Hiểu training và inference
- ✅ Hiểu ưu/nhược điểm của RED-CNN
- ✅ Biết cách train RED-CNN từ đầu đến cuối

**Chưa cần**:
- ❌ Viết code
- ❌ Tính toán dimensions
- ❌ Hiểu công thức toán học phức tạp

---

## 🎓 Tips cho người dạy

1. **Dạy tuần tự**: Đừng skip câu hỏi
2. **Check understanding**: Hỏi lại sinh viên sau mỗi câu
3. **Dùng hình ảnh**: Vẽ sơ đồ đơn giản trên giấy
4. **Ví dụ thực tế**: So sánh với việc chỉnh ảnh trên điện thoại
5. **Không vội**: 1 buổi nên học 4-5 câu thôi

---

**Bắt đầu**: Mở file `CAU_HOI_01.md` 🚀
