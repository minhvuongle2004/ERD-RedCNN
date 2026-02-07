# 📚 Tài liệu hướng dẫn RED-CNN

Thư mục này chứa đầy đủ tài liệu hướng dẫn về mô hình RED-CNN cho người mới bắt đầu.

---

## 🎯 Tài liệu chính

### 1. [RED-CNN_TUTORIAL_VI.md](./RED-CNN_TUTORIAL_VI.md) 
**📖 Tutorial chi tiết (2+ giờ đọc)**

Tài liệu toàn diện nhất, bao gồm:
- ✅ Giới thiệu vấn đề LDCT denoising
- ✅ Kiến trúc RED-CNN chi tiết
- ✅ Cách mô hình xử lý ảnh (forward pass)
- ✅ Quá trình training đầy đủ
- ✅ Hướng dẫn thực hành
- ✅ Q&A và bài tập

**Dành cho**: Sinh viên muốn hiểu sâu về RED-CNN từ A-Z

---

### 2. [RED-CNN_QUICK_REFERENCE.md](./RED-CNN_QUICK_REFERENCE.md)
**⚡ Tham khảo nhanh (15 phút đọc)**

Cheat sheet ngắn gọn gồm:
- Layer dimensions table
- Forward pass formula
- Training config
- Code snippets
- Common issues & tips

**Dành cho**: Tham khảo nhanh khi code, debug, hoặc ôn lại kiến thức

---

### 3. [TEACHING_OUTLINE.md](./TEACHING_OUTLINE.md)
**👨‍🏫 Outline hướng dẫn (2-3 giờ)**

Kế hoạch bài giảng chi tiết:
- Timeline từng phần
- Mục tiêu học tập
- Hoạt động thực hành
- Câu hỏi kiểm tra hiểu
- Bài tập về nhà
- Tips cho giảng viên

**Dành cho**: Người hướng dẫn/giảng viên

---

## 🛠️ Công cụ hỗ trợ

### [../scripts/visualize_redcnn.py](../scripts/visualize_redcnn.py)
**🎨 Visualization tool**

Script Python để visualize:
- Kiến trúc RED-CNN (ASCII art)
- Forward pass step-by-step
- Skip connections diagram
- Model parameters count
- Giải thích residual learning

**Cách chạy**:
```bash
python scripts/visualize_redcnn.py
```

---

## 📋 Lộ trình học tập đề xuất

### Cho sinh viên mới (Newbie):

#### Tuần 1: Hiểu vấn đề và kiến trúc
1. Đọc phần 1-2 của Tutorial (Giới thiệu + Kiến trúc)
2. Chạy visualization script
3. Vẽ sơ đồ RED-CNN trên giấy
4. Tính toán dimensions của từng layer

#### Tuần 2: Đọc hiểu code
1. Đọc phần 3-4 của Tutorial (Forward pass + Training)
2. Đọc file `network.py` và comment từng dòng
3. Đọc file `Trainer.py` và hiểu training loop
4. Trả lời các câu hỏi trong Q&A section

#### Tuần 3: Thực hành
1. Setup environment
2. Chạy training với config mặc định
3. Modify hyperparameters và quan sát
4. Làm các bài tập về nhà

#### Tuần 4: Advanced
1. Implement modifications (thay đổi architecture)
2. Visualize feature maps
3. Experiment với skip connections
4. Viết report tổng kết

---

### Cho sinh viên có kiến thức về DL:

#### Session 1 (2 giờ):
1. Đọc Quick Reference (15 phút)
2. Chạy visualization script (15 phút)
3. Đọc code: network.py, Trainer.py (30 phút)
4. Run training và modify (60 phút)

#### Session 2 (2 giờ):
1. Deep dive vào skip connections (30 phút)
2. Experiment với architecture (60 phút)
3. Visualization và analysis (30 phút)

---

## 📊 Checklist đánh giá

Sau khi học xong, sinh viên cần có khả năng:

### Kiến thức nền tảng:
- [ ] Giải thích được vấn đề LDCT denoising
- [ ] Vẽ được sơ đồ kiến trúc RED-CNN
- [ ] Giải thích được vai trò của skip connections
- [ ] So sánh RED-CNN với vanilla CNN

### Kỹ năng thực hành:
- [ ] Tính toán được dimensions sau mỗi layer
- [ ] Đọc và hiểu code trong network.py
- [ ] Chạy được training script
- [ ] Modify được hyperparameters
- [ ] Debug được common errors

### Ứng dụng:
- [ ] Load checkpoint và inference
- [ ] Visualize feature maps
- [ ] Interpret training metrics
- [ ] Modify architecture cho experiments

---

## 🎓 Tài liệu bổ sung

### Papers:
1. **RED-CNN (2017)**: H. Chen et al., "Low-dose CT with a residual encoder-decoder convolutional neural network"
2. **ResNet (2016)**: K. He et al., "Deep Residual Learning for Image Recognition"

### Online resources:
- [Project Documentation](https://eeulig.github.io/ldct-benchmark/)
- [GitHub Repository](https://github.com/eeulig/ldct-benchmark)
- [PyTorch Conv2d Docs](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [PyTorch ConvTranspose2d Docs](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)

### Videos (tự tìm):
- CNN fundamentals
- Encoder-Decoder architectures
- Residual connections / Skip connections
- Medical image processing

---

## 💬 Q&A nhanh

### Q: Tôi nên bắt đầu từ đâu?
**A**: Đọc `RED-CNN_TUTORIAL_VI.md` từ đầu đến cuối, sau đó chạy visualization script.

### Q: Tôi không hiểu về CNN, có vấn đề không?
**A**: Nên đọc thêm về CNN basics trước. Tutorial giả định bạn đã biết Conv2D, ReLU, etc.

### Q: Làm sao để test kiến thức của mình?
**A**: Làm các bài tập trong phần 8 của Tutorial và checklist ở trên.

### Q: Code báo lỗi, phải làm sao?
**A**: Check phần "Common issues" trong Quick Reference, hoặc search error message.

### Q: Tôi muốn contribute vào project?
**A**: Đọc [CONTRIBUTING.md](../CONTRIBUTING.md) ở root directory.

---

## 📞 Liên hệ

Nếu có thắc mắc về tài liệu hoặc cần hỗ trợ thêm:
- Tạo issue trên GitHub repository
- Email project maintainers (xem README.md chính)

---

## 📝 Notes cho người duy trì tài liệu

### Khi update:
- [ ] Sync code examples với implementation thực tế
- [ ] Update line numbers trong Teaching Outline
- [ ] Test visualization script với Python version mới
- [ ] Kiểm tra links không bị broken

### Phản hồi từ học viên:
- Ghi nhận phần nào khó hiểu nhất
- Thu thập câu hỏi thường gặp để bổ sung
- Cập nhật bài tập dựa trên feedback

---

**Chúc học tốt! 🚀**

*Last updated: 2025-12-31*
