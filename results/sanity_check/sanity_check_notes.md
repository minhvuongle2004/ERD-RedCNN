# Sanity Check Notes — EDR-REDNet
Date: 2026-05-06

## Config Used
- Config: configs/edrrednet.yaml
- mbs: 8, max_iterations: 2000, data_subset: 0.1
- num_edge_blocks: 2, loss_alpha: 0.1
- GPU: CUDA (local machine)

## Results
- Iterations: 2000 (4 epochs × 500 iter/epoch)
- Train: 500/500 per epoch, ~2:40/epoch (~3.1 it/s)
- Validate: 11/11 batches, ~10s/epoch

## SSIM Checkpoints
| Iteration | SSIM    |
|-----------|---------|
| 500       | 0.55888 |
| 1500      | 0.55959 |

## Checklist
- [x] Loss giảm (SSIM tăng dần)
- [x] Không có NaN trong loss
- [x] Shape output = shape input (validate passed)
- [x] Checkpoint lưu được (best_SSIM.pt)
- [x] CUDA device mismatch đã được fix (criterion.to(device))
- [x] val_step override hoạt động đúng

## Issues Fixed During Sanity Check
1. use_config: thêm encoding='utf-8' + merge strategy (CLI args preserved)
2. Trainer.train_step: batch format là dict {"x", "y"} không phải tuple
3. CombinedLoss: .to(device) để Sobel kernels lên GPU
4. val_step: override để unpack tuple (total_loss, components)

## Conclusion
✅ PASSED — Sẵn sàng cho Pilot train (mbs=16, 20000 iter, 50% data)
