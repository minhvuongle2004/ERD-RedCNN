"""
Script để visualize kiến trúc RED-CNN và forward pass
Giúp sinh viên hiểu rõ hơn về data flow và dimensions

Usage:
    python scripts/visualize_redcnn.py
"""

import torch
import torch.nn as nn
from argparse import Namespace


# Import model từ project
import sys
sys.path.append('.')
from ldctbench.methods.redcnn.network import Model


def print_layer_info(name, input_shape, output_shape, params=None):
    """In thông tin về một layer"""
    print(f"\n{'='*60}")
    print(f"Layer: {name}")
    print(f"Input shape:  {input_shape}")
    print(f"Output shape: {output_shape}")
    if params:
        print(f"Parameters: {params}")
    print(f"{'='*60}")


def trace_forward_pass():
    """Trace và visualize forward pass qua model"""
    print("\n" + "="*80)
    print(" RED-CNN FORWARD PASS VISUALIZATION ".center(80, "="))
    print("="*80)
    
    # Tạo model
    args = Namespace()
    model = Model(args, out_ch=96)
    model.eval()
    
    # Tạo dummy input
    batch_size = 1
    input_tensor = torch.randn(batch_size, 1, 128, 128)
    
    print(f"\n🔵 INPUT: {tuple(input_tensor.shape)}")
    print("   (batch_size, channels, height, width)")
    
    # Trace qua từng layer
    with torch.no_grad():
        # ENCODER
        print("\n" + "─"*80)
        print("📥 ENCODER PHASE (Feature extraction)")
        print("─"*80)
        
        # Conv1
        residual_1 = input_tensor
        out = model.relu(model.conv1(input_tensor))
        print_layer_info(
            "Conv1 + ReLU", 
            tuple(input_tensor.shape), 
            tuple(out.shape),
            "kernel=5x5, channels: 1→96"
        )
        print("💾 Saved residual_1 for skip connection")
        
        # Conv2
        prev_shape = out.shape
        out = model.relu(model.conv2(out))
        print_layer_info(
            "Conv2 + ReLU",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→96"
        )
        residual_2 = out
        print("💾 Saved residual_2 for skip connection")
        
        # Conv3
        prev_shape = out.shape
        out = model.relu(model.conv3(out))
        print_layer_info(
            "Conv3 + ReLU",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→96"
        )
        
        # Conv4
        prev_shape = out.shape
        out = model.relu(model.conv4(out))
        print_layer_info(
            "Conv4 + ReLU",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→96"
        )
        residual_3 = out
        print("💾 Saved residual_3 for skip connection")
        
        # Conv5 - Bottleneck
        prev_shape = out.shape
        out = model.relu(model.conv5(out))
        print_layer_info(
            "Conv5 + ReLU (BOTTLENECK)",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→96"
        )
        print("⭐ This is the most compressed representation!")
        
        # DECODER
        print("\n" + "─"*80)
        print("📤 DECODER PHASE (Image reconstruction)")
        print("─"*80)
        
        # TConv1
        prev_shape = out.shape
        out = model.tconv1(out)
        print_layer_info(
            "TransConv1",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→96"
        )
        out += residual_3
        print("➕ Added residual_3 (skip connection)")
        
        # TConv2
        prev_shape = out.shape
        out = model.tconv2(model.relu(out))
        print_layer_info(
            "ReLU + TransConv2",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→96"
        )
        
        # TConv3
        prev_shape = out.shape
        out = model.tconv3(model.relu(out))
        print_layer_info(
            "ReLU + TransConv3",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→96"
        )
        out += residual_2
        print("➕ Added residual_2 (skip connection)")
        
        # TConv4
        prev_shape = out.shape
        out = model.tconv4(model.relu(out))
        print_layer_info(
            "ReLU + TransConv4",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→96"
        )
        
        # TConv5
        prev_shape = out.shape
        out = model.tconv5(model.relu(out))
        print_layer_info(
            "ReLU + TransConv5",
            tuple(prev_shape),
            tuple(out.shape),
            "kernel=5x5, channels: 96→1"
        )
        out += residual_1
        print("➕ Added residual_1 (original input - final skip connection)")
    
    print(f"\n🔴 OUTPUT: {tuple(out.shape)}")
    print("   Denoised image with same dimension as input!")
    
    print("\n" + "="*80)
    print(" END OF FORWARD PASS ".center(80, "="))
    print("="*80)
    
    return out


def count_parameters():
    """Đếm số parameters trong model"""
    print("\n" + "="*80)
    print(" MODEL PARAMETERS ".center(80, "="))
    print("="*80)
    
    args = Namespace()
    model = Model(args, out_ch=96)
    
    total_params = 0
    trainable_params = 0
    
    print("\nLayer-wise parameters:")
    print("-"*80)
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        print(f"{name:20s} | Shape: {str(tuple(param.shape)):20s} | Params: {num_params:,}")
    
    print("-"*80)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")  # 4 bytes per float32
    print("="*80)


def visualize_skip_connections():
    """Visualize skip connections với ASCII art"""
    print("\n" + "="*80)
    print(" SKIP CONNECTIONS DIAGRAM ".center(80, "="))
    print("="*80)
    print("""
    INPUT (1×128×128)
        |
        |──────────────────────────────────────────────────┐ residual_1
        ↓                                                  |
    [Conv1 + ReLU] → (96×124×124)                         |
        ↓                                                  |
    [Conv2 + ReLU] → (96×120×120)                         |
        |──────────────────────────┐ residual_2           |
        ↓                          |                      |
    [Conv3 + ReLU] → (96×116×116)  |                      |
        ↓                          |                      |
    [Conv4 + ReLU] → (96×112×112)  |                      |
        |───────┐ residual_3       |                      |
        ↓       |                  |                      |
    [Conv5 + ReLU] → (96×108×108) BOTTLENECK             |
        ↓       |                  |                      |
    [TransConv1] ←─┘               |                      |
        ↓                          |                      |
    [ReLU + TransConv2] → (96×116×116)                    |
        ↓                          |                      |
    [ReLU + TransConv3] → (96×120×120)                    |
        ←─────────────────────────┘                       |
        ↓                                                 |
    [ReLU + TransConv4] → (96×124×124)                    |
        ↓                                                 |
    [ReLU + TransConv5] → (1×128×128)                     |
        ←────────────────────────────────────────────────┘
        ↓
    OUTPUT (1×128×128)
    
    Legend:
    ─────→  : Forward flow
    ←─────  : Skip connection (addition)
    """)
    print("="*80)


def compare_with_without_skip():
    """So sánh gradient flow với và không có skip connections"""
    print("\n" + "="*80)
    print(" WHY SKIP CONNECTIONS? ".center(80, "="))
    print("="*80)
    
    print("""
    WITHOUT skip connections:
    ════════════════════════════════════════════════════════════════════
    
    Network phải học TOÀN BỘ mapping: Input → Output
    
        Input (noisy)  →  [Black box]  →  Output (clean)
        
    ❌ Gradient vanishing khi network sâu
    ❌ Khó training
    ❌ Mất thông tin chi tiết
    
    
    WITH skip connections (Residual learning):
    ════════════════════════════════════════════════════════════════════
    
    Network chỉ cần học PHẦN KHÁC BIỆT (residual):
    
        Input + [Network learns noise pattern] → Output
        
    ✅ Gradient flow tốt hơn (shortcut path)
    ✅ Dễ training
    ✅ Giữ được thông tin chi tiết từ input
    
    Mathematical formulation:
    ────────────────────────────────────────────────────────────────────
    Traditional:     F(x) = y
    Residual:        F(x) = y - x,  therefore  y = F(x) + x
    
    → Network học phần residual F(x) = noise, easier to learn!
    """)
    print("="*80)


def main():
    """Main function"""
    print("\n" + "🎓"*40)
    print(" RED-CNN ARCHITECTURE VISUALIZATION ".center(80))
    print("Công cụ giúp hiểu rõ kiến trúc và cách hoạt động của RED-CNN")
    print("🎓"*40)
    
    # 1. Visualize skip connections
    visualize_skip_connections()
    input("\nPress Enter to continue...")
    
    # 2. Trace forward pass
    trace_forward_pass()
    input("\nPress Enter to continue...")
    
    # 3. Count parameters
    count_parameters()
    input("\nPress Enter to continue...")
    
    # 4. Explain skip connections
    compare_with_without_skip()
    
    print("\n" + "✅"*40)
    print("Visualization completed!".center(80))
    print("Để hiểu rõ hơn, hãy đọc file RED-CNN_TUTORIAL_VI.md")
    print("✅"*40 + "\n")


if __name__ == "__main__":
    main()
