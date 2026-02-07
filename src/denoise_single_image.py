
import argparse
import torch
import numpy as np
import skimage.io as io
from ldctbench.hub import load_model
import os

# Normalization constants from ldctbench/data/info.yml
MEAN = 481.45419786099086
STD = 502.18507379395044

def denoise_image(input_path: str, output_path: str, model_name: str):
    """
    Denoises a single CT image using a pre-trained model from the ldct-benchmark.

    Args:
        input_path: Path to the input TIFF image.
        output_path: Path to save the denoised PNG image.
        model_name: Name of the pre-trained model to use (e.g., 'redcnn', 'cnn10').
    """
    print(f"Đang tải mô hình '{model_name}'...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, eval=True).to(device)

    print(f"Đang đọc ảnh đầu vào từ: {input_path}")
    # Read the TIFF image
    input_image = io.imread(input_path).astype(np.float32)

    # --- Pre-processing ---
    # 1. Normalize the image using the project's mean and std
    normalized_image = (input_image - MEAN) / STD

    # 2. Convert to a PyTorch tensor
    input_tensor = torch.from_numpy(normalized_image)

    # 3. Add batch and channel dimensions (C, H, W) -> (N, C, H, W)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    # 4. Move tensor to the selected device
    input_tensor = input_tensor.to(device)

    print("Đang thực hiện khử nhiễu...")
    with torch.no_grad():
        # --- Inference ---
        output_tensor = model(input_tensor)

    # --- Post-processing ---
    # 1. Remove batch and channel dimensions and move to CPU
    output_array = output_tensor.squeeze().cpu().numpy()

    # 2. De-normalize the output
    denoised_image = (output_array * STD) + MEAN
    
    # 3. Clip the output to a plausible range, e.g., the original image's range
    clipped_image = np.clip(denoised_image, np.min(input_image), np.max(input_image))

    # 4. Scale to 0-255 and convert to uint8 for PNG format
    min_val = np.min(clipped_image)
    max_val = np.max(clipped_image)
    # Avoid division by zero if the image is flat
    if max_val > min_val:
        scaled_image = (clipped_image - min_val) / (max_val - min_val)
    else:
        scaled_image = np.zeros_like(clipped_image)
        
    output_image = (scaled_image * 255).astype(np.uint8)


    print(f"Đang lưu ảnh đã khử nhiễu vào: {output_path}")
    io.imsave(output_path, output_image, check_contrast=False)
    print("Hoàn thành!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise a single CT image using a pre-trained ldct-benchmark model.")
    parser.add_argument(
        "--input",
        type=str,
        default="tests/ldct_iqa/test011.tiff",
        help="Path to the input TIFF image.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="redcnn",
        help="Name of the pre-trained model to use (e.g., 'redcnn', 'cnn10', 'wganvgg').",
    )
    args = parser.parse_args()

    # Create result directory
    output_dir = "result"
    os.makedirs(output_dir, exist_ok=True)

    # Generate output path
    input_filename = os.path.basename(args.input)
    base_name = os.path.splitext(input_filename)[0]
    output_filename = f"{base_name}_denoised.png"
    output_path = os.path.join(output_dir, output_filename)

    denoise_image(args.input, output_path, args.model)
