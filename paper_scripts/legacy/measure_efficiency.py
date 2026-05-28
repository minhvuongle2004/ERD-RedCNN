import os
import time
import torch
import pandas as pd
from ldctbench.hub import load_model, Methods

def measure_efficiency():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    methods = ["cnn10", "redcnn", "wganvgg", "resnet", "qae", "dugan", "transct"]
    results = []
    
    # Dummy input representing a 1-channel CT slice (512x512)
    dummy_input = torch.randn(1, 1, 512, 512).to(device)
    
    for method_name in methods:
        print(f"Benchmarking {method_name}...")
        
        # Load model
        model = load_model(method_name, eval=True).to(device)
        
        # Count Parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_million = total_params / 1e6
        
        with torch.no_grad():
            # Warm-up (10 iterations)
            for _ in range(10):
                _ = model(dummy_input)
            
            # Measure Runtime (average over 50 iterations)
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(50):
                _ = model(dummy_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.time()
            runtime_ms = ((end_time - start_time) / 50.0) * 1000.0
            
            # Measure Peak VRAM (if CUDA)
            vram_mb = 0
            if device.type == "cuda":
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                torch.cuda.reset_peak_memory_stats(device)
        
        # Clear model from memory to measure clean VRAM for next model
        del model
        torch.cuda.empty_cache()
        
        results.append({
            "Method": method_name,
            "Parameters (M)": round(params_million, 3),
            "Runtime (ms/image)": round(runtime_ms, 2),
            "Peak VRAM (MB)": round(vram_mb, 2)
        })
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("results/efficiency_metrics.csv", index=False)
    print("\n--- EFFICIENCY BENCHMARK RESULTS ---")
    print(df.to_markdown(index=False))
    print("------------------------------------")
    print("Saved to results/efficiency_metrics.csv")

if __name__ == "__main__":
    measure_efficiency()
