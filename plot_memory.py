#!/usr/bin/env python3
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
    exit(1)

def main():
    # Model size used in the benchmark (13,639,936 parameters)
    # Each FP32 parameter takes 4 bytes.
    param_count = 13639936
    bytes_per_param = 4
    
    # 1. Global Model Weights (FP32)
    # 2. Outer Optimizer Momentum (FP32)
    # 3. Aggregation Buffer (FP32)
    ps_multiplier = 3  
    
    base_model_size_mb = (param_count * bytes_per_param) / (1024 ** 2)
    ps_memory_mb = base_model_size_mb * ps_multiplier

    # In original methods (mla, diloco, heloco), a centralized parameter server is required.
    # In decoupled methods (decoupled_diloco, decoupled_heloco), the parameter server is eliminated.
    methods = ["mla", "diloco", "heloco", "decoupled_diloco", "decoupled_heloco"]
    
    ps_memory = {
        "mla": ps_memory_mb,
        "diloco": ps_memory_mb,
        "heloco": ps_memory_mb,
        "decoupled_diloco": 0.0,
        "decoupled_heloco": 0.0
    }

    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(ps_memory.keys(), ps_memory.values(), color=['salmon', 'salmon', 'salmon', 'lightgreen', 'lightgreen'])
    
    plt.ylabel("Parameter Server Memory Overhead (MB)", fontsize=12)
    plt.title(f"Parameter Server Bottleneck Elimination (15M Parameter Model)", fontsize=14, fontweight="bold")
    plt.grid(axis='y', alpha=0.3)
    
    # Add data labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f} MB", ha='center', va='bottom', fontweight='bold')

    # Add descriptive text
    plt.text(0.5, 0.8, 
             "Original methods require an active central Parameter Server node.\n"
             "Decoupled methods eliminate this bottleneck entirely by\n"
             "distributing the outer optimizer state across fragmented peers.", 
             transform=plt.gca().transAxes, 
             fontsize=11, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    out_file = "run_script/parameter_server_memory_plot.png"
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"\nParameter Server memory comparison plot saved to {out_file}")

if __name__ == "__main__":
    main()