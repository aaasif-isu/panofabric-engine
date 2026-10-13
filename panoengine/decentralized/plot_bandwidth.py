import argparse
import json
import csv
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

def generate_individual_plots(folder: Path):
    timeseries = folder / "bandwidth_timeseries.csv"
    if not timeseries.exists():
        return
    
    elapsed, up_mbps, down_mbps = [], [], []
    with open(timeseries) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["window_width_s"]) == 1.0:
                elapsed.append(float(row["elapsed_s"]))
                up_mbps.append(float(row["upload_MBps"]))
                down_mbps.append(float(row["download_MBps"]))
    
    if not elapsed: return
    
    fig_dir = folder / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(elapsed, up_mbps, label="Upload (MB/s)", alpha=0.8)
    plt.plot(elapsed, down_mbps, label="Download (MB/s)", alpha=0.8)
    plt.xlabel("Elapsed Training Time (s)")
    plt.ylabel("Bandwidth (MB/s)")
    plt.title(f"Bandwidth Over Time (1s windows)\n{folder.name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "bandwidth_over_time.png", dpi=300)
    plt.savefig(fig_dir / "bandwidth_over_time.pdf")
    plt.close()

def generate_comparison_plots(folder: Path):
    # folder is the main experiment comparison folder
    # Contains method directories: e.g. method-heloco, method-diloco, etc.
    methods = []
    peak_up_1s, peak_down_1s = [], []
    peak_up_100ms, peak_down_100ms = [], []
    total_up, total_down = [], []
    
    for summary_path in sorted(folder.rglob("bandwidth_summary.json")):
        method_dir = summary_path.parent
        # Avoid double-counting if there's any nesting
        if method_dir == folder:
            continue
        try:
            with open(summary_path) as f:
                data = json.load(f)
            methods.append(data.get("method", method_dir.name))
            peak_up_1s.append(data.get("peak_upload_MBps_1s", 0))
            peak_down_1s.append(data.get("peak_download_MBps_1s", 0))
            peak_up_100ms.append(data.get("peak_upload_MBps_100ms", 0))
            peak_down_100ms.append(data.get("peak_download_MBps_100ms", 0))
            total_up.append(data.get("total_upload_MB", 0))
            total_down.append(data.get("total_download_MB", 0))
        except Exception:
            pass

    if not methods:
        return
    
    fig_dir = folder / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    import numpy as np
    x = np.arange(len(methods))
    width = 0.2
    
    # Figure B: Peak Bandwidth
    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5*width, peak_up_100ms, width, label='Peak Upload (100ms)', color='tab:blue')
    plt.bar(x - 0.5*width, peak_up_1s, width, label='Peak Upload (1s)', color='tab:cyan')
    plt.bar(x + 0.5*width, peak_down_100ms, width, label='Peak Download (100ms)', color='tab:orange')
    plt.bar(x + 1.5*width, peak_down_1s, width, label='Peak Download (1s)', color='tab:red')
    plt.ylabel("Peak Bandwidth (MB/s)")
    plt.title("Peak Bandwidth Comparison (1s and 100ms windows)")
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / "peak_bandwidth.png", dpi=300)
    plt.savefig(fig_dir / "peak_bandwidth.pdf")
    plt.close()
    
    # Figure C: Total Communication
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, total_up, width*2, label='Total Upload (MB)', color='tab:blue')
    plt.bar(x + width/2, total_down, width*2, label='Total Download (MB)', color='tab:orange')
    plt.ylabel("Total Data (MB)")
    plt.title("Total Communication Volume Comparison")
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / "total_communication.png", dpi=300)
    plt.savefig(fig_dir / "total_communication.pdf")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate bandwidth figures")
    parser.add_argument("--dir", type=Path, required=True, help="Directory to process")
    parser.add_argument("--comparison", action="store_true", help="Generate cross-method plots from subdirectories")
    args = parser.parse_args()
    
    if args.comparison:
        generate_comparison_plots(args.dir)
    else:
        generate_individual_plots(args.dir)
