import csv
import json
import threading
import time
from pathlib import Path

class BandwidthMonitor:
    def __init__(self, folder, method=""):
        self.folder = Path(folder)
        self.method = method
        self.lock = threading.Lock()
        
        self.total_upload_bytes = 0
        self.total_download_bytes = 0
        
        self.phase = "training"
        self.phase_totals = {"training": {"upload": 0, "download": 0}}
        
        self.window_100ms_up = {}
        self.window_100ms_down = {}
        self.window_1s_up = {}
        self.window_1s_down = {}
        
        self.origin = time.monotonic()
        self.enabled = True

    def set_phase(self, phase_name):
        with self.lock:
            self.phase = phase_name
            if phase_name not in self.phase_totals:
                self.phase_totals[phase_name] = {"upload": 0, "download": 0}

    def record(self, direction, num_bytes, timestamp=None):
        if not self.enabled or num_bytes == 0: return
        if timestamp is None:
            timestamp = time.monotonic() - self.origin
        
        with self.lock:
            if direction == "upload":
                self.total_upload_bytes += num_bytes
                self.phase_totals[self.phase]["upload"] += num_bytes
                w100 = int(timestamp * 10)
                w1 = int(timestamp)
                self.window_100ms_up[w100] = self.window_100ms_up.get(w100, 0) + num_bytes
                self.window_1s_up[w1] = self.window_1s_up.get(w1, 0) + num_bytes
            elif direction == "download":
                self.total_download_bytes += num_bytes
                self.phase_totals[self.phase]["download"] += num_bytes
                w100 = int(timestamp * 10)
                w1 = int(timestamp)
                self.window_100ms_down[w100] = self.window_100ms_down.get(w100, 0) + num_bytes
                self.window_1s_down[w1] = self.window_1s_down.get(w1, 0) + num_bytes

    def close(self):
        if not self.enabled: return
        with self.lock:
            timeseries_path = self.folder / "bandwidth_timeseries.csv"
            with timeseries_path.open("w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["window_width_s", "elapsed_s", "upload_bytes", "download_bytes", "upload_MBps", "download_MBps"])
                
                max_1s = max(list(self.window_1s_up.keys()) + list(self.window_1s_down.keys()) + [-1])
                peak_1s_up, peak_1s_down = 0.0, 0.0
                for w in range(max_1s + 1):
                    ub = self.window_1s_up.get(w, 0)
                    db = self.window_1s_down.get(w, 0)
                    umb = ub / 1e6
                    dmb = db / 1e6
                    peak_1s_up = max(peak_1s_up, umb)
                    peak_1s_down = max(peak_1s_down, dmb)
                    writer.writerow([1.0, w, ub, db, umb, dmb])
                
                max_100ms = max(list(self.window_100ms_up.keys()) + list(self.window_100ms_down.keys()) + [-1])
                peak_100ms_up, peak_100ms_down = 0.0, 0.0
                for w in range(max_100ms + 1):
                    ub = self.window_100ms_up.get(w, 0)
                    db = self.window_100ms_down.get(w, 0)
                    umb = (ub * 10) / 1e6
                    dmb = (db * 10) / 1e6
                    peak_100ms_up = max(peak_100ms_up, umb)
                    peak_100ms_down = max(peak_100ms_down, dmb)
                    writer.writerow([0.1, w / 10.0, ub, db, umb, dmb])

            summary = {
                "method": self.method,
                "peak_upload_MBps_100ms": peak_100ms_up,
                "peak_download_MBps_100ms": peak_100ms_down,
                "peak_upload_MBps_1s": peak_1s_up,
                "peak_download_MBps_1s": peak_1s_down,
                "total_upload_MB": self.total_upload_bytes / 1e6,
                "total_download_MB": self.total_download_bytes / 1e6,
                "phases_MB": {p: {"upload": v["upload"] / 1e6, "download": v["download"] / 1e6} for p, v in self.phase_totals.items()},
                "measurement_scope": "socket payload and framing bytes measured at the central Parameter Server or Decoupled Syncer; excludes TCP overhead"
            }
            with (self.folder / "bandwidth_summary.json").open("w") as f:
                json.dump(summary, f, indent=2)

class _DummyBandwidthMonitor:
    def record(self, *args, **kwargs): pass
    def close(self): pass

_global_monitor = None
_monitor_lock = threading.Lock()

def get_monitor():
    with _monitor_lock:
        return _global_monitor or _DummyBandwidthMonitor()

def set_monitor(monitor):
    global _global_monitor
    with _monitor_lock:
        _global_monitor = monitor
