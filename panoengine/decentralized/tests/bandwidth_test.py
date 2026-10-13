import json
import time
import csv
from pathlib import Path
import tempfile
import threading
import torch

import socket
from urllib.request import Request, urlopen

import torch
import torch.nn as nn
from torch.optim import SGD

from panoengine.decentralized.async_diloco import AsyncDiLoCoServer
from panoengine.decentralized.decoupled_heloco.transport import FramedTransport
from panoengine.decentralized.bandwidth import BandwidthMonitor, get_monitor, set_monitor
from panoengine.decentralized.plot_bandwidth import generate_individual_plots, generate_comparison_plots

def test_byte_accounting_and_peaks():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        monitor = BandwidthMonitor(folder, method="test")
        set_monitor(monitor)

        # Upload 10 MB in window 0
        monitor.origin = time.monotonic() - 0.05
        monitor.record("upload", 10_000_000)
        
        # Download 5 MB in window 0
        monitor.record("download", 5_000_000)
        
        # Wait until window 1 (time > 1.0)
        monitor.origin = time.monotonic() - 1.1
        monitor.record("upload", 20_000_000)
        
        # Incomplete window at 1.5
        monitor.origin = time.monotonic() - 1.5
        monitor.record("download", 1_000_000)
        
        monitor.close()
        
        with open(folder / "bandwidth_summary.json") as f:
            summary = json.load(f)
            
        assert summary["total_upload_MB"] == 30.0
        assert summary["total_download_MB"] == 6.0
        
        # 10 MB in 0.1s -> 100 MB/s
        assert summary["peak_upload_MBps_100ms"] >= 100.0
        # 20 MB in 1s -> 20 MB/s
        assert summary["peak_upload_MBps_1s"] >= 20.0
        
        assert (folder / "bandwidth_timeseries.csv").exists()

def test_concurrent_accounting():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        monitor = BandwidthMonitor(folder, method="test")
        set_monitor(monitor)
        
        def worker():
            for _ in range(10):
                monitor.record("upload", 1000, timestamp=0.05)
                monitor.record("download", 500, timestamp=0.05)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        
        assert monitor.total_upload_bytes == 50000
        assert monitor.total_download_bytes == 25000

def test_figure_generation():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        method1 = base / "method-m1"
        method1.mkdir()
        
        monitor = BandwidthMonitor(method1, method="m1")
        monitor.record("upload", 50_000_000, timestamp=0.1)
        monitor.record("download", 20_000_000, timestamp=0.1)
        monitor.close()
        
        generate_individual_plots(method1)
        assert (method1 / "figures" / "bandwidth_over_time.png").exists()
        
        method2 = base / "method-m2"
        method2.mkdir()
        monitor = BandwidthMonitor(method2, method="m2")
        monitor.record("upload", 60_000_000, timestamp=0.1)
def test_parameter_server_real_traffic():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        monitor = BandwidthMonitor(folder, method="test_ps")
        set_monitor(monitor)

        model = nn.Linear(100, 100) # 10000 params (~40KB)
        optimizer = SGD(model.parameters(), lr=0.1)
        server = AsyncDiLoCoServer(model, optimizer, port=0)
        
        try:
            url = server.address()
            # Construct a raw push (full_sync = True)
            numel = sum(p.numel() for p in model.parameters())
            header = {"flag": 1, "dtype": "float32", "numel": numel, "baseline_revision": 0, "speed": 10.0}
            header_bytes = (json.dumps(header) + "\n").encode()
            payload_bytes = bytes(numel * 4) # fake zeros for pseudo-gradients
            body = header_bytes + payload_bytes
            
            req = Request(url, data=body, headers={"Content-Type": "application/octet-stream"})
            with urlopen(req) as response:
                resp_body = response.read()
                
            # Both upload (worker pushing pseudograds) and download (PS returning new params)
            # should have been intercepted.
            assert monitor.total_upload_bytes > 40000, f"Got {monitor.total_upload_bytes}"
            assert monitor.total_download_bytes > 40000, f"Got {monitor.total_download_bytes}"
            
        finally:
            server.shutdown()
            monitor.close()

def test_decoupled_syncer_real_traffic():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        monitor = BandwidthMonitor(folder, method="test_decoupled")
        set_monitor(monitor)

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        
        # Client connects to "syncer"
        client_sock = socket.socket()
        client_sock.connect(("127.0.0.1", port))
        
        # Server accepts
        server_sock, _ = listener.accept()
        
        server_transport = FramedTransport(server_sock)
        client_transport = FramedTransport(client_sock)
        
        try:
            # Syncer sends update (download)
            fake_tensor = torch.zeros(10000, dtype=torch.float32)
            # This triggers encode_message -> _writer -> send -> download count
            server_transport.send("update", {"fragment_id": 0, "revision": 1, "parameters": {"weight": fake_tensor}})
            
            # Client must read it to unblock stream
            msg = client_transport.receive()
            
            # Client sends ack (upload)
            client_transport.send("update_ack", {"metadata": {"learner_id": 0, "fragments": []}})
            msg2 = server_transport.receive()
            
            # Let threads process
            time.sleep(0.1)
            
            # The Syncer (orchestrator) monitor should see:
            assert monitor.total_download_bytes > 40000, f"Got {monitor.total_download_bytes}"
            assert monitor.total_upload_bytes > 0, f"Got {monitor.total_upload_bytes}"
        finally:
            server_transport.close()
            client_transport.close()
            listener.close()
            monitor.close()

if __name__ == "__main__":
    test_byte_accounting_and_peaks()
    test_concurrent_accounting()
    test_figure_generation()
    test_parameter_server_real_traffic()
    test_decoupled_syncer_real_traffic()
    print("All tests passed!")