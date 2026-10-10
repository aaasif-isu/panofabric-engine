"""CPU test fixture for the same fixed-budget worker/coordinator protocol."""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F

from panoengine.decentralized.decoupled_heloco.config import load_config
from panoengine.decentralized.decoupled_heloco.gpu_worker import run_session
from panoengine.decentralized.decoupled_heloco.smoke import _TinyTokenModel
from panoengine.decentralized.decoupled_heloco.transport import FramedTransport


class CPUTrainer:
    def __init__(self, fragments, *, fail_after=None):
        self.model_parts = [_TinyTokenModel(fragments)]
        self.optimizer = torch.optim.SGD(self.model_parts[0].parameters(), lr=0.1, momentum=0.9)
        self.metrics_processor = SimpleNamespace(log=lambda *args, **kwargs: None)
        self.step = 0
        self.tokens = torch.arange(8).repeat(2, 1)
        self.dataloader = None
        self.fail_after = fail_after

    def batch_generator(self, _):
        while True:
            yield {"input": self.tokens}, (self.tokens + 1) % 8

    def train_step(self, iterator):
        if self.fail_after is not None and self.step >= self.fail_after:
            os._exit(17)  # Inject a hard process failure, without a fatal frame.
        inputs, labels = next(iterator)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.model_parts[0](inputs["input"]).reshape(-1, 8), labels.flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model_parts[0].parameters(), 1.0)
        self.optimizer.step()
        self.metrics_processor.log(self.step, float(loss.detach()), float(loss.detach()), float(norm))
        time.sleep(0.03)


class DelayedSnapshots:
    """Test-only delayed delivery; metadata and younger snapshots keep flowing."""
    def __init__(self, transport, fragment, delay):
        self.transport, self.fragment, self.delay = transport, fragment, delay
        self.pending = []

    def send(self, kind, body):
        if kind == "snapshot" and body["snapshot"]["fragment_id"] == self.fragment:
            self.pending.append((time.monotonic() + self.delay, kind, body))
        else:
            self.transport.send(kind, body)

    def send_reliable(self, kind, body, **kwargs):
        if kind == "snapshot" and body["snapshot"]["fragment_id"] == self.fragment:
            self.send(kind, body)
        else:
            self.transport.send_reliable(kind, body, **kwargs)

    def send_progress(self, body):
        self.transport.send_progress(body)

    def diagnostics(self):
        return self.transport.diagnostics()

    def _deliver(self):
        now = time.monotonic()
        ready = [item for item in self.pending if item[0] <= now]
        self.pending = [item for item in self.pending if item[0] > now]
        for _, kind, body in ready:
            self.transport.send(kind, body)

    def receive(self):
        self._deliver()
        return self.transport.receive()

    def flush(self, timeout=2.):
        deadline = time.monotonic() + timeout
        while self.pending:
            self._deliver()
            if time.monotonic() >= deadline:
                raise TimeoutError("test delayed snapshots did not drain")
            time.sleep(.002)
        self.transport.flush(timeout=max(.001, deadline - time.monotonic()))

    def close(self):
        self.transport.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--learner-id", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--fail-learner", type=int)
    parser.add_argument("--fail-after", type=int, default=4)
    parser.add_argument("--capture-delay-fragment", type=int, default=0)
    parser.add_argument("--capture-delay-seconds", type=float, default=0.)
    parser.add_argument("--exit-delay-seconds", type=float, default=0.)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.run_dir / "experiment.yaml")
    options = SimpleNamespace(**json.loads((args.run_dir / "run-options.json").read_text()))
    transport = FramedTransport(socket.create_connection(("127.0.0.1", args.port)))
    if args.capture_delay_seconds:
        transport = DelayedSnapshots(transport, args.capture_delay_fragment, args.capture_delay_seconds)
    try:
        transport.send("hello", {"learner_id": args.learner_id, "token": os.environ["DECOUPLED_RUN_TOKEN"]})
        fail_after = args.fail_after if args.learner_id == args.fail_learner else None
        run_session(transport, CPUTrainer(config.decoupled.num_fragments, fail_after=fail_after), config, options, args.learner_id, args.run_dir / f"learner_{args.learner_id}", time.monotonic() + args.timeout)
    finally:
        transport.close()
    # Exercise real teardown after the stopped ACK/socket closure.
    time.sleep(args.exit_delay_seconds)


if __name__ == "__main__":
    main()
