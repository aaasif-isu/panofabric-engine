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
    def __init__(self, fragments):
        self.model_parts = [_TinyTokenModel(fragments)]
        self.optimizer = torch.optim.SGD(self.model_parts[0].parameters(), lr=0.1, momentum=0.9)
        self.metrics_processor = SimpleNamespace(log=lambda *args, **kwargs: None)
        self.step = 0
        self.tokens = torch.arange(8).repeat(2, 1)
        self.dataloader = None

    def batch_generator(self, _):
        while True:
            yield {"input": self.tokens}, (self.tokens + 1) % 8

    def train_step(self, iterator):
        inputs, labels = next(iterator)
        self.optimizer.zero_grad()
        loss = F.cross_entropy(self.model_parts[0](inputs["input"]).reshape(-1, 8), labels.flatten())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model_parts[0].parameters(), 1.0)
        self.optimizer.step()
        self.metrics_processor.log(self.step, float(loss.detach()), float(loss.detach()), float(norm))
        time.sleep(0.03)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--learner-id", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.run_dir / "experiment.yaml")
    options = SimpleNamespace(**json.loads((args.run_dir / "run-options.json").read_text()))
    transport = FramedTransport(socket.create_connection(("127.0.0.1", args.port)))
    try:
        transport.send("hello", {"learner_id": args.learner_id, "token": os.environ["DECOUPLED_RUN_TOKEN"]})
        run_session(transport, CPUTrainer(config.decoupled.num_fragments), config, options, args.learner_id, args.run_dir / f"learner_{args.learner_id}", time.monotonic() + args.timeout)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
