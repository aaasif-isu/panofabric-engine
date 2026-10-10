"""Real CPU optimizer/HTTP fixture; never substitutes a CUDA Trainer."""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from cpu_training_worker import CPUTrainer
from panoengine.decentralized.decoupled_heloco.baseline_worker import run_baseline_session
from panoengine.decentralized.decoupled_heloco.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--learner-id", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.run_dir / "experiment.yaml")
    options = SimpleNamespace(**json.loads((args.run_dir / "run-options.json").read_text()))
    # Late learner startup must not let a faster learner train before all
    # initial pulls. Pacing also differs once the start barrier opens.
    if args.learner_id == 1:
        time.sleep(0.3)
    run_baseline_session(CPUTrainer(2), config, options, args.learner_id, args.run_dir, time.monotonic() + args.timeout)


if __name__ == "__main__":
    main()
