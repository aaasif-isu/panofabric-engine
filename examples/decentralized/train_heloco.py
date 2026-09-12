# Copyright (c) Panocular AI
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
import csv

REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))
os.environ["CUDA_VISIBLE_DEVICES"] = str(REPLICA_GROUP_ID % 4)

import torch
from torch import nn, optim
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter

from panoengine.decentralized.heloco import HeLoCoOptimizer, HeLoCoServer, HeLoCoWorker

# Seed initialization for reproducibility
_pf_seed = int(os.environ.get("PANOENGINE_SEED", 42))
torch.manual_seed(_pf_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_pf_seed)
import numpy as np
np.random.seed(_pf_seed)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@record
def main() -> None:
    REPLICA_GROUP_ID = int(os.environ.get("REPLICA_GROUP_ID", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_hid = 128
    n_layers = 2
    num_workers = 4

    class MLPModule(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Linear(d_hid, d_hid), nn.ReLU()] * n_layers
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 10000

        def __getitem__(self, _):
            return torch.rand(d_hid), int(torch.randint(0, 10, (1,)))

    server_addr = os.environ.get("HELOCO_SERVER_ADDR", "")
    hb_addr = os.environ.get("HELOCO_HEARTBEAT_ADDR", "")
    server = None
    if not server_addr:
        global_model = MLPModule()
        outer_optimizer: optim.Optimizer = HeLoCoOptimizer(
            global_model.parameters(), lr=0.7, momentum=0.9,
        )
        server = HeLoCoServer(
            global_model, outer_optimizer, port=0,
            rho=1.0 / (num_workers ** 0.5),
            dylu_H=20, grace_period=0.5,
        )
        server_addr = server.address()
        hb_addr = server.heartbeat_address()
        logger.info(f"HeLoCoServer started at {server_addr} (heartbeat: {hb_addr})")

    if os.environ.get("SERVER_ONLY", "0") == "1":
        logger.info(
            "SERVER_ONLY=1: set HELOCO_SERVER_ADDR=%s HELOCO_HEARTBEAT_ADDR=%s in workers",
            server_addr, hb_addr,
        )
        while True:
            time.sleep(3600)

    m = MLPModule().to(device)
    if server is not None:
        m.load_state_dict(server._model.state_dict())

    inner_optimizer: optim.Optimizer = torch.optim.AdamW(
        m.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
    )
    criterion = nn.CrossEntropyLoss()

    trainloader = torch.utils.data.DataLoader(
        DummyDataset(), batch_size=64, num_workers=2, shuffle=True
    )

    output_folder = f"output/replica-{REPLICA_GROUP_ID}"
    os.makedirs(output_folder, exist_ok=True)
    writer = SummaryWriter(f"{output_folder}/tensorboard", max_queue=1000)
    
    # Training loss tracking for reproducibility analysis
    loss_history_path = os.path.join(output_folder, "training_loss_history.csv")
    loss_history_file = open(loss_history_path, "w", newline="")
    loss_writer = csv.DictWriter(loss_history_file, fieldnames=["step", "loss"])
    loss_writer.writeheader()

    num_params = sum(p.numel() for p in m.parameters())
    logger.info(f"Worker {REPLICA_GROUP_ID}: {num_params:,} params")

    with HeLoCoWorker(
        server_address=server_addr,
        model=m,
        inner_optimizer=inner_optimizer,
        sync_every=20,
        heartbeat_address=hb_addr or None,
    ):
        # Log step 0: initial loss before any training updates
        step = 0
        try:
            first_batch = next(iter(trainloader))
            first_inputs, first_labels = first_batch
            first_inputs = first_inputs.to(device)
            first_labels = first_labels.to(device)
            
            with torch.no_grad():
                initial_out = m(first_inputs)
                initial_loss = criterion(initial_out, first_labels)
            
            loss_writer.writerow({"step": 0, "loss": f"{initial_loss.item():.6f}"})
            loss_history_file.flush()
            logger.info(f"[worker {REPLICA_GROUP_ID}] step=0 (initial) loss={initial_loss.item():.4f}")
        except Exception as e:
            logger.warning(f"[worker {REPLICA_GROUP_ID}] Failed to log step 0 loss: {e}")
        
        step = 0
        while True:
            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                inner_optimizer.zero_grad()
                out = m(inputs)
                loss = criterion(out, labels)
                loss.backward()
                inner_optimizer.step()

                writer.add_scalar("loss", loss.item(), step)
                loss_writer.writerow({"step": step, "loss": f"{loss.item():.6f}"})
                loss_history_file.flush()

                if step % 100 == 0:
                    logger.info(f"[worker {REPLICA_GROUP_ID}] step={step} loss={loss.item():.4f}")

                step += 1
                if step >= 1000:
                    writer.flush()
                    loss_history_file.close()
                    return


if __name__ == "__main__":
    main()
