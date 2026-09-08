# Copyright (c) Panocular AI. All rights reserved.
"""
AsyncTrainer — Standalone Async/Sync Trainer Entry Point
=========================================================
Path A: New Async Trainer Entry Point (as per roadmap)

This module provides a unified trainer that supports both:
  1. Synchronous mode: torchft semi_sync_method=heloco (existing)
  2. Asynchronous mode: HTTP push/pull via AsyncDiLoCoServer/HeLoCoServer

The trainer is decoupled from torchtitan's FT manager and can be used
as a standalone entry point for research and production deployments.

Usage:
    python -m panoengine.decentralized.async_trainer \\
        --coordination-method async \\
        --model llama3_15m \\
        --steps 100 \\
        [other torchtitan flags...]

Features:
  - Reuses existing AsyncDiLoCoServer and HeLoCoServer
  - Supports both sync (torchft barrier) and async (HTTP) coordination
  - Configurable staleness weighting and MAX_WAIT_TIME
  - Compatible with existing parameter server setup
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def add_async_args(parser: argparse.ArgumentParser) -> None:
    """Add async trainer specific arguments to parser.
    
    Args:
        parser: ArgumentParser to add args to
    """
    group = parser.add_argument_group("Async Trainer Options")
    
    group.add_argument(
        "--coordination-method",
        choices=["sync", "async"],
        default="sync",
        help="Coordination method: sync (torchft barrier) or async (HTTP push/pull). "
        "(default: sync)",
    )
    
    group.add_argument(
        "--async-interval",
        type=int,
        default=1,
        help="Check staleness every N window pulses; staleness = current_model_id - worker_model_id. "
        "(default: 1)",
    )
    
    group.add_argument(
        "--max-wait-time",
        type=float,
        default=0.0,
        help="Max seconds to wait for parameter server response before timeout. "
        "0.0 = no timeout. (default: 0.0)",
    )
    
    group.add_argument(
        "--diloco-server-addr",
        type=str,
        default=None,
        help="Parameter server address (host:port). If not set, read from env DILOCO_SERVER_ADDR. "
        "(default: env or None)",
    )
    
    group.add_argument(
        "--diloco-hb-addr",
        type=str,
        default=None,
        help="Parameter server heartbeat address (host:port). If not set, read from env DILOCO_HB_ADDR. "
        "(default: env or None)",
    )


def get_async_config() -> dict:
    """Retrieve async configuration from command-line args and environment.
    
    Returns:
        dict with keys: coordination_method, async_interval, max_wait_time,
                       diloco_server_addr, diloco_hb_addr
    """
    parser = argparse.ArgumentParser(
        description="Async Trainer Configuration Parser"
    )
    add_async_args(parser)
    
    # Parse known args (let torchtitan handle unknown ones)
    args, _ = parser.parse_known_args()
    
    # Fallback to environment variables
    server_addr = (
        args.diloco_server_addr
        or os.environ.get("DILOCO_SERVER_ADDR")
    )
    hb_addr = (
        args.diloco_hb_addr
        or os.environ.get("DILOCO_HB_ADDR")
    )
    
    return {
        "coordination_method": args.coordination_method,
        "async_interval": args.async_interval,
        "max_wait_time": args.max_wait_time,
        "diloco_server_addr": server_addr,
        "diloco_hb_addr": hb_addr,
    }


def setup_async_trainer(
    coordination_method: str = "sync",
    async_interval: int = 1,
    max_wait_time: float = 0.0,
    diloco_server_addr: Optional[str] = None,
    diloco_hb_addr: Optional[str] = None,
) -> None:
    """Configure async trainer parameters in environment for torchtitan FT.
    
    This function sets up the environment variables and configuration
    that the torchtitan FT manager expects.
    
    Args:
        coordination_method: "sync" or "async"
        async_interval: Staleness check interval
        max_wait_time: Max wait timeout
        diloco_server_addr: Parameter server address
        diloco_hb_addr: Heartbeat server address
    """
    if coordination_method not in ("sync", "async"):
        raise ValueError(
            f"coordination_method must be 'sync' or 'async', got {coordination_method}"
        )
    
    # Set environment for FT manager
    os.environ["COORDINATION_METHOD"] = coordination_method
    os.environ["ASYNC_INTERVAL"] = str(async_interval)
    os.environ["MAX_WAIT_TIME"] = str(max_wait_time)
    
    if diloco_server_addr:
        os.environ["DILOCO_SERVER_ADDR"] = diloco_server_addr
    if diloco_hb_addr:
        os.environ["DILOCO_HB_ADDR"] = diloco_hb_addr
    
    logger.info(
        f"Async trainer configured: method={coordination_method}, "
        f"async_interval={async_interval}, max_wait_time={max_wait_time}"
    )
    if diloco_server_addr:
        logger.info(f"  Server: {diloco_server_addr}")
    if diloco_hb_addr:
        logger.info(f"  Heartbeat: {diloco_hb_addr}")


def main() -> None:
    """Entry point for standalone async trainer."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
    )
    
    logger.info("Async Trainer Entry Point (Path A)")
    logger.info(f"coordination_method: {os.environ.get('COORDINATION_METHOD', 'sync')}")
    
    # This is a configuration layer; actual training is delegated to torchtitan
    # or a custom training loop. For now, just display configuration.
    config = get_async_config()
    logger.info(f"Async configuration: {config}")


if __name__ == "__main__":
    main()
