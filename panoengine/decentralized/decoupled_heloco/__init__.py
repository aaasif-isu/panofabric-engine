"""Isolated development of Decoupled HeLoCo.

Phase 6 adds localhost TCP learner processes and timed quorum/grace control to
the HeLoCo CPU prototype. Production GPU/FSDP trainer integration comes later.
Importing this package does not import torch, torchft, or existing HeLoCo code.
"""
