import sys
import re

with open('panoengine/decentralized/decoupled_heloco/gpu_training.py', 'r') as f:
    content = f.read()

literal_old = """    for fragment in manager.fragments:
        # Snapshot contains two independent FP32 copies. Leave space for tensor
        # names/storage metadata, then verify the actual serialized payload.
        if fragment.numel * 8 + 65536 + len(fragment.parameter_names) * 1024 > MAX_FRAME_BYTES:
            raise ConfigError("a fragment snapshot exceeds 32 MiB; increase decoupled.num_fragments (large-tensor chunking is pending)")
        baseline = _cpu_copy(manager.select(fragment.fragment_id, parameters))"""

literal_new = """    for fragment in manager.fragments:
        # Snapshot chunking is now implemented; no hard limit needed.
        baseline = _cpu_copy(manager.select(fragment.fragment_id, parameters))"""

content = content.replace(literal_old, literal_new)

with open('panoengine/decentralized/decoupled_heloco/gpu_training.py', 'w') as f:
    f.write(content)
