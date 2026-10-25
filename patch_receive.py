import sys
import re

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    text = f.read()

def replace_receive(text):
    start = text.find('if kind in {"update_begin", "snapshot_begin"}:')
    end = text.find('if transfer["kind"] == "update":', start) + len('if transfer["kind"] == "update":')
    if start == -1 or end == -1:
        print("Could not find receive boundaries!")
        return text

    new_receive = """if kind in {"update_begin", "snapshot_begin"}:
                t_id = message["body"]["transfer_id"]
                if len(self._active_transfers) > 8:
                    raise ConnectionError("too many active transfers")
                self._active_transfers[t_id] = {
                    "body": message["body"],
                    "parameters": {},
                    "kind": "update" if kind == "update_begin" else "snapshot",
                    "received_chunks": {},
                    "total_chunks": {}
                }
                continue
            elif kind in {"update_chunk", "snapshot_chunk"}:
                t_id = message["body"]["transfer_id"]
                if t_id not in self._active_transfers:
                    continue
                transfer = self._active_transfers[t_id]
                name = message["body"]["name"]
                if name not in transfer["parameters"]:
                    import torch
                    chunk_data = message["body"]["data"]
                    shape = message["body"]["shape"]
                    try:
                        transfer["parameters"][name] = torch.empty(shape, dtype=chunk_data.dtype, device='cpu')
                    except Exception as exc:
                        raise ConnectionError(f"allocation failed for chunked tensor: {exc}")
                    transfer["received_chunks"][name] = 0
                    transfer["total_chunks"][name] = message["body"]["total_chunks"]
                
                flat_target = transfer["parameters"][name].view(-1)
                chunk_data = message["body"]["data"]
                start = message["body"]["start"]
                numel = chunk_data.numel()
                if start < 0 or start + numel > flat_target.numel():
                    raise ConnectionError("chunk out of bounds")
                
                flat_target[start:start+numel].copy_(chunk_data.view(-1))
                transfer["received_chunks"][name] += 1
                continue
            elif kind in {"update_commit", "snapshot_commit"}:
                t_id = message["body"]["transfer_id"]
                if t_id not in self._active_transfers:
                    continue
                transfer = self._active_transfers.pop(t_id)
                body = transfer["body"]
                parameters = transfer["parameters"]
                
                if len(parameters) != body.get("total_tensors", -1):
                    raise ConnectionError("missing tensors in transfer")
                for name, count in transfer["received_chunks"].items():
                    if count != transfer["total_chunks"][name]:
                        raise ConnectionError(f"missing chunks in tensor {name}")
                
                if transfer["kind"] == "update":"""

    return text[:start] + new_receive + text[end:]

text = replace_receive(text)

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(text)

