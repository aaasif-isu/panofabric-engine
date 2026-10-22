import sys

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    t_content = f.read()

# I will replace `class TransportBackpressure(ConnectionError):` with the tensor extraction helpers
# and keep `TransportBackpressure`

helpers = """import torch

def _extract_tensors(obj, tensors):
    if isinstance(obj, dict):
        return {k: _extract_tensors(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_extract_tensors(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_extract_tensors(v, tensors) for v in obj)
    elif isinstance(obj, torch.Tensor):
        obj = obj.detach().contiguous()
        tensors.append(obj)
        return {"__tensor_index__": len(tensors) - 1}
    else:
        return obj

def _inject_tensors(obj, tensors):
    if isinstance(obj, dict):
        if "__tensor_index__" in obj:
            return tensors[obj["__tensor_index__"]]
        return {k: _inject_tensors(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_inject_tensors(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_inject_tensors(v, tensors) for v in obj)
    else:
        return obj

class TransportBackpressure(ConnectionError):"""

t_content = t_content.replace("class TransportBackpressure(ConnectionError):", helpers)

old_writer = """    def _writer(self):
        try:
            from panoengine.decentralized.bandwidth import get_monitor
            monitor = get_monitor()
        except ImportError:
            monitor = None
        try:
            while not self._closed.is_set():
                try:
                    kind, body = self._outgoing.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if kind in {"update", "snapshot"}:
                        self._send_bulk(kind, body, monitor)
                    else:
                        self._send_frame(kind, body, monitor)
                except Exception as exc:
                    self._fail(exc)
                    return
                finally:
                    self._outgoing.task_done()
        except Exception as exc:
            self._fail(exc)

    def _flush_control_messages(self, monitor):
        while not self._closed.is_set():
            try:
                item = self._outgoing.get_control_nowait()
                kind, body = item
                try:
                    self._send_frame(kind, body, monitor)
                finally:
                    self._outgoing.task_done()
            except queue.Empty:
                break

    def _send_bulk(self, kind, body, monitor):
        import time
        if kind == "update":
            parameters = body["parameters"]
            transfer_id = f"upd_{body['fragment_id']}_{body['revision']}"
            begin_kind = "update_begin"
            chunk_kind = "update_chunk"
            commit_kind = "update_commit"
            begin_body = {
                "transfer_id": transfer_id,
                "fragment_id": body["fragment_id"],
                "revision": body["revision"],
                "layout_signature": body["layout_signature"],
                "global_step": body.get("global_step", 0),
                "acked_request_id": body.get("acked_request_id"),
                "total_tensors": len(parameters)
            }
        else:
            snapshot = body["snapshot"]
            transfer_id = f"snap_{snapshot['fragment_id']}_{snapshot['base_revision']}_{time.monotonic()}"
            parameters = snapshot["pseudo_gradient"]
            begin_kind = "snapshot_begin"
            chunk_kind = "snapshot_chunk"
            commit_kind = "snapshot_commit"
            begin_body = {
                "transfer_id": transfer_id,
                "request_id": body["request_id"],
                "fragment_id": snapshot["fragment_id"],
                "layout_signature": snapshot["layout_signature"],
                "base_revision": snapshot["base_revision"],
                "local_steps": snapshot["local_steps"],
                "tokens": snapshot["tokens"],
                "total_tensors": len(parameters)
            }
            
        self._send_frame(begin_kind, begin_body, monitor)
        
        for name, tensor in parameters.items():
            if self._closed.is_set(): break
            flat = tensor.detach().flatten()
            numel = flat.numel()
            element_size = flat.element_size()
            chunk_size_elements = (16 * 1024 * 1024) // element_size
            if chunk_size_elements < 1: chunk_size_elements = 1
            chunks = (numel + chunk_size_elements - 1) // chunk_size_elements
            for i in range(chunks):
                if self._closed.is_set(): break
                start = i * chunk_size_elements
                end = min(start + chunk_size_elements, numel)
                chunk_tensor = flat[start:end].clone()
                chunk_body = {
                    "transfer_id": transfer_id,
                    "name": name,
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "chunk_index": i,
                    "total_chunks": chunks,
                    "start": start,
                    "data": chunk_tensor
                }
                self._send_frame(chunk_kind, chunk_body, monitor)
                self._flush_control_messages(monitor)
                
        if not self._closed.is_set():
            self._send_frame(commit_kind, {"transfer_id": transfer_id}, monitor)
        self._flush_control_messages(monitor)

    def _send_frame(self, kind, body, monitor):
        started = time.monotonic()
        payload = encode_message(kind, body)
        self._stats["encode_seconds"] += time.monotonic() - started
        started = time.monotonic()
        frame = memoryview(struct.pack("!I", len(payload)) + payload)
        while frame and not self._closed.is_set():
            try:
                chunk_size = min(len(frame), 256 * 1024)
                sent = self.socket.send(frame[:chunk_size])
            except socket.timeout:
                continue
            if not sent:
                raise EOFError("peer disconnected while sending")
            if monitor:
                monitor.record("download", sent)
            frame = frame[sent:]
        self._stats["send_seconds"] += time.monotonic() - started
        self.sent_bytes += 4 + len(payload)"""

new_writer = """    def _writer(self):
        try:
            from panoengine.decentralized.bandwidth import get_monitor
            monitor = get_monitor()
        except ImportError:
            monitor = None
        try:
            while not self._closed.is_set():
                try:
                    kind, body = self._outgoing.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if kind in {"update", "snapshot"}:
                        self._send_bulk(kind, body, monitor)
                    else:
                        self._send_frame(kind, body, monitor)
                except Exception as exc:
                    self._fail(exc)
                    return
                finally:
                    self._outgoing.task_done()
        except Exception as exc:
            self._fail(exc)

    def _flush_control_messages(self, monitor):
        while not self._closed.is_set():
            try:
                item = self._outgoing.get_control_nowait()
                kind, body = item
                try:
                    self._send_frame(kind, body, monitor)
                finally:
                    self._outgoing.task_done()
            except queue.Empty:
                break

    def _send_bulk(self, kind, body, monitor):
        import time
        tensors = []
        body_safe = _extract_tensors(body, tensors)
        transfer_id = f"bulk_{time.monotonic()}_{id(body)}"
        
        begin_body = {
            "transfer_id": transfer_id,
            "original_kind": kind,
            "body_safe": body_safe,
            "tensor_metadata": [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in tensors]
        }
        self._send_frame("bulk_begin", begin_body, monitor)
        
        for idx, tensor in enumerate(tensors):
            if self._closed.is_set(): break
            flat = tensor.view(-1)
            numel = flat.numel()
            element_size = flat.element_size()
            chunk_size_elements = (16 * 1024 * 1024) // element_size
            if chunk_size_elements < 1: chunk_size_elements = 1
            chunks = (numel + chunk_size_elements - 1) // chunk_size_elements
            for i in range(chunks):
                if self._closed.is_set(): break
                start = i * chunk_size_elements
                end = min(start + chunk_size_elements, numel)
                chunk_tensor = flat[start:end].clone()
                chunk_body = {
                    "transfer_id": transfer_id,
                    "tensor_index": idx,
                    "start": start,
                    "data": chunk_tensor
                }
                self._send_frame("bulk_chunk", chunk_body, monitor)
                self._flush_control_messages(monitor)
                
        if not self._closed.is_set():
            self._send_frame("bulk_commit", {"transfer_id": transfer_id}, monitor)
        self._flush_control_messages(monitor)

    def _send_frame(self, kind, body, monitor):
        started = time.monotonic()
        payload = encode_message(kind, body)
        self._stats["encode_seconds"] += time.monotonic() - started
        started = time.monotonic()
        frame = memoryview(struct.pack("!I", len(payload)) + payload)
        while frame and not self._closed.is_set():
            try:
                chunk_size = min(len(frame), 256 * 1024)
                sent = self.socket.send(frame[:chunk_size])
            except socket.timeout:
                continue
            if not sent:
                raise EOFError("peer disconnected while sending")
            if monitor:
                monitor.record("download", sent)
            frame = frame[sent:]
        self._stats["send_seconds"] += time.monotonic() - started
        self.sent_bytes += 4 + len(payload)"""

assert old_writer in t_content
t_content = t_content.replace(old_writer, new_writer)

old_receive = """    def receive(self):
        while True:
            try:
                message = self._incoming.get_nowait()
                self._incoming.task_done()
            except queue.Empty:
                self.check()
                return None
                
            kind = message["kind"]
            if kind in {"update_begin", "snapshot_begin"}:
                t_id = message["body"]["transfer_id"]
                self._active_transfers[t_id] = {
                    "body": message["body"],
                    "parameters": {},
                    "kind": "update" if kind == "update_begin" else "snapshot"
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
                    transfer["parameters"][name] = torch.empty(shape, dtype=chunk_data.dtype, device='cpu')
                
                flat_target = transfer["parameters"][name].view(-1)
                chunk_data = message["body"]["data"]
                start = message["body"]["start"]
                flat_target[start:start+chunk_data.numel()].copy_(chunk_data.view(-1))
                continue
            elif kind in {"update_commit", "snapshot_commit"}:
                t_id = message["body"]["transfer_id"]
                if t_id not in self._active_transfers:
                    continue
                transfer = self._active_transfers.pop(t_id)
                body = transfer["body"]
                parameters = transfer["parameters"]
                if transfer["kind"] == "update":
                    result_body = {
                        "fragment_id": body["fragment_id"],
                        "revision": body["revision"],
                        "layout_signature": body["layout_signature"],
                        "global_step": body["global_step"],
                        "acked_request_id": body["acked_request_id"],
                        "parameters": parameters
                    }
                    return {"version": 1, "kind": "update", "body": result_body}
                else:
                    result_body = {
                        "request_id": body["request_id"],
                        "snapshot": {
                            "fragment_id": body["fragment_id"],
                            "layout_signature": body["layout_signature"],
                            "base_revision": body["base_revision"],
                            "local_steps": body["local_steps"],
                            "tokens": body["tokens"],
                            "pseudo_gradient": parameters
                        }
                    }
                    return {"version": 1, "kind": "snapshot", "body": result_body}
            else:
                return message"""

new_receive = """    def receive(self):
        while True:
            try:
                message = self._incoming.get_nowait()
                self._incoming.task_done()
            except queue.Empty:
                self.check()
                return None
                
            kind = message["kind"]
            if kind == "bulk_begin":
                t_id = message["body"]["transfer_id"]
                tensors = []
                import torch
                for meta in message["body"]["tensor_metadata"]:
                    dtype = getattr(torch, meta["dtype"].split('.')[-1])
                    tensors.append(torch.empty(tuple(meta["shape"]), dtype=dtype, device='cpu'))
                
                self._active_transfers[t_id] = {
                    "original_kind": message["body"]["original_kind"],
                    "body_safe": message["body"]["body_safe"],
                    "tensors": tensors
                }
                continue
            elif kind == "bulk_chunk":
                t_id = message["body"]["transfer_id"]
                if t_id not in self._active_transfers:
                    continue
                transfer = self._active_transfers[t_id]
                idx = message["body"]["tensor_index"]
                chunk_data = message["body"]["data"]
                start = message["body"]["start"]
                transfer["tensors"][idx].view(-1)[start:start+chunk_data.numel()].copy_(chunk_data.view(-1))
                continue
            elif kind == "bulk_commit":
                t_id = message["body"]["transfer_id"]
                if t_id not in self._active_transfers:
                    continue
                transfer = self._active_transfers.pop(t_id)
                body = _inject_tensors(transfer["body_safe"], transfer["tensors"])
                return {"version": 1, "kind": transfer["original_kind"], "body": body}
            else:
                return message"""

assert old_receive in t_content
t_content = t_content.replace(old_receive, new_receive)

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(t_content)

