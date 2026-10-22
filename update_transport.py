import sys
import re

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    content = f.read()

# 1. Update _Outbox
old_outbox = """class _Outbox:
    \"\"\"FIFO required frames plus one replaceable, unsent progress report.\"\"\"
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = deque()
        self.condition = threading.Condition()
        self.required = self.unfinished_tasks = self.peak = self.coalesced = 0

    def put_nowait(self, item):
        with self.condition:
            if self.required >= self.capacity:
                raise queue.Full
            self.required += 1
            self._append(item, False)

    def progress(self, body):
        with self.condition:
            for index, (_, replaceable) in enumerate(self.items):
                if replaceable:
                    del self.items[index]
                    self.unfinished_tasks -= 1
                    self.coalesced += 1
                    break
            # Append at the tail: newer counters cannot overtake an older ACK.
            self._append(("metadata", body), True)

    def _append(self, item, replaceable):
        self.items.append((item, replaceable))
        self.unfinished_tasks += 1
        self.peak = max(self.peak, len(self.items))
        self.condition.notify_all()

    def get(self, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self.condition.wait(remaining)
            item, replaceable = self.items.popleft()
            if not replaceable:
                self.required -= 1
            self.condition.notify_all()
            return item

    def task_done(self):
        with self.condition:
            self.unfinished_tasks -= 1
            self.condition.notify_all()"""

new_outbox = """class _Outbox:
    def __init__(self, capacity):
        self.capacity = capacity
        self.control_items = deque()
        self.bulk_items = deque()
        self.condition = threading.Condition()
        self.required = self.unfinished_tasks = self.peak = self.coalesced = 0

    def put_nowait(self, item, is_control=False):
        with self.condition:
            if not is_control:
                if self.required >= self.capacity:
                    raise queue.Full
                self.required += 1
                self.bulk_items.append((item, False))
            else:
                self.control_items.append((item, False))
            self.unfinished_tasks += 1
            self.peak = max(self.peak, len(self.control_items) + len(self.bulk_items))
            self.condition.notify_all()

    def progress(self, body):
        with self.condition:
            for index, (item, replaceable) in enumerate(self.control_items):
                if replaceable:
                    del self.control_items[index]
                    self.unfinished_tasks -= 1
                    self.coalesced += 1
                    break
            self.control_items.append((("metadata", body), True))
            self.unfinished_tasks += 1
            self.peak = max(self.peak, len(self.control_items) + len(self.bulk_items))
            self.condition.notify_all()

    def get(self, timeout):
        deadline = time.monotonic() + timeout
        with self.condition:
            while not self.control_items and not self.bulk_items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise queue.Empty
                self.condition.wait(remaining)
            if self.control_items:
                item, replaceable = self.control_items.popleft()
            else:
                item, replaceable = self.bulk_items.popleft()
                if not replaceable:
                    self.required -= 1
            self.condition.notify_all()
            return item

    def get_control_nowait(self):
        with self.condition:
            if not self.control_items:
                raise queue.Empty
            item, replaceable = self.control_items.popleft()
            self.condition.notify_all()
            return item

    def task_done(self):
        with self.condition:
            self.unfinished_tasks -= 1
            self.condition.notify_all()"""
content = content.replace(old_outbox, new_outbox)

# 2. Add _active_transfers to __init__
content = content.replace(
    "self._incoming = queue.Queue(maxsize=capacity)\n        self._closed = threading.Event()",
    "self._incoming = queue.Queue(maxsize=capacity)\n        self._active_transfers = {}\n        self._closed = threading.Event()"
)

# 3. Update diagnostics to count both queues
old_diag = """    def diagnostics(self):
        with self._outgoing.condition:
            return dict(self._stats, sent_bytes=self.sent_bytes,
                        received_bytes=self.received_bytes,
                        outgoing_peak=self._outgoing.peak,
                        progress_coalesced=self._outgoing.coalesced,
                        outgoing_pending=len(self._outgoing.items),
                        incoming_pending=self._incoming.qsize())"""

new_diag = """    def diagnostics(self):
        with self._outgoing.condition:
            return dict(self._stats, sent_bytes=self.sent_bytes,
                        received_bytes=self.received_bytes,
                        outgoing_peak=self._outgoing.peak,
                        progress_coalesced=self._outgoing.coalesced,
                        outgoing_pending=len(self._outgoing.control_items) + len(self._outgoing.bulk_items),
                        incoming_pending=self._incoming.qsize())"""
content = content.replace(old_diag, new_diag)

# 4. Modify send() to use is_control
old_send = """    def send(self, kind: str, body: dict):
        self.check()
        try:
            self._outgoing.put_nowait((kind, body))
        except queue.Full as exc:
            raise TransportBackpressure("outgoing TCP queue is full") from exc"""

new_send = """    def send(self, kind: str, body: dict):
        self.check()
        is_control = kind not in {"update", "snapshot"}
        try:
            self._outgoing.put_nowait((kind, body), is_control)
        except queue.Full as exc:
            raise TransportBackpressure("outgoing TCP queue is full") from exc"""
content = content.replace(old_send, new_send)

# 5. Modify _writer to handle bulk sending and flushing controls
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
                    started = time.monotonic()
                    payload, tensors = encode_message(kind, body)
                    self._stats["encode_seconds"] += time.monotonic() - started
                    started = time.monotonic()
                    frame = memoryview(struct.pack("!I", len(payload)) + payload + struct.pack("!I", len(tensors)))
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
                        self.sent_bytes += sent

                    for t in tensors:
                        view = t.view(torch.uint8).numpy()
                        data_view = memoryview(view).cast('B')

                        header = memoryview(struct.pack("!I", len(data_view)))
                        while header and not self._closed.is_set():
                            try:
                                sent = self.socket.send(header)
                            except socket.timeout:
                                continue
                            if not sent:
                                raise EOFError("peer disconnected while sending")
                            header = header[sent:]

                        while data_view and not self._closed.is_set():
                            try:
                                chunk_size = min(len(data_view), 256 * 1024)
                                sent = self.socket.send(data_view[:chunk_size])
                            except socket.timeout:
                                continue
                            if not sent:
                                raise EOFError("peer disconnected while sending")
                            if monitor:
                                monitor.record("download", sent)
                            data_view = data_view[sent:]
                            self.sent_bytes += sent

                    self._stats["send_seconds"] += time.monotonic() - started
                except Exception as exc:
                    self._fail(exc)
                    return
                finally:
                    self._outgoing.task_done()
        except Exception as exc:
            self._fail(exc)"""

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
        payload, tensors = encode_message(kind, body)
        self._stats["encode_seconds"] += time.monotonic() - started
        started = time.monotonic()
        frame = memoryview(struct.pack("!I", len(payload)) + payload + struct.pack("!I", len(tensors)))
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
            self.sent_bytes += sent

        for t in tensors:
            view = t.view(torch.uint8).numpy()
            data_view = memoryview(view).cast('B')
            header = memoryview(struct.pack("!I", len(data_view)))
            while header and not self._closed.is_set():
                try:
                    sent = self.socket.send(header)
                except socket.timeout:
                    continue
                if not sent:
                    raise EOFError("peer disconnected while sending")
                header = header[sent:]
            while data_view and not self._closed.is_set():
                try:
                    chunk_size = min(len(data_view), 256 * 1024)
                    sent = self.socket.send(data_view[:chunk_size])
                except socket.timeout:
                    continue
                if not sent:
                    raise EOFError("peer disconnected while sending")
                if monitor:
                    monitor.record("download", sent)
                data_view = data_view[sent:]
                self.sent_bytes += sent

        self._stats["send_seconds"] += time.monotonic() - started"""
content = content.replace(old_writer, new_writer)

# 6. Update receive()
old_receive = """    def receive(self):
        # Drain already received messages, including a final stopped message,
        # before surfacing an EOF that followed them.
        try:
            message = self._incoming.get_nowait()
            self._incoming.task_done()
            return message
        except queue.Empty:
            self.check()
            return None"""

new_receive = """    def receive(self):
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
content = content.replace(old_receive, new_receive)

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(content)
