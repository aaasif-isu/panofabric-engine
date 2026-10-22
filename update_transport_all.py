import sys
import re

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    content = f.read()

# 1. Update _Outbox class
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
            return item"""

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
            self.unfinished_tasks -= 1
            self.condition.notify_all()
            return item

    def get_control_nowait(self):
        with self.condition:
            if not self.control_items:
                raise queue.Empty
            item, replaceable = self.control_items.popleft()
            self.unfinished_tasks -= 1
            self.condition.notify_all()
            return item"""

content = content.replace(old_outbox, new_outbox)

# 2. update FramedTransport.__init__
old_init = """    def __init__(self, connection: socket.socket, *, capacity: int = 32):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("transport capacity must be a positive integer")
        self.socket = connection
        self.socket.settimeout(1.0)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._outgoing = _Outbox(capacity)
        self._stats = {"backpressure_retries": 0, "incoming_full_retries": 0, "encode_seconds": 0., "send_seconds": 0.}
        self._incoming = queue.Queue(maxsize=capacity)
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._error = None
        self.sent_bytes = 0
        self.received_bytes = 0
        self._threads = [threading.Thread(target=self._writer, daemon=True), threading.Thread(target=self._reader, daemon=True)]
        for thread in self._threads:
            thread.start()"""

new_init = """    def __init__(self, connection: socket.socket, *, capacity: int = 32):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("transport capacity must be a positive integer")
        self.socket = connection
        self.socket.settimeout(1.0)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._outgoing = _Outbox(capacity)
        self._stats = {"backpressure_retries": 0, "incoming_full_retries": 0, "encode_seconds": 0., "send_seconds": 0.}
        self._incoming = queue.Queue(maxsize=capacity)
        self._active_transfers = {}
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._error = None
        self.sent_bytes = 0
        self.received_bytes = 0
        self._threads = [threading.Thread(target=self._writer, daemon=True), threading.Thread(target=self._reader, daemon=True)]
        for thread in self._threads:
            thread.start()"""
content = content.replace(old_init, new_init)

# 3. update send
old_send = """    def send(self, kind: str, body: dict):
        self.check()
        try:
            self._outgoing.put_nowait((kind, body))
        except queue.Full as exc:
            raise TransportBackpressure("outgoing TCP queue is full") from exc"""

new_send = """    def send(self, kind: str, body: dict):
        self.check()
        is_control = kind not in {"update", "snapshot"}
        if kind == "update": kind = "update_bulk"
        if kind == "snapshot": kind = "snapshot_bulk"
        try:
            self._outgoing.put_nowait((kind, body), is_control)
        except queue.Full as exc:
            raise TransportBackpressure("outgoing TCP queue is full") from exc"""
content = content.replace(old_send, new_send)

# 4. update _writer
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
                    payload = encode_message(kind, body)
                    self._stats["encode_seconds"] += time.monotonic() - started
                    started = time.monotonic()
                    frame = memoryview(struct.pack("!I", len(payload)) + payload)
                    while frame and not self._closed.is_set():
                        try:
                            chunk_size = min(len(frame), 256 * 1024)
                            sent = self.socket.send(frame[:chunk_size])
                            frame = frame[sent:]
                            self.sent_bytes += sent
                            if monitor is not None:
                                monitor.record_send(sent)
                        except socket.timeout:
                            continue
                        except BlockingIOError:
                            time.sleep(0.001)
                            continue
                    self._stats["send_seconds"] += time.monotonic() - started
                except Exception as exc:
                    raise ConnectionError("wire serialization or send failed") from exc
        except (ConnectionError, OSError) as exc:
            self._fail(exc)
        finally:
            self._closed.set()"""

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
                    if kind in {"update_bulk", "snapshot_bulk"}:
                        self._send_bulk(kind, body, monitor)
                    else:
                        self._send_frame(kind, body, monitor)
                except Exception as exc:
                    raise ConnectionError("wire serialization or send failed") from exc
        except (ConnectionError, OSError) as exc:
            self._fail(exc)
        finally:
            self._closed.set()

    def _flush_control_messages(self, monitor):
        while not self._closed.is_set():
            try:
                item = self._outgoing.get_control_nowait()
                kind, body = item
                self._send_frame(kind, body, monitor)
            except queue.Empty:
                break

    def _send_bulk(self, kind, body, monitor):
        if kind == "update_bulk":
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
            import time
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
                frame = frame[sent:]
                self.sent_bytes += sent
                if monitor is not None:
                    monitor.record_send(sent)
            except socket.timeout:
                continue
            except BlockingIOError:
                time.sleep(0.001)
                continue
        self._stats["send_seconds"] += time.monotonic() - started"""
content = content.replace(old_writer, new_writer)

# 5. update receive
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
                    return {"kind": "update", "body": result_body}
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
                    return {"kind": "snapshot", "body": result_body}
            else:
                return message"""
content = content.replace(old_receive, new_receive)

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(content)
