import sys

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    text = f.read()

# Fix 1: _Outbox accounting
old_get = """    def get(self, timeout):
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
            return item"""

new_get = """    def get(self, timeout):
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
            return item"""

text = text.replace(old_get, new_get)

old_get_ctrl = """    def get_control_nowait(self):
        with self.condition:
            if not self.control_items:
                raise queue.Empty
            item, replaceable = self.control_items.popleft()
            self.unfinished_tasks -= 1
            self.condition.notify_all()
            return item"""

new_get_ctrl = """    def get_control_nowait(self):
        with self.condition:
            if not self.control_items:
                raise queue.Empty
            item, replaceable = self.control_items.popleft()
            self.condition.notify_all()
            return item"""

text = text.replace(old_get_ctrl, new_get_ctrl)

# Fix 2: diagnostics() obsolete attribute
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

text = text.replace(old_diag, new_diag)

# Fix 3: _writer missing chunking protocol
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
                        except socket.timeout:
                            continue
                        if not sent:
                            raise EOFError("peer disconnected while sending")
                        if monitor:
                            monitor.record("download", sent)
                        frame = frame[sent:]
                    self._stats["send_seconds"] += time.monotonic() - started
                    self.sent_bytes += 4 + len(payload)
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
                    if kind in {"update_bulk", "snapshot_bulk"}:
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

text = text.replace(old_writer, new_writer)

# Fix 4: receive() needs "version": 1 in returned dicts
old_receive_ret1 = 'return {"kind": "update", "body": result_body}'
new_receive_ret1 = 'return {"version": 1, "kind": "update", "body": result_body}'
text = text.replace(old_receive_ret1, new_receive_ret1)

old_receive_ret2 = 'return {"kind": "snapshot", "body": result_body}'
new_receive_ret2 = 'return {"version": 1, "kind": "snapshot", "body": result_body}'
text = text.replace(old_receive_ret2, new_receive_ret2)

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(text)

print("transport.py patched")
