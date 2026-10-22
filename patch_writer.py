import sys

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    text = f.read()

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
                                raise EOFError("peer disconnected")
                            header = header[sent:]
                            self.sent_bytes += sent
                        
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

new_writer = """    def _flush_control_messages(self, monitor):
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
            if chunks == 0: chunks = 1
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
                    raise EOFError("peer disconnected")
                header = header[sent:]
                self.sent_bytes += sent
            
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

    def _writer(self):
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
            self._fail(exc)"""

if old_writer in text:
    text = text.replace(old_writer, new_writer)
    with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
        f.write(text)
    print("Successfully replaced _writer!")
else:
    print("Could not find old_writer. Maybe it has different formatting?")
