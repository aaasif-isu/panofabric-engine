import sys

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    t_content = f.read()

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
        self.items = deque()
        self.condition = threading.Condition()
        self.required = self.unfinished_tasks = self.peak = self.coalesced = 0

    def put_nowait(self, item, is_control=False):
        with self.condition:
            if self.required >= self.capacity:
                raise queue.Full
            self.required += 1
            self.items.append(((item, is_control), False))
            self.unfinished_tasks += 1
            self.peak = max(self.peak, len(self.items))
            self.condition.notify_all()

    def progress(self, body):
        with self.condition:
            for index, ((item, is_control), replaceable) in enumerate(self.items):
                if replaceable:
                    del self.items[index]
                    self.unfinished_tasks -= 1
                    self.coalesced += 1
                    break
            self.items.append(((("metadata", body), True), True))
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
            (item, is_control), replaceable = self.items.popleft()
            if not replaceable:
                self.required -= 1
            self.condition.notify_all()
            return item

    def get_control_nowait(self):
        with self.condition:
            for index, ((item, is_control), replaceable) in enumerate(self.items):
                if is_control:
                    del self.items[index]
                    if not replaceable:
                        self.required -= 1
                    self.condition.notify_all()
                    return item
            raise queue.Empty

    def task_done(self):
        with self.condition:
            self.unfinished_tasks -= 1
            self.condition.notify_all()"""
assert old_outbox in t_content
t_content = t_content.replace(old_outbox, new_outbox)

# 2. Update __init__
old_init = "self._incoming = queue.Queue(maxsize=capacity)\n        self._closed = threading.Event()"
new_init = "self._incoming = queue.Queue(maxsize=capacity)\n        self._active_transfers = {}\n        self._closed = threading.Event()"
assert old_init in t_content
t_content = t_content.replace(old_init, new_init)

# 3. Update send()
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
assert old_send in t_content
t_content = t_content.replace(old_send, new_send)

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(t_content)


with open('panoengine/decentralized/decoupled_heloco/gpu_training.py', 'r') as f:
    g_content = f.read()

# Remove the max fragment size limit
old_frag_check = """    for fragment in manager.fragments:
        # Snapshot contains two independent FP32 copies. Leave space for tensor
        # names/storage metadata, then verify the actual serialized payload.
        if fragment.numel * 8 + 65536 + len(fragment.parameter_names) * 1024 > MAX_FRAME_BYTES:
            raise ConfigError("a fragment snapshot exceeds 32 MiB; increase decoupled.num_fragments (large-tensor chunking is pending)")
        baseline = _cpu_copy(manager.select(fragment.fragment_id, parameters))"""
new_frag_check = """    for fragment in manager.fragments:
        # Snapshot chunking is implemented; no hard frame limit needed for large tensors.
        baseline = _cpu_copy(manager.select(fragment.fragment_id, parameters))"""
assert old_frag_check in g_content
g_content = g_content.replace(old_frag_check, new_frag_check)

with open('panoengine/decentralized/decoupled_heloco/gpu_training.py', 'w') as f:
    f.write(g_content)

print("Patch 4 applied.")
