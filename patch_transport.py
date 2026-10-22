import sys

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    text = f.read()

# Block 1: _Outbox
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

if old_outbox in text:
    print("_Outbox found!")
else:
    print("_Outbox NOT found!")

# Block 2: FramedTransport init
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
if old_init in text:
    print("Init found!")
else:
    print("Init NOT found!")

# Block 3: diagnostics
old_diag = """    def diagnostics(self):
        with self._outgoing.condition:
            return dict(self._stats, sent_bytes=self.sent_bytes,
                        received_bytes=self.received_bytes,
                        outgoing_peak=self._outgoing.peak,
                        progress_coalesced=self._outgoing.coalesced,
                        outgoing_pending=len(self._outgoing.items),
                        incoming_pending=self._incoming.qsize())"""
if old_diag in text:
    print("Diag found!")
else:
    print("Diag NOT found!")

# Block 4: send
old_send = """    def send(self, kind: str, body: dict):
        self.check()
        try:
            self._outgoing.put_nowait((kind, body))
        except queue.Full as exc:
            raise TransportBackpressure("outgoing TCP queue is full") from exc"""
if old_send in text:
    print("Send found!")
else:
    print("Send NOT found!")

