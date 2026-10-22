import sys

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    text = f.read()

old_send = """    def send(self, kind: str, body: dict):
        self.check()
        is_control = kind not in {"update", "snapshot"}
        if kind == "update": kind = "update_bulk"
        if kind == "snapshot": kind = "snapshot_bulk"
        try:
            self._outgoing.put_nowait((kind, body), is_control)
        except queue.Full as exc:
            raise TransportBackpressure("outgoing TCP queue is full") from exc"""

new_send = """    def send(self, kind: str, body: dict):
        self.check()
        is_control = kind not in {"update", "snapshot"}
        try:
            self._outgoing.put_nowait((kind, body), is_control)
        except queue.Full as exc:
            raise TransportBackpressure("outgoing TCP queue is full") from exc"""

text = text.replace(old_send, new_send)

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(text)
