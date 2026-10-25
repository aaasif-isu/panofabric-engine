import sys

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    text = f.read()

old_put = """    def put_nowait(self, item, is_control=False):
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
            self.condition.notify_all()"""

new_put = """    def put_nowait(self, item, is_control=False):
        with self.condition:
            if self.required >= self.capacity:
                raise queue.Full
            self.required += 1
            if not is_control:
                self.bulk_items.append((item, False))
            else:
                self.control_items.append((item, False))
            self.unfinished_tasks += 1
            self.peak = max(self.peak, len(self.control_items) + len(self.bulk_items))
            self.condition.notify_all()"""

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

old_get_ctrl = """    def get_control_nowait(self):
        with self.condition:
            if not self.control_items:
                raise queue.Empty
            item, replaceable = self.control_items.popleft()
            self.condition.notify_all()
            return item"""

new_get_ctrl = """    def get_control_nowait(self):
        with self.condition:
            if not self.control_items:
                raise queue.Empty
            item, replaceable = self.control_items.popleft()
            if not replaceable:
                self.required -= 1
            self.condition.notify_all()
            return item"""


if old_put in text:
    text = text.replace(old_put, new_put)
    print("Replaced put_nowait")
else:
    print("Could not find old_put")

if old_get in text:
    text = text.replace(old_get, new_get)
    print("Replaced get")
else:
    print("Could not find old_get")

if old_get_ctrl in text:
    text = text.replace(old_get_ctrl, new_get_ctrl)
    print("Replaced get_control_nowait")
else:
    print("Could not find old_get_ctrl")

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(text)

