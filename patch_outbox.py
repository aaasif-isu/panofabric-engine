import sys

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'r') as f:
    text = f.read()

old_put = """    def put_nowait(self, item, is_control=False):
        with self.condition:
            if self.required >= self.capacity:
                raise queue.Full
            self.required += 1
            if not is_control:
                self.bulk_items.append((item, False))
            else:
                self.control_items.append((item, False))
            self.unfinished_tasks += 1"""

new_put = """    def put_nowait(self, item, is_control=False):
        with self.condition:
            if not is_control:
                if self.required >= self.capacity:
                    raise queue.Full
                self.required += 1
                self.bulk_items.append((item, False))
            else:
                self.control_items.append((item, False))
            self.unfinished_tasks += 1"""

old_get_control = """    def get_control_nowait(self):
        with self.condition:
            if not self.control_items:
                raise queue.Empty
            item, replaceable = self.control_items.popleft()
            if not replaceable:
                self.required -= 1
            self.condition.notify_all()
            return item"""

new_get_control = """    def get_control_nowait(self):
        with self.condition:
            if not self.control_items:
                raise queue.Empty
            item, replaceable = self.control_items.popleft()
            self.condition.notify_all()
            return item"""

text = text.replace(old_put, new_put)
text = text.replace(old_get_control, new_get_control)

with open('panoengine/decentralized/decoupled_heloco/transport.py', 'w') as f:
    f.write(text)

