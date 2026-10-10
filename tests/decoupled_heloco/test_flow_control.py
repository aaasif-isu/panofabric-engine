"""Congestion retains required frames; progress has a bounded latest slot."""
import queue
import threading
import time
import unittest
from unittest.mock import patch

import test_transport
from test_transport import _wait
from panoengine.decentralized.decoupled_heloco.transport import _Outbox, encode_message
from panoengine.decentralized.decoupled_heloco.sharding import ShardedOptimizer


class FlowControlTests(unittest.TestCase):
    pair = test_transport.TransportTests.pair

    def test_progress_order_and_bound(self):
        out = _Outbox(2)
        out.progress({'steps': 1})
        out.put_nowait(('snapshot', {'id': 1}))
        out.progress({'steps': 2})
        out.put_nowait(('update_ack', {'id': 2}))
        out.progress({'steps': 500})
        with self.assertRaises(queue.Full):
            out.put_nowait(('stopped', {}))
        self.assertEqual(out.peak, 3)
        self.assertEqual(out.coalesced, 2)
        self.assertEqual([out.get(.1) for _ in range(3)], [
            ('snapshot', {'id': 1}), ('update_ack', {'id': 2}),
            ('metadata', {'steps': 500})])
        for _ in range(3): out.task_done()
        self.assertEqual(out.unfinished_tasks, 0)

    def test_slow_reader_survives_more_than_old_timeout(self):
        sender, receiver = self.pair(capacity=1)
        for number in range(3):
            sender.send_reliable('snapshot', {'id': number}, deadline=time.monotonic()+3)
        # Old incoming.put(timeout=1) killed the reader here.
        time.sleep(1.2)
        received = [_wait(receiver.receive, 3)['body']['id'] for _ in range(3)]
        self.assertEqual(received, [0, 1, 2])
        self.assertGreater(receiver.diagnostics()['incoming_full_retries'], 0)
        receiver.check()

    def test_required_retry_and_latest_counters(self):
        sender, receiver = self.pair(capacity=1)
        entered, resume = threading.Event(), threading.Event()
        original = encode_message
        def slow(kind, body):
            if kind == 'snapshot' and body.get('id') == 0:
                entered.set()
                if not resume.wait(3): raise TimeoutError('test writer blocked')
            return original(kind, body)
        with patch('panoengine.decentralized.decoupled_heloco.transport.encode_message', slow):
            sender.send('snapshot', {'id': 0})
            self.assertTrue(entered.wait(2))
            sender.send('update_ack', {'id': 1})
            for step in range(500): sender.send_progress({'steps': step+1, 'tokens': (step+1)*512})
            with self.assertRaises(TimeoutError):
                sender.send_reliable('paused', {}, deadline=time.monotonic()+.03)
            resume.set()
            frames = [_wait(receiver.receive, 3) for _ in range(3)]
            sender.send_reliable('stopped', {}, deadline=time.monotonic()+3)
            frames.append(_wait(receiver.receive, 3))
        self.assertEqual([x['kind'] for x in frames], ['snapshot', 'update_ack', 'metadata', 'stopped'])
        self.assertEqual(frames[2]['body'], {'steps':500, 'tokens':256000})
        self.assertEqual(sender.diagnostics()['progress_coalesced'],499)
        self.assertLessEqual(sender.diagnostics()['outgoing_peak'],2)

    def test_large_frames_retry_partial_socket_sends(self):
        import torch
        sender, receiver = self.pair(capacity=2)
        sender.socket.setsockopt(__import__('socket').SOL_SOCKET, __import__('socket').SO_SNDBUF, 8192)
        tensor = torch.arange(512*1024, dtype=torch.float32)
        failures = []
        def produce():
            try:
                for number in range(5):
                    sender.send_reliable('snapshot', {'id': number, 'tensor': tensor}, deadline=time.monotonic()+10)
                sender.flush(10)
            except Exception as exc:
                failures.append(exc)
        thread = threading.Thread(target=produce)
        thread.start()
        # Fill the receiver queue and TCP buffers; exceed socket timeout.
        time.sleep(1.3)
        try:
            for number in range(5):
                frame = _wait(receiver.receive, 5)
                self.assertEqual(frame['body']['id'], number)
                self.assertTrue(torch.equal(frame['body']['tensor'], tensor))
        finally:
            thread.join(10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])

    def test_replica_wait_pumps_on_owner_thread(self):
        optimizer = object.__new__(ShardedOptimizer)
        owner = threading.get_ident()
        calls = []
        optimizer.progress = lambda: calls.append(threading.get_ident())
        class Connection:
            def poll(self, timeout):
                time.sleep(timeout)
                return len(calls) >= 3
            def recv_bytes(self):
                from panoengine.decentralized.decoupled_heloco.sharding import _pack
                return _pack(('ok', 7))
        self.assertEqual(optimizer._receive(Connection(), time.monotonic()+1),7)
        self.assertGreaterEqual(len(calls),3)
        self.assertEqual(set(calls),{owner})
