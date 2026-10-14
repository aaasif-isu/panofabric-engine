import torch
import unittest
from panoengine.decentralized.decoupled_heloco.state import FragmentState

class DummyManager:
    def __init__(self):
        self.layout_signature = "dummy"
    def select(self, fragment_id, parameters):
        return parameters
    def validate_update(self, fragment_id, parameters):
        pass

class TestRebaseLedger(unittest.TestCase):
    def test_original_behavior(self):
        manager = DummyManager()
        params = {"w": torch.tensor([1.0, 2.0])}
        state = FragmentState(manager, 0, params, rebase=False, ledger=False)
        
        state.record_step(10)
        self.assertEqual(state.metadata().local_steps, 1)
        
        snap = state.snapshot(params, request_id="1")
        state.queue_update({"w": torch.tensor([5.0, 6.0])}, 1, layout_signature="dummy", acked_request_id="1")
        
        params["w"] += 10.0
        state.record_step(20)
        self.assertEqual(state.metadata().local_steps, 2)
        
        state.apply_pending(params)
        self.assertEqual(params["w"].tolist(), [5.0, 6.0])
        self.assertEqual(state.metadata().local_steps, 0)
        self.assertEqual(state.metadata().tokens, 0)

    def test_rebase_only_rejected(self):
        manager = DummyManager()
        params = {"w": torch.tensor([1.0, 2.0])}
        state = FragmentState(manager, 0, params, rebase=True, ledger=False)
        
        state.record_step(10)
        params["w"] += 3.0
        
        snap = state.snapshot(params, request_id="1")
        state.queue_update({"w": torch.tensor([5.0, 6.0])}, 1, layout_signature="dummy", acked_request_id=None)
        
        state.apply_pending(params)
        self.assertEqual(params["w"].tolist(), [8.0, 9.0])
        self.assertEqual(state.metadata().local_steps, 0)

    def test_rebase_only_accepted(self):
        manager = DummyManager()
        params = {"w": torch.tensor([1.0, 2.0])}
        state = FragmentState(manager, 0, params, rebase=True, ledger=False)
        
        params["w"] += 3.0
        state.record_step(10)
        snap = state.snapshot(params, request_id="1")
        
        params["w"] += 2.0
        state.record_step(20)
        
        state.queue_update({"w": torch.tensor([10.0, 11.0])}, 1, layout_signature="dummy", acked_request_id="1")
        state.apply_pending(params)
        
        self.assertEqual(params["w"].tolist(), [12.0, 13.0])
        self.assertEqual(state.metadata().local_steps, 0)
        
    def test_rebase_and_ledger_accepted(self):
        manager = DummyManager()
        params = {"w": torch.tensor([1.0, 2.0])}
        state = FragmentState(manager, 0, params, rebase=True, ledger=True)
        
        params["w"] += 3.0
        state.record_step(10)
        state.record_step(10)
        
        snap = state.snapshot(params, request_id="1")
        
        params["w"] += 2.0
        state.record_step(20)
        
        state.queue_update({"w": torch.tensor([10.0, 11.0])}, 1, layout_signature="dummy", acked_request_id="1")
        state.apply_pending(params)
        
        self.assertEqual(params["w"].tolist(), [12.0, 13.0])
        self.assertEqual(state.metadata().local_steps, 1)
        self.assertEqual(state.metadata().tokens, 20)

    def test_rebase_and_ledger_rejected(self):
        manager = DummyManager()
        params = {"w": torch.tensor([1.0, 2.0])}
        state = FragmentState(manager, 0, params, rebase=True, ledger=True)
        
        params["w"] += 3.0
        state.record_step(10)
        
        snap = state.snapshot(params, request_id="1")
        
        state.queue_update({"w": torch.tensor([10.0, 11.0])}, 1, layout_signature="dummy", acked_request_id=None)
        state.apply_pending(params)
        
        self.assertEqual(params["w"].tolist(), [13.0, 14.0])
        self.assertEqual(state.metadata().local_steps, 1)

    def test_superseded_request(self):
        manager = DummyManager()
        params = {"w": torch.tensor([1.0, 2.0])}
        state = FragmentState(manager, 0, params, rebase=True, ledger=True)
        
        params["w"] += 3.0
        state.record_step(10)
        
        snap = state.snapshot(params, request_id="1")
        # Overwrite with delayed duplicate/retry request
        snap2 = state.snapshot(params, request_id="2")
        
        # Ack comes for "2", because "1" was aborted.
        state.queue_update({"w": torch.tensor([5.0, 6.0])}, 1, layout_signature="dummy", acked_request_id="2")
        state.apply_pending(params)
        
        self.assertEqual(state.metadata().local_steps, 0)
        # 1.0, 2.0 -> 4.0, 5.0 (displ=3)
        # staged = [4.0, 5.0] + [5.0, 6.0] - [1.0, 2.0] + [-3.0, -3.0] = [5.0, 6.0]
        self.assertEqual(params["w"].tolist(), [5.0, 6.0])

if __name__ == '__main__':
    unittest.main()
