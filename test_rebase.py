import sys
import torch
from panoengine.decentralized.decoupled_heloco.state import FragmentState
from panoengine.decentralized.decoupled_heloco.fragment_manager import FragmentManager

class DummyManager:
    def __init__(self):
        self.layout_signature = "dummy"
    def select(self, fragment_id, parameters):
        return parameters
    def validate_update(self, fragment_id, parameters):
        pass

def test_rebase_and_ledger_rejected():
    manager = DummyManager()
    params = {"w": torch.tensor([1.0, 2.0])}
    state = FragmentState(manager, 0, params, rebase=True, ledger=True)
    
    params["w"] += 3.0
    state.record_step(10)
    
    snap = state.snapshot(params, request_id="1")
    
    state.queue_update({"w": torch.tensor([10.0, 11.0])}, 1, layout_signature="dummy", acked_request_id=None)
    state.apply_pending(params)
    
    # Rebase rejected: staged = local + update - baseline = [4,5] + [10,11] - [1,2] = [13.0, 14.0]
    assert params["w"].tolist() == [13.0, 14.0]
    assert state.metadata().local_steps == 1

test_rebase_and_ledger_rejected()
