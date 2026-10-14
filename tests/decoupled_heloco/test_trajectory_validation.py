import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "run_script"))
import evaluate_trajectories

class DummyPath:
    def __init__(self, name):
        self.name = name
    def __lt__(self, other):
        return self.name < other.name

class TestTrajectoryValidation(unittest.TestCase):
    @patch("evaluate_trajectories.load_run")
    @patch("evaluate_trajectories.check_compatible_runs")
    @patch("evaluate_trajectories.build_recipe")
    @patch("evaluate_trajectories.validate_tokenizer")
    @patch("evaluate_trajectories.tokenizer_fingerprint")
    @patch("evaluate_trajectories.build_initial_model")
    @patch("evaluate_trajectories.parameter_fingerprint")
    @patch("evaluate_trajectories.freeze_batches")
    @patch("evaluate_trajectories.load_validation_cache")
    @patch("torch.save")
    @patch("torch.load")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_missing_trajectory_dir(self, mock_loader, *args):
        mock_run = {"folder": MagicMock(), "config": MagicMock(), "options": MagicMock(), "summary": {"sync_updates": 20}}
        mock_run["folder"].name = "test_method"
        mock_run["folder"].__truediv__.return_value.is_dir.return_value = False
        evaluate_trajectories.load_run.return_value = mock_run
        
        with patch("torch.load") as mock_load:
            mock_load.return_value = {"batches_sha256": "fake"}
            with self.assertRaisesRegex(RuntimeError, "Missing trajectory directory for test_method"):
                evaluate_trajectories.run_trajectory_validation([Path("/tmp")], REPO_ROOT, batches=1, device="cuda:0", validation_cache_path="/tmp/cache.pt")

    @patch("evaluate_trajectories.load_run")
    @patch("evaluate_trajectories.check_compatible_runs")
    @patch("evaluate_trajectories.build_recipe")
    @patch("evaluate_trajectories.validate_tokenizer")
    @patch("evaluate_trajectories.tokenizer_fingerprint")
    @patch("evaluate_trajectories.build_initial_model")
    @patch("evaluate_trajectories.parameter_fingerprint")
    @patch("evaluate_trajectories.freeze_batches")
    @patch("evaluate_trajectories.load_validation_cache")
    @patch("torch.save")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_missing_initial_checkpoint(self, *args):
        mock_run = {"folder": MagicMock(), "config": MagicMock(), "options": MagicMock(), "summary": {"sync_updates": 20}}
        mock_run["folder"].name = "test_method"
        traj_dir = MagicMock()
        traj_dir.is_dir.return_value = True
        traj_dir.glob.return_value = [DummyPath("00000010.pt"), DummyPath("00000020-final.pt")]
        mock_run["folder"].__truediv__.return_value = traj_dir
        evaluate_trajectories.load_run.return_value = mock_run
        
        with patch("torch.load") as mock_load:
            mock_load.return_value = {"batches_sha256": "fake"}
            with self.assertRaisesRegex(RuntimeError, "Missing initial checkpoint for test_method"):
                evaluate_trajectories.run_trajectory_validation([Path("/tmp")], REPO_ROOT, batches=1, device="cuda:0", validation_cache_path="/tmp/cache.pt")
                
    @patch("evaluate_trajectories.load_run")
    @patch("evaluate_trajectories.check_compatible_runs")
    @patch("evaluate_trajectories.build_recipe")
    @patch("evaluate_trajectories.validate_tokenizer")
    @patch("evaluate_trajectories.tokenizer_fingerprint")
    @patch("evaluate_trajectories.build_initial_model")
    @patch("evaluate_trajectories.parameter_fingerprint")
    @patch("evaluate_trajectories.freeze_batches")
    @patch("evaluate_trajectories.load_validation_cache")
    @patch("torch.save")
    @patch("torch.cuda.set_device")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.device_count", return_value=1)
    def test_missing_intermediate_checkpoint(self, *args):
        mock_run = {"folder": MagicMock(), "config": MagicMock(), "options": MagicMock(), "summary": {"sync_updates": 20}}
        mock_run["folder"].name = "test_method"
        mock_run["config"].monitoring.get.return_value = 10
        traj_dir = MagicMock()
        traj_dir.is_dir.return_value = True
        traj_dir.glob.return_value = [DummyPath("00000000-initial.pt"), DummyPath("00000020-final.pt")]
        mock_run["folder"].__truediv__.return_value = traj_dir
        evaluate_trajectories.load_run.return_value = mock_run
        
        with patch("torch.load") as mock_load:
            def fake_load(path, **kwargs):
                if hasattr(path, "name"):
                    if "initial" in path.name: return {"step": 0}
                    if "final" in path.name: return {"step": 20}
                    return {"step": 10}
                return {"batches_sha256": "fake"}
            mock_load.side_effect = fake_load
            
            with self.assertRaisesRegex(RuntimeError, r"Missing expected intermediate checkpoints for test_method at steps: \[10\]"):
                evaluate_trajectories.run_trajectory_validation([Path("/tmp")], REPO_ROOT, batches=1, device="cuda:0", validation_cache_path="/tmp/cache.pt")

if __name__ == "__main__":
    unittest.main()
