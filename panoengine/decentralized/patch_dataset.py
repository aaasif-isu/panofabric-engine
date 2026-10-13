import os
import functools

def patch_torchtitan_c4():
    """Patches TorchTitan's C4 dataloader to support non-IID language partitions and offline caching."""
    try:
        import torchtitan.hf_datasets.text_datasets as text_datasets
        from datasets import load_dataset
        
        def _load_c4_dataset_patched(dataset_path: str, split: str):
            if "OFFLINE_TRAIN_CACHE_DIR" in os.environ and split == "train":
                raise RuntimeError("Hugging Face load_dataset was called but OFFLINE_TRAIN_CACHE_DIR is set. This means the intercept failed!")
            lang = os.environ.get("ISLAND_LANGUAGE", "en")
            return load_dataset(dataset_path, name=lang, split=split, streaming=True)
        
        if "c4" in text_datasets.DATASETS:
            text_datasets.DATASETS["c4"].loader = functools.partial(_load_c4_dataset_patched, split="train")
        if "c4_validation" in text_datasets.DATASETS:
            text_datasets.DATASETS["c4_validation"].loader = functools.partial(_load_c4_dataset_patched, split="validation")
            
        from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
        original_build = HuggingFaceTextDataLoader.Config.build
        
        def patched_build(self, *args, **kwargs):
            if "OFFLINE_TRAIN_CACHE_DIR" in os.environ and self.dataset == "c4":
                import torch
                from pathlib import Path
                dp_rank = kwargs.get("dp_rank", 0)
                if len(args) > 1:
                    dp_rank = args[1]
                learner_id = int(os.environ.get("LEARNER_ID", dp_rank))
                cache_path = Path(os.environ["OFFLINE_TRAIN_CACHE_DIR"]) / f"train_cache_learner_{learner_id}.pt"
                if not cache_path.is_file():
                    raise RuntimeError(f"Offline cache not found: {cache_path}")
                batches = torch.load(cache_path, map_location="cpu", weights_only=False)
                
                class OfflineLoader:
                    def __init__(self, batches):
                        self.batches = batches
                        
                    def __iter__(self):
                        for batch in self.batches:
                            yield batch
                        raise RuntimeError("Offline training data exhausted explicitly")
                        
                return OfflineLoader(batches)
                
            return original_build(self, *args, **kwargs)
            
        HuggingFaceTextDataLoader.Config.build = patched_build

    except ImportError:
        pass
