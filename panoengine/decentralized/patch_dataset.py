import os
import functools

def patch_torchtitan_c4():
    """Patches TorchTitan's C4 dataloader to support non-IID language partitions."""
    try:
        import torchtitan.hf_datasets.text_datasets as text_datasets
        from datasets import load_dataset
        
        def _load_c4_dataset_patched(dataset_path: str, split: str):
            lang = os.environ.get("ISLAND_LANGUAGE", "en")
            return load_dataset(dataset_path, name=lang, split=split, streaming=True)
        
        if "c4" in text_datasets.DATASETS:
            text_datasets.DATASETS["c4"].loader = functools.partial(_load_c4_dataset_patched, split="train")
        if "c4_test" in text_datasets.DATASETS:
            # c4_test is local text, no language support needed usually, but we patch it just in case
            pass
        if "c4_validation" in text_datasets.DATASETS:
            text_datasets.DATASETS["c4_validation"].loader = functools.partial(_load_c4_dataset_patched, split="validation")
    except ImportError:
        pass
