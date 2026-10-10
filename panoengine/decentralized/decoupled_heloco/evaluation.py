"""Shared, token-weighted language-model evaluation on frozen CPU batches.

No TorchTitan, optimizer, network or distributed-runtime imports are needed
here. Labels from the text loader already target the next token; never shift
them a second time. The CUDA entry point supplies the real packed-document
FlexAttention model.
"""

from collections.abc import Mapping
import hashlib
import json
import math

import torch
import torch.nn.functional as F

from .fragment_manager import FragmentManager
from .baseline_options import BASELINE_FORMAT
from .config import DECOUPLED_METHODS, LEGACY_METHODS

IGNORE_INDEX = -100
CACHE_FORMAT = "decoupled_heloco_validation_v1"
GLOBAL_FORMAT = "decoupled_heloco_global_v1"


def _tensor_hash(digest, name, value):
    if not isinstance(value, torch.Tensor) or value.is_meta or value.layout != torch.strided:
        raise ValueError(f"{name}: fingerprint requires materialized dense tensors")
    value = value.detach().cpu().contiguous()
    metadata = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    # Byte view also supports BF16 and scalar tensors without numpy conversion
    # of unsupported source dtypes.
    data = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)


def parameter_fingerprint(parameters):
    digest = hashlib.sha256()
    for name, value in parameters.items():
        _tensor_hash(digest, name, value)
    return digest.hexdigest()


def validate_batch(batch, *, seq_len, batch_size, vocab_size):
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("a validation batch must contain (input_dict, labels)")
    inputs, labels = batch
    if not isinstance(inputs, dict) or set(inputs) != {"input", "positions"}:
        raise ValueError("validation inputs require exactly input and document positions")
    tensors = [inputs["input"], inputs["positions"], labels]
    for value in tensors:
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.long or value.layout != torch.strided:
            raise ValueError("frozen validation tensors must be dense CPU int64 tensors")
        if tuple(value.shape) != (batch_size, seq_len):
            raise ValueError("validation tensor shapes must match batch size and sequence length")
    tokens, positions = tensors[:2]
    if bool(((tokens < 0) | (tokens >= vocab_size)).any()):
        raise ValueError("validation input token ID exceeds model vocabulary")
    if bool(((positions < 0) | (positions >= seq_len)).any()):
        raise ValueError("validation document position exceeds the sequence length")
    valid = labels != IGNORE_INDEX
    if not bool(valid.any()):
        raise ValueError("validation batch has no valid target tokens")
    if bool(((labels[valid] < 0) | (labels[valid] >= vocab_size)).any()):
        raise ValueError("validation target token ID exceeds model vocabulary")


def freeze_batches(loader, count, *, seq_len, batch_size, vocab_size):
    if type(count) is not int or not 1 <= count <= 10000:
        raise ValueError("validation batches must be an integer in 1..10000")
    iterator = iter(loader)
    batches = []
    for _ in range(count):
        try:
            inputs, labels = next(iterator)
        except StopIteration as exc:
            raise ValueError(f"validation data ended after {len(batches)} batches; requested {count}") from exc
        validate_batch((inputs, labels), seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size)
        batches.append(({key: value.clone() for key, value in inputs.items()}, labels.clone()))
    return batches


def batch_fingerprint(batches):
    digest = hashlib.sha256()
    for index, (inputs, labels) in enumerate(batches):
        for name in ("input", "positions"):
            _tensor_hash(digest, f"{index}/{name}", inputs[name])
        _tensor_hash(digest, f"{index}/labels", labels)
    return digest.hexdigest()


def validation_cache(batches, metadata):
    return {"format": CACHE_FORMAT, "metadata": dict(metadata), "batches": batches,
            "batches_sha256": batch_fingerprint(batches)}


def load_validation_cache(payload, expected_metadata, *, vocab_size):
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        raise ValueError("unrecognized validation cache format")
    if payload.get("metadata") != expected_metadata:
        raise ValueError("validation cache tokenizer, split, batch size, sequence length or batch count differs")
    batches = payload.get("batches")
    if not isinstance(batches, (list, tuple)) or len(batches) != expected_metadata["batch_count"]:
        raise ValueError("validation cache batch count differs")
    for batch in batches:
        validate_batch(batch, seq_len=expected_metadata["seq_len"], batch_size=expected_metadata["batch_size"], vocab_size=vocab_size)
    if batch_fingerprint(batches) != payload.get("batches_sha256"):
        raise ValueError("validation cache tensor fingerprint differs")
    return batches


def load_global_parameters(model, payload, *, num_fragments, expected_revisions=None, expected_method=None):
    """Validate the complete exported global checkpoint before replacing weights."""
    if not isinstance(payload, dict) or payload.get("format") not in {GLOBAL_FORMAT, BASELINE_FORMAT}:
        raise ValueError("expected a supported named-parameter global export")
    method = payload.get("method", "decoupled_heloco")
    allowed = LEGACY_METHODS if payload["format"] == BASELINE_FORMAT else DECOUPLED_METHODS
    if method not in allowed or (expected_method is not None and method != expected_method):
        raise ValueError("checkpoint method differs from the saved experiment")
    if payload["format"] == BASELINE_FORMAT and num_fragments != 1:
        raise ValueError("matched baseline export must contain the whole model as one fragment")
    manager = FragmentManager.from_model(model, num_fragments)
    if payload.get("layout_signature") != manager.layout_signature:
        raise ValueError("checkpoint layout signature does not match the evaluation model")
    revisions = payload.get("fragment_revisions")
    if not isinstance(revisions, (tuple, list)) or len(revisions) != num_fragments or any(type(r) is not int or r < 1 for r in revisions):
        raise ValueError("checkpoint fragment revisions are invalid")
    if expected_revisions is not None and list(revisions) != list(expected_revisions):
        raise ValueError("checkpoint revisions differ from the run summary")
    values = payload.get("parameters")
    if not isinstance(values, Mapping):
        raise ValueError("checkpoint parameters must be a mapping")
    for name, value in values.items():
        if not isinstance(value, torch.Tensor) or hasattr(value, "device_mesh") or value.device.type != "cpu" or value.dtype != torch.float32 or value.layout != torch.strided or not bool(torch.isfinite(value).all()):
            raise ValueError(f"checkpoint parameter {name} requires finite dense CPU FP32 values")
    manager.validate_model(values)
    live = dict(model.named_parameters())
    # Stage every allocation/conversion before changing any live parameter.
    staged = {name: value.to(device=live[name].device, dtype=live[name].dtype, copy=True) for name, value in values.items()}
    with torch.no_grad():
        for name, value in staged.items():
            live[name].copy_(value)


def evaluate_model(model, batches, *, device, vocab_size, progress=None):
    """Return mean CE, exp(mean CE), and top-1 next-token accuracy.

    Sum losses and correct predictions over nonignored tokens. Do not average
    per-batch means: a padded batch has fewer valid tokens. No backward or
    optimizer operation runs, and all module training flags are restored.
    """
    if not batches:
        raise ValueError("validation needs at least one batch")
    modes = [(module, module.training) for module in model.modules()]
    loss_sums, total_tokens, total_correct = [], 0, 0
    model.eval()
    try:
        with torch.no_grad():
            for index, (inputs, cpu_labels) in enumerate(batches):
                seq_len = cpu_labels.shape[-1]
                validate_batch((inputs, cpu_labels), seq_len=seq_len, batch_size=cpu_labels.shape[0], vocab_size=vocab_size)
                tokens = inputs["input"].to(device)
                positions = inputs["positions"].to(device)
                labels = cpu_labels.to(device)
                kwargs = {"positions": positions}
                mask_builder = getattr(model, "get_attention_masks", None)
                if callable(mask_builder):
                    kwargs["attention_masks"] = mask_builder(positions=positions)
                logits = model(tokens, **kwargs)
                if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (*labels.shape, vocab_size):
                    raise ValueError("model must return dense [batch, sequence, vocabulary] logits")
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("validation model returned nonfinite logits")
                valid = labels != IGNORE_INDEX
                per_token = F.cross_entropy(logits.float().reshape(-1, vocab_size), labels.reshape(-1),
                                            reduction="none", ignore_index=IGNORE_INDEX)
                loss_sum = float(per_token[valid.reshape(-1)].double().sum())
                if not math.isfinite(loss_sum):
                    raise ValueError("validation loss is nonfinite")
                loss_sums.append(loss_sum)
                total_tokens += int(valid.sum())
                total_correct += int(((logits.argmax(dim=-1) == labels) & valid).sum())
                if progress is not None:
                    progress(index + 1, len(batches))
    finally:
        for module, training in modes:
            module.training = training
    loss = math.fsum(loss_sums) / total_tokens
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = None
    return {"loss": loss, "perplexity": perplexity, "perplexity_overflow": perplexity is None,
            "next_token_accuracy": total_correct / total_tokens, "correct_tokens": total_correct,
            "valid_tokens": total_tokens, "batches": len(batches)}
