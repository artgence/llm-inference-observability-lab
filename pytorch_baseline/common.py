#!/usr/bin/env python3
"""Shared utilities for the Hugging Face/PyTorch inference baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = os.environ.get(
    "PYTORCH_MODEL_ID",
    os.environ.get("MODEL_ID", "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8"),
)
DEFAULT_REVISION = os.environ.get("MODEL_REVISION", "main")
PROMPT_SEED = (
    "capacity latency throughput queueing prefill decode memory tokens batching "
    "telemetry timeout saturation utilization cache request service "
)


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise ValueError("expected a comma-separated list of positive integers")
    return values


def load_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch baseline dependencies are missing. Install torch, transformers, "
            "accelerate, and the model's quantization package when required."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        raise RuntimeError(
            f"device {requested!r} requested but CUDA is unavailable. "
            f"This PyTorch build targets CUDA {torch_cuda or 'unknown'}; verify that the "
            "host NVIDIA driver supports it, or install the PyTorch build selected for "
            "the existing vLLM environment. Do not continue a GPU benchmark on CPU."
        )
    return requested


def resolve_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "auto": "auto",
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    return mapping[name]


def synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def cuda_memory_stats(torch: Any, device: str) -> dict[str, int | None]:
    if not device.startswith("cuda"):
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "max_allocated_bytes": None,
            "max_reserved_bytes": None,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def reset_peak_memory(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)


def tensor_bytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def cache_bytes(cache: Any) -> int:
    if cache is None:
        return 0
    if hasattr(cache, "to_legacy_cache"):
        try:
            return tensor_bytes(cache.to_legacy_cache())
        except Exception:
            pass
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return tensor_bytes(cache.key_cache) + tensor_bytes(cache.value_cache)
    if hasattr(cache, "layers"):
        total = 0
        for layer in cache.layers:
            total += tensor_bytes(getattr(layer, "keys", None))
            total += tensor_bytes(getattr(layer, "values", None))
        if total:
            return total
    return tensor_bytes(cache)


def model_weight_bytes(model: Any) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        pointer = int(tensor.data_ptr()) if hasattr(tensor, "data_ptr") else id(tensor)
        if pointer in seen:
            continue
        seen.add(pointer)
        total += tensor_bytes(tensor)
    return total


def normalize_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer output for one prompt across Transformers versions."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("tokenizer result does not contain input_ids")
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(
        value[0], (list, tuple)
    ):
        value = list(value[0])
    if not isinstance(value, list):
        raise TypeError(
            f"unsupported tokenizer result type: {type(value).__name__}"
        )
    if any(isinstance(token_id, (list, tuple, Mapping)) for token_id in value):
        raise ValueError("expected token IDs for exactly one prompt")
    return [int(token_id) for token_id in value]


def render_chat_ids(tokenizer: Any, content: str) -> list[int]:
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        ids = tokenizer.encode(content, add_special_tokens=True)
    return normalize_token_ids(ids)


def exact_length_text(seed: str, target_chars: int) -> str:
    return (seed * math.ceil(target_chars / len(seed)))[:target_chars]


def build_prompt(tokenizer: Any, prompt_tokens: int) -> tuple[str, list[int]]:
    """Find deterministic user text whose rendered chat prompt is near the token target."""
    if prompt_tokens < 8:
        raise ValueError("prompt_tokens must be >= 8 to allow chat-template overhead")
    low = 0
    high = max(128, prompt_tokens * 12)
    best: tuple[int, str, list[int]] | None = None
    for _ in range(32):
        midpoint = (low + high) // 2
        text = exact_length_text(PROMPT_SEED, midpoint)
        ids = render_chat_ids(tokenizer, text)
        distance = abs(len(ids) - prompt_tokens)
        if best is None or distance < best[0]:
            best = (distance, text, ids)
        if len(ids) == prompt_tokens:
            return text, ids
        if len(ids) < prompt_tokens:
            low = midpoint + 1
        else:
            high = midpoint - 1
        if low > high:
            break
    assert best is not None
    return best[1], best[2]


def batch_inputs(torch: Any, token_ids: list[int], batch_size: int, device: str) -> dict[str, Any]:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device).repeat(
        batch_size, 1
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def load_model_and_tokenizer(
    model_id: str,
    revision: str,
    dtype_name: str,
    device_name: str,
    trust_remote_code: bool = False,
) -> tuple[Any, Any, Any, str]:
    torch, AutoModelForCausalLM, AutoTokenizer = load_dependencies()
    device = resolve_device(torch, device_name)
    if device == "cpu" and "fp8" in model_id.lower():
        raise RuntimeError(
            f"{model_id!r} is an FP8 checkpoint, but the requested device resolved to "
            "CPU. Fix CUDA availability and rerun with --device cuda. Loading this "
            "checkpoint on CPU may disable compressed execution and initialize missing "
            "quantization parameters, which invalidates the benchmark."
        )
    dtype = resolve_dtype(torch, dtype_name)
    token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model_kwargs: dict[str, Any] = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        model_kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    try:
        model.to(device)
    except Exception as exc:
        raise RuntimeError(
            "The checkpoint could not be moved with plain Transformers. If it uses a "
            "specialized quantization format, install its runtime integration or select "
            "one Transformers-compatible model revision and use it for both PyTorch and vLLM."
        ) from exc
    model.eval()
    return torch, tokenizer, model, device


def prompt_fingerprint(text: str, token_ids: list[int]) -> dict[str, Any]:
    digest = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "prompt_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": digest,
        "prompt_tokens_actual": len(token_ids),
    }


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
