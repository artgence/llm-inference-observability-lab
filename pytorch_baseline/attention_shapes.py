#!/usr/bin/env python3
"""Inspect bounded attention input/output tensor shapes for one PyTorch prefill."""

from __future__ import annotations

import argparse
import json
from typing import Any

from common import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    batch_inputs,
    build_prompt,
    load_model_and_tokenizer,
    prompt_fingerprint,
    synchronize,
    write_json,
)


def tensor_metadata(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return "<nested>"
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [tensor_metadata(item, depth + 1) for item in value[:4]]
    if isinstance(value, dict):
        return {
            str(key): tensor_metadata(item, depth + 1)
            for key, item in list(value.items())[:8]
        }
    return type(value).__name__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="bf16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-modules", type=int, default=2)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", default="benchmarks/pytorch/attention_shapes.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.prompt_tokens < 8 or args.max_modules < 1:
        raise ValueError("batch, prompt target, and max modules must be positive")
    plan = {
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "max_modules": args.max_modules,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    torch, tokenizer, model, device = load_model_and_tokenizer(
        args.model,
        args.revision,
        args.dtype,
        args.device,
        args.trust_remote_code,
    )
    prompt_text, token_ids = build_prompt(tokenizer, args.prompt_tokens)
    inputs = batch_inputs(torch, token_ids, args.batch_size, device)
    records: list[dict[str, Any]] = []
    handles = []

    def hook_for(module_name: str) -> Any:
        def hook(module: Any, module_inputs: Any, module_output: Any) -> None:
            records.append(
                {
                    "module": module_name,
                    "class": type(module).__name__,
                    "inputs": tensor_metadata(module_inputs),
                    "output": tensor_metadata(module_output),
                    "num_heads": getattr(module, "num_heads", None),
                    "num_key_value_heads": getattr(module, "num_key_value_heads", None),
                    "head_dim": getattr(module, "head_dim", None),
                }
            )

        return hook

    for name, module in model.named_modules():
        if "attention" not in type(module).__name__.lower():
            continue
        handles.append(module.register_forward_hook(hook_for(name)))
        if len(handles) >= args.max_modules:
            break
    if not handles:
        raise RuntimeError("no attention modules were found by class name")
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=True)
        synchronize(torch, device)
    finally:
        for handle in handles:
            handle.remove()

    config = model.config
    payload = {
        **plan,
        "device_resolved": device,
        **prompt_fingerprint(prompt_text, token_ids),
        "model_config": {
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
            "head_dim": getattr(config, "head_dim", None),
        },
        "attention_modules": records,
        "note": "Only shape/dtype/device metadata is retained; activation tensors are not stored.",
    }
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
