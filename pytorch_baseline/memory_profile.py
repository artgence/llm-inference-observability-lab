#!/usr/bin/env python3
"""Profile PyTorch prefill/decode timing, CUDA memory, KV cache, and operators."""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    batch_inputs,
    build_prompt,
    cache_bytes,
    cuda_memory_stats,
    load_dependencies,
    load_model_and_tokenizer,
    model_weight_bytes,
    prompt_fingerprint,
    reset_peak_memory,
    resolve_device,
    synchronize,
    write_json,
)


def profiler_context(torch: Any, device: str, enabled: bool) -> Any:
    if not enabled:
        return contextlib.nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    )


def snapshot(label: str, torch: Any, device: str) -> dict[str, Any]:
    return {"phase": label, **cuda_memory_stats(torch, device)}


def enable_memory_history(torch: Any, device: str, output: str | None) -> str | None:
    if not output or not device.startswith("cuda"):
        return None
    try:
        torch.cuda.memory._record_memory_history(  # noqa: SLF001 - diagnostic API.
            enabled="all",
            context="all",
            stacks="all",
            max_entries=100000,
        )
    except Exception as exc:
        return f"could not enable CUDA memory history: {exc}"
    return None


def dump_memory_snapshot(torch: Any, output: str | None) -> str | None:
    if not output:
        return None
    try:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(path))  # noqa: SLF001 - diagnostic API.
        torch.cuda.memory._record_memory_history(enabled=None)  # noqa: SLF001
    except Exception as exc:
        return f"could not write CUDA memory snapshot: {exc}"
    return None


def run_prefill_decode(
    torch: Any,
    model: Any,
    inputs: dict[str, Any],
    new_tokens: int,
    device: str,
    profile: bool,
) -> tuple[dict[str, Any], Any]:
    memory = [snapshot("after_inputs", torch, device)]
    profile_object = None
    with profiler_context(torch, device, profile) as prof:
        profile_object = prof
        reset_peak_memory(torch, device)
        synchronize(torch, device)
        prefill_started = time.perf_counter()
        with torch.inference_mode(), torch.profiler.record_function("prefill"):
            outputs = model(**inputs, use_cache=True)
        synchronize(torch, device)
        prefill_s = time.perf_counter() - prefill_started
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        prefill_cache_bytes = cache_bytes(past_key_values)
        memory.append(snapshot("after_prefill", torch, device))

        reset_peak_memory(torch, device)
        synchronize(torch, device)
        decode_started = time.perf_counter()
        attention_mask = inputs["attention_mask"]
        decode_forward_steps = max(0, new_tokens - 1)
        with torch.inference_mode(), torch.profiler.record_function("decode"):
            for _ in range(decode_forward_steps):
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token)], dim=-1
                )
                outputs = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        synchronize(torch, device)
        decode_s = time.perf_counter() - decode_started
        decode_cache_bytes = cache_bytes(past_key_values)
        memory.append(snapshot("after_decode", torch, device))

    result = {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "generated_tokens": new_tokens * int(inputs["input_ids"].shape[0]),
        "decode_forward_steps": decode_forward_steps,
        "decode_tokens_per_sec": (
            decode_forward_steps * int(inputs["input_ids"].shape[0]) / decode_s
            if decode_s > 0
            else None
        ),
        "output_tokens_per_sec_end_to_end": (
            new_tokens
            * int(inputs["input_ids"].shape[0])
            / (prefill_s + decode_s)
            if prefill_s + decode_s > 0
            else None
        ),
        "kv_cache_bytes_after_prefill": prefill_cache_bytes,
        "kv_cache_bytes_after_decode": decode_cache_bytes,
        "memory": memory,
    }
    return result, profile_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default="bf16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-out", default="benchmarks/pytorch/inference_trace.json")
    parser.add_argument(
        "--operator-table-out",
        default="benchmarks/pytorch/operator_table.txt",
    )
    parser.add_argument("--memory-snapshot")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", default="benchmarks/pytorch/memory_profile.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.prompt_tokens < 8 or args.new_tokens < 1:
        raise ValueError("batch, prompt, and output lengths must be positive")
    plan = {
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "profile": args.profile,
        "memory_snapshot": args.memory_snapshot,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    torch, _, _ = load_dependencies()
    resolved_device = resolve_device(torch, args.device)
    before_load = snapshot("before_model_load", torch, resolved_device)
    torch, tokenizer, model, device = load_model_and_tokenizer(
        args.model,
        args.revision,
        args.dtype,
        args.device,
        args.trust_remote_code,
    )
    after_load = snapshot("after_model_load", torch, device)
    prompt_text, token_ids = build_prompt(tokenizer, args.prompt_tokens)
    inputs = batch_inputs(torch, token_ids, args.batch_size, device)
    history_error = enable_memory_history(torch, device, args.memory_snapshot)
    phase_result, prof = run_prefill_decode(
        torch,
        model,
        inputs,
        args.new_tokens,
        device,
        args.profile,
    )
    snapshot_error = dump_memory_snapshot(torch, args.memory_snapshot)
    weights = model_weight_bytes(model)
    peak = max(
        int(item.get("max_allocated_bytes") or 0)
        for item in phase_result["memory"]
    )
    kv_peak = int(phase_result["kv_cache_bytes_after_decode"])
    payload = {
        **plan,
        "device_resolved": device,
        "torch_version": torch.__version__,
        **prompt_fingerprint(prompt_text, token_ids),
        "model_weight_bytes": weights,
        "estimated_non_weight_non_kv_peak_bytes": max(0, peak - weights - kv_peak),
        "memory": [before_load, after_load, *phase_result.pop("memory")],
        **phase_result,
        "memory_history_error": history_error,
        "memory_snapshot_error": snapshot_error,
    }
    if prof is not None:
        trace_path = Path(args.trace_out)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_path))
        table = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total" if device.startswith("cuda") else "self_cpu_time_total",
            row_limit=100,
        )
        table_path = Path(args.operator_table_out)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.write_text(table, encoding="utf-8")
        payload["trace_out"] = str(trace_path)
        payload["operator_table_out"] = str(table_path)
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
