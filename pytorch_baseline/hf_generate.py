#!/usr/bin/env python3
"""Benchmark plain Hugging Face/PyTorch generate() across prompt and batch sizes."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from typing import Any

from common import (
    DEFAULT_MODEL_ID,
    DEFAULT_REVISION,
    batch_inputs,
    build_prompt,
    cuda_memory_stats,
    load_model_and_tokenizer,
    parse_int_list,
    prompt_fingerprint,
    reset_peak_memory,
    synchronize,
    write_json,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def generation_kwargs(
    tokenizer: Any, max_new_tokens: int, force_output_length: bool
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if force_output_length:
        kwargs["min_new_tokens"] = max_new_tokens
    return kwargs


def run_case(
    torch: Any,
    tokenizer: Any,
    model: Any,
    device: str,
    prompt_tokens: int,
    batch_size: int,
    max_new_tokens: int,
    warmups: int,
    repetitions: int,
    force_output_length: bool,
) -> dict[str, Any]:
    prompt_text, token_ids = build_prompt(tokenizer, prompt_tokens)
    inputs = batch_inputs(torch, token_ids, batch_size, device)
    kwargs = generation_kwargs(tokenizer, max_new_tokens, force_output_length)

    with torch.inference_mode():
        for _ in range(warmups):
            model.generate(**inputs, **kwargs)
    synchronize(torch, device)

    samples: list[dict[str, Any]] = []
    last_output = None
    for repetition in range(repetitions):
        reset_peak_memory(torch, device)
        synchronize(torch, device)
        started = time.perf_counter()
        with torch.inference_mode():
            last_output = model.generate(**inputs, **kwargs)
        synchronize(torch, device)
        latency_s = time.perf_counter() - started
        generated_per_sequence = int(last_output.shape[-1] - inputs["input_ids"].shape[-1])
        generated_tokens = generated_per_sequence * batch_size
        memory = cuda_memory_stats(torch, device)
        samples.append(
            {
                "repetition": repetition,
                "latency_s": latency_s,
                "generated_tokens": generated_tokens,
                "generated_tokens_per_sequence": generated_per_sequence,
                "output_tokens_per_sec": (
                    generated_tokens / latency_s if latency_s > 0 else None
                ),
                **memory,
            }
        )

    latencies = [float(sample["latency_s"]) for sample in samples]
    throughputs = [
        float(sample["output_tokens_per_sec"])
        for sample in samples
        if sample["output_tokens_per_sec"] is not None
    ]
    result = {
        "status": "completed",
        "batch_size": batch_size,
        "prompt_tokens_target": prompt_tokens,
        **prompt_fingerprint(prompt_text, token_ids),
        "max_new_tokens": max_new_tokens,
        "force_output_length": force_output_length,
        "latency_mean_s": statistics.fmean(latencies),
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p95_s": percentile(latencies, 0.95),
        "output_tokens_per_sec_mean": statistics.fmean(throughputs),
        "max_allocated_bytes_max": max(
            int(sample["max_allocated_bytes"] or 0) for sample in samples
        ),
        "max_reserved_bytes_max": max(
            int(sample["max_reserved_bytes"] or 0) for sample in samples
        ),
        "samples": samples,
    }
    del inputs, last_output
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


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
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--prompt-tokens", default="512,2048,8192")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--force-output-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", default="benchmarks/pytorch/hf_generate.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_sizes = parse_int_list(args.batch_sizes)
    prompt_targets = parse_int_list(args.prompt_tokens)
    if args.max_new_tokens < 1 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("token count/repetitions must be positive and warmups non-negative")
    plan = {
        "model": args.model,
        "revision": args.revision,
        "device": args.device,
        "dtype": args.dtype,
        "batch_sizes": batch_sizes,
        "prompt_tokens": prompt_targets,
        "max_new_tokens": args.max_new_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "force_output_length": args.force_output_length,
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
    payload = {
        **plan,
        "device_resolved": device,
        "torch_version": torch.__version__,
        "memory_after_model_load": cuda_memory_stats(torch, device),
        "prompt_manifest": [],
        "cases": [],
    }
    for prompt_tokens in prompt_targets:
        prompt_text, token_ids = build_prompt(tokenizer, prompt_tokens)
        payload["prompt_manifest"].append(
            {
                "prompt_tokens_target": prompt_tokens,
                "user_content": prompt_text,
                "rendered_chat_token_ids": token_ids,
                **prompt_fingerprint(prompt_text, token_ids),
            }
        )
    write_json(args.out, payload)
    for prompt_tokens in prompt_targets:
        for batch_size in batch_sizes:
            try:
                case = run_case(
                    torch,
                    tokenizer,
                    model,
                    device,
                    prompt_tokens,
                    batch_size,
                    args.max_new_tokens,
                    args.warmups,
                    args.repetitions,
                    args.force_output_length,
                )
            except torch.cuda.OutOfMemoryError as exc:
                case = {
                    "status": "cuda_oom",
                    "batch_size": batch_size,
                    "prompt_tokens_target": prompt_tokens,
                    "error": str(exc)[:2000],
                }
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                case = {
                    "status": "runtime_error",
                    "batch_size": batch_size,
                    "prompt_tokens_target": prompt_tokens,
                    "error": str(exc)[:2000],
                }
                gc.collect()
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            payload["cases"].append(case)
            write_json(args.out, payload)
    write_json(args.out, payload)
    print(
        json.dumps(
            {
                "wrote": args.out,
                "model": args.model,
                "dtype": args.dtype,
                "case_count": len(payload["cases"]),
                "failed_cases": sum(
                    1
                    for case in payload["cases"]
                    if case.get("status") != "completed"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
