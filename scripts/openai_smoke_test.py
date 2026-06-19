#!/usr/bin/env python3
"""Send one OpenAI-compatible chat request to a running vLLM server."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
    model = os.environ.get("SERVED_MODEL_NAME") or os.environ.get("MODEL_ID", "Qwen/Qwen3.6-35B-A3B")
    api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    endpoint = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with one sentence explaining why TTFT matters in inference serving.",
            }
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        sys.stderr.write(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}\n")
        return 1
    except Exception as exc:  # noqa: BLE001 - command-line tool should print the failure.
        sys.stderr.write(f"Request failed: {exc}\n")
        return 1

    latency = time.perf_counter() - started
    message = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = body.get("usage", {})

    print(json.dumps(
        {
            "ok": True,
            "model": model,
            "latency_s": round(latency, 3),
            "usage": usage,
            "message": message,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
