#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible NIM endpoint with reproducible profiles.

The script does not change a NIM engine's precision: that is chosen by the
deployed NIM profile.  Instead it records the exact engine settings supplied
by the operator alongside latency, throughput, and time-to-first-token (TTFT).
Run one result per engine/profile, then compare the JSON files.

Examples:
  python scripts/benchmark_nim_serving.py --profile fp16-trtllm \
    --engine-settings '{"precision":"fp16","kv_cache":"managed"}'
  python scripts/benchmark_nim_serving.py --profile int8-trtllm \
    --engine-settings '{"precision":"int8","quantization":"weight-only"}'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = (
    "Classify this banking complaint and state the applicable regulatory "
    "category in one sentence: My credit report contains an account that "
    "does not belong to me, and the bank will not investigate."
)


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    # Nearest-rank percentile: p95 of three samples must include the slowest
    # sample rather than silently reporting the median.
    return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]


def run_once(
    client: httpx.Client, url: str, headers: dict[str, str], payload: dict
) -> tuple[float, float, int]:
    started = time.perf_counter()
    first_token_s: Optional[float] = None
    text = ""
    with client.stream("POST", url, headers=headers, json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            if first_token_s is None:
                first_token_s = time.perf_counter() - started
            try:
                chunk = json.loads(data)
                text += chunk["choices"][0].get("delta", {}).get("content", "")
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
    elapsed_s = time.perf_counter() - started
    return elapsed_s, first_token_s or elapsed_s, max(1, len(text.split()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    parser.add_argument("--model", default=os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct"))
    parser.add_argument("--api-key", default=os.getenv("NIM_API_KEY"))
    parser.add_argument("--profile", required=True, help="Human-readable deployed engine/profile name.")
    parser.add_argument("--engine-settings", default="{}", help="JSON metadata; recorded, never applied.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Concurrent requests per measured round.")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "optimization" / "results"))
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set NIM_API_KEY or provide --api-key; do not commit the key.")
    if args.runs < 1 or args.warmup < 0 or args.concurrency < 1:
        raise SystemExit("--runs and --concurrency must be positive; --warmup cannot be negative.")
    try:
        settings = json.loads(args.engine_settings)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--engine-settings must be JSON: {exc}") from exc

    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}", "Accept": "text/event-stream"}
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
    }
    timings: list[float] = []
    ttfts: list[float] = []
    output_tokens: list[int] = []
    def invoke() -> tuple[float, float, int]:
        with httpx.Client(timeout=120) as client:
            return run_once(client, url, headers, payload)

    for _ in range(args.warmup):
        invoke()
    for index in range(args.runs):
        round_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(invoke) for _ in range(args.concurrency)]
            results = [future.result() for future in as_completed(futures)]
        round_elapsed = time.perf_counter() - round_started
        for elapsed, ttft, tokens in results:
            timings.append(elapsed)
            ttfts.append(ttft)
            output_tokens.append(tokens)
        print(f"round {index + 1}/{args.runs}: concurrency={args.concurrency} "
              f"wall={round_elapsed:.3f}s, requests={len(results)}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "endpoint": args.base_url,
        "model": args.model,
        "engine_settings": settings,
        "workload": {
            "rounds": args.runs, "warmup": args.warmup,
            "concurrency": args.concurrency, "max_tokens": args.max_tokens,
        },
        "metrics": {
            "ttft_p50_s": round(statistics.median(ttfts), 4),
            "ttft_p95_s": round(percentile(ttfts, 0.95), 4),
            "latency_p50_s": round(statistics.median(timings), 4),
            "latency_p95_s": round(percentile(timings, 0.95), 4),
            "output_tokens_per_s": round(sum(output_tokens) / sum(timings), 3),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.profile}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
