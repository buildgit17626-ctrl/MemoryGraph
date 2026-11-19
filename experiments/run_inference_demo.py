"""Log two local Ollama responses without scoring or executing generated code."""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from experiments.config import (
    DEMO_MEMORY,
    DEMO_QUERY,
    INFERENCE_MODEL,
    INFERENCE_MODEL_BLOB_SHA256,
    MAX_OUTPUT_TOKENS,
    RESULTS_DIR,
    SEED,
)
from memorygraph.inference import ask_ollama


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    variants = [
        ("no_memory", DEMO_QUERY),
        ("with_memory", f"Relevant user memories:\n- {DEMO_MEMORY}\n\nUser request:\n{DEMO_QUERY}"),
    ]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = RESULTS_DIR / f"inference_demo_{run_id}.jsonl"
    failures = 0
    # A unique file preserves every raw attempt, including failed calls.
    with path.open("x", encoding="utf-8") as handle:
        for variant, prompt in variants:
            record = {
                "variant": variant,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": INFERENCE_MODEL,
                "model_blob_sha256": INFERENCE_MODEL_BLOB_SHA256,
                "seed": SEED,
                "temperature": 0,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "os": platform.platform(),
                "cpu": platform.processor(),
                "energy": "not measured",
                "correctness": "not scored",
            }
            try:
                result = ask_ollama(prompt, INFERENCE_MODEL)
                record.update({"status": "ok", "response": result.text,
                               "latency_seconds": result.elapsed_seconds,
                               "prompt_tokens": result.prompt_tokens,
                               "output_tokens": result.output_tokens})
            except Exception as error:
                failures += 1
                record.update({"status": "failed", "reason": f"{type(error).__name__}: {error}"})
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(variants)} raw generation records to {path}")
    if failures:
        raise RuntimeError(f"{failures} inference call(s) failed; see raw records")


if __name__ == "__main__":
    main()
