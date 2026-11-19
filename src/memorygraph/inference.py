"""Optional local Ollama text generation; never executes model output."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from experiments.config import MAX_OUTPUT_TOKENS, SEED


@dataclass
class ResponseMeasurement:
    text: str
    elapsed_seconds: float
    prompt_tokens: int | None
    output_tokens: int | None


def ask_ollama(
    prompt: str, model: str, endpoint: str = "http://127.0.0.1:11434/api/generate"
) -> ResponseMeasurement:
    if not model.strip():
        raise ValueError("A local Ollama model name is required")
    start = time.perf_counter()
    response = requests.post(
        endpoint,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "seed": SEED, "num_predict": MAX_OUTPUT_TOKENS},
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return ResponseMeasurement(
        text=str(payload.get("response", "")),
        elapsed_seconds=time.perf_counter() - start,
        prompt_tokens=payload.get("prompt_eval_count"),
        output_tokens=payload.get("eval_count"),
    )
