"""Download only required files from the pinned embedding-model snapshot."""

from __future__ import annotations

import hashlib
import json

from huggingface_hub import snapshot_download

from experiments.config import EMBEDDING_MODEL, EMBEDDING_REVISION, MODEL_SNAPSHOT

FILES = [
    "config_sentence_transformers.json",
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "1_Pooling/config.json",
]


def main() -> None:
    snapshot_download(
        repo_id=EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        allow_patterns=FILES,
        local_dir=MODEL_SNAPSHOT,
    )
    weight = MODEL_SNAPSHOT / "model.safetensors"
    print(
        json.dumps(
            {
                "model": EMBEDDING_MODEL,
                "revision": EMBEDDING_REVISION,
                "weights_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
