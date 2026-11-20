# MemoryGraph local MVP specification

## Purpose

Demonstrate editable, selective memory retrieval for a local coding assistant. The public examples and evaluation data are fictional. The prototype does not establish real-user benefit.

## Implemented pipeline

Store plain-text memories in a local JSON file. Rank query-memory pairs with TF-IDF cosine, pretrained all-MiniLM-L6-v2 embeddings, or a simple NetworkX entity-bonus method. Show the candidate scores and exact selected context. A user may edit or delete memories and must approve the selected context before the optional local Ollama call. The app does not execute generated code.

Detect explicit conflicts in a documented preference vocabulary. Warn the user and block a model call when both sides of a detected conflict are selected. Do not silently resolve or delete a preference.

## Evaluation

Generate 24 fictional users, five query categories, and five candidate memories per query: 600 labeled pairs and 120 queries. Keep the seed and model revision in `experiments/config.py`. Group classifier validation by fictional user. Save raw predictions and generated metrics without editing them. Report retrieval precision@3 and MRR alongside classification precision, recall, and F1. The current results are synthetic demonstration results only.

The fixed local two-prompt generation run logs prompt, raw response, latency, and Ollama token counts. It does not score correctness or measure energy. The recorded pytest-memory call produced `unittest` code, so the public case study treats it as a failure example.

## Planned work

Memory extraction from conversation history, nuanced conflict resolution, feedback learning, selective replay and forgetting, consenting-user evaluation, correctness scoring, and documented energy measurement are outside this MVP.

## Safety

Model output remains plain text. No generated code is evaluated, imported, run, or installed. Local model requests target loopback. The memory file is local and git-ignored. See `docs/SAFETY.md`.
