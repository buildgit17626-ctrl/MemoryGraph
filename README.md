# MemoryGraph MVP

An evidence-aware, local demonstration of personalized memory retrieval for coding assistance. It uses fictional synthetic memories and does **not** claim real-user improvement or production readiness.

## Research question

Can selective retrieval of relevant user memories improve AI coding assistance while avoiding unnecessary context? This MVP implements retrieval and transparency components; it does not answer the question with real-user evidence.

## Setup

Python 3.12 is approved for this MVP. From this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
```

If `py -3.12` is unavailable, use a Python 3.12 executable to create the environment. Run commands from the project root with `PYTHONPATH=src` set (`$env:PYTHONPATH='src'` in PowerShell). Dependencies are pinned; keep the environment and local generated files out of version control.

## Run the experiments

```powershell
.\.venv\Scripts\python.exe -m experiments.run_baselines
.\.venv\Scripts\python.exe -m experiments.prepare_model
.\.venv\Scripts\python.exe -m experiments.run_semantic
.\.venv\Scripts\python.exe -m experiments.plot_results
```

The first script creates `data/generated/synthetic_relevance.csv` and writes grouped cross-validation metrics, per-pair predictions, and trained model artifacts in `results/`. The model preparation script downloads only files required from the pinned pretrained snapshot into local `work/`. The semantic script compares embeddings with TF-IDF on the same synthetic pairs, and the plot script charts those results. Generated files are never hand edited.

## ML pipeline

Each synthetic user has five fictional preferences and five matching query categories. Each query is paired with the user's five candidate memories, of which exactly one is labeled relevant. The templates are transparent in `src/memorygraph/data.py`; no real user data enters this dataset. Query and memory text are joined for TF-IDF features, followed by either balanced logistic regression or a balanced random forest in a scikit-learn Pipeline. `GroupKFold` splits by synthetic user ID so one persona cannot appear in both training and validation folds.

Precision, recall, and F1 describe binary relevance classification. Precision@3 and mean reciprocal rank describe ranked retrieval per query. TF-IDF cosine is a label-free lexical baseline. Pretrained embeddings test whether semantic similarity changes ranking without task-specific training. All values describe synthetic templates only; they are not estimates of real-user performance.

## Graph and retrieval

`src/memorygraph/retrieval.py` normalizes a small alias vocabulary (for example VS Code → `vscode` and pytest → `testing`) and creates a NetworkX bipartite graph from memories to canonical entities. Graph-assisted retrieval adds a bounded score bonus to lexical retrieval when the query and memory share an entity. This is a simple graph lookup, not GraphRAG.

## Local demo

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The demo lets a user add, edit, and delete local memories; load a fictional example; inspect every candidate score; compare TF-IDF, semantic, and graph-assisted retrieval; and choose the exact context. An explicit approval checkbox is required before optional Ollama inference. The app warns about explicit pytest/unittest or VS Code/PyCharm preference conflicts and blocks a call if both sides are selected. This narrow detector does not resolve arbitrary contradictions. The comparison runs the same question with and without approved memory. The user enters an already installed local model name; no generated code is executed. Latency is measured locally, and token counts are displayed only if Ollama supplies them. Correctness and energy are not automatically measured.

For a fixed two-prompt local inference demonstration, start Ollama with the pre-registered `qwen2.5-coder:0.5b` model installed, then run `python -m experiments.run_inference_demo`. Each run creates a unique raw JSONL file, preserving failed calls as well as successes. The two prompts differ in whether a fictional pytest preference is supplied. Output text, Ollama token counts, and wall-clock latency are logged; correctness is not scored and energy is not measured. One local run produced 46/128 prompt/output tokens without memory and 63/128 with memory. Despite the pytest preference, the memory-enabled response began with `import unittest`. These two capped generations do not establish an effect on quality or speed.

## Limitations and next steps

The dataset is small, templated, and synthetic. Grouped validation reduces persona leakage but does not remove template shortcuts. The graph uses a fixed alias vocabulary. Conflict detection covers only two explicit preference pairs. The local demo has no authentication or multi-user privacy boundary. Future work needs genuine consenting-user evaluation, stronger memory extraction and conflict handling, more robust retrieval, response scoring, and a documented energy method before any energy claim.

See `docs/SAFETY.md`, `docs/REFERENCES.md`, and `docs/learning/P2.md`.
