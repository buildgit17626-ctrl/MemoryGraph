# Models, libraries, and licenses

- [scikit-learn](https://scikit-learn.org/stable/) — TF-IDF, classifiers, Pipeline, GroupKFold, and metrics; BSD-3-Clause.
- [NumPy](https://numpy.org/) — numeric arrays and deterministic random generation; BSD-3-Clause.
- [Pandas](https://pandas.pydata.org/) — synthetic tabular dataset and CSV I/O; BSD-3-Clause.
- [NetworkX](https://networkx.org/) — simple memory-to-entity graph; BSD-3-Clause.
- [Sentence Transformers](https://www.sbert.net/) — pretrained embedding inference; Apache-2.0.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — intended embedding model; model card lists Apache-2.0. Verify the exact downloaded snapshot and license before public redistribution.
- [Streamlit](https://streamlit.io/) — local demo; Apache-2.0.
- [Ollama generate API](https://docs.ollama.com/api/generate) and [Modelfile settings](https://docs.ollama.com/modelfile) — local inference and token-count fields. The fixed demo uses the locally installed `qwen2.5-coder:0.5b` blob recorded in `experiments/config.py`; its local Modelfile includes an Apache-2.0 license. The interactive demo lets the user choose another installed model, whose license should be checked separately.
- [CodeCarbon](https://codecarbon.io/) — installed for future energy estimates but not used for MVP energy claims. No energy metric is reported.

Package versions are pinned in `requirements.txt`. The synthetic dataset is generated in this project and is not a published benchmark.
