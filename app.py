"""Transparent local Streamlit demo for editable coding-assistant memories."""

from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer

from experiments.config import MODEL_SNAPSHOT, ROOT
from memorygraph.inference import ask_ollama
from memorygraph.retrieval import rank_memories
from memorygraph.store import MemoryStore

st.set_page_config(page_title="MemoryGraph MVP", layout="wide")
store = MemoryStore(ROOT / "data" / "generated" / "demo_memories.json")


@st.cache_resource
def load_encoder() -> SentenceTransformer:
    if not (MODEL_SNAPSHOT / "model.safetensors").exists():
        raise FileNotFoundError("Run python -m experiments.prepare_model first")
    return SentenceTransformer(str(MODEL_SNAPSHOT))


st.title("MemoryGraph MVP")
st.caption("Local research demonstration · inspectable retrieval · synthetic example memories only")
st.info(
    "This prototype does not measure energy or establish real-user performance. "
    "Model output is displayed as text and never executed."
)

with st.sidebar:
    st.header("Memory management")
    new_memory = st.text_input(
        "Add a memory", placeholder="The user prefers pytest for Python tests."
    )
    if st.button("Add memory", disabled=not new_memory.strip()):
        store.add(new_memory)
        st.rerun()
    if st.button("Load fictional sample memories"):
        if not store.list():
            for text in [
                "The user prefers pytest for Python tests.",
                "The user writes TypeScript for web interfaces.",
                "The user prefers concise documentation with examples.",
                "The user prioritizes low response latency.",
            ]:
                store.add(text)
        st.rerun()
    for memory in store.list():
        with st.expander(memory.text):
            changed = st.text_area("Edit memory", memory.text, key=f"edit_{memory.id}")
            if st.button("Save", key=f"save_{memory.id}"):
                store.update(memory.id, changed)
                st.rerun()
            if st.button("Delete", key=f"delete_{memory.id}"):
                store.delete(memory.id)
                st.rerun()

query = st.text_area("Coding question", placeholder="Write Python tests for this function.")
method = st.selectbox(
    "Retrieval method",
    ["tfidf", "semantic", "graph_assisted"],
    format_func=lambda x: {
        "tfidf": "TF-IDF cosine",
        "semantic": "Pretrained embeddings",
        "graph_assisted": "TF-IDF + entity graph",
    }[x],
)
top_k = st.slider("Maximum memories", 1, 5, 3)

if st.button("Retrieve relevant memories", disabled=not query.strip() or not store.list()):
    try:
        encoder = load_encoder() if method == "semantic" else None
        ranked = rank_memories(query, store.list(), method=method, encoder=encoder, k=top_k)
        st.session_state["ranked"] = ranked
        st.session_state["query"] = query
    except Exception as exc:
        st.error(f"Retrieval unavailable: {exc}")

ranked = st.session_state.get("ranked", [])
if ranked:
    st.subheader("Retrieved context")
    for item in ranked:
        st.write(f"{item.score:.3f} · {item.memory.text}")
    context = "\n".join(f"- {item.memory.text}" for item in ranked)
    st.code(context, language="text")
    st.caption("The context above is the exact memory text passed to the optional local model.")
    st.divider()
    st.subheader("Optional local inference")
    model_name = st.text_input(
        "Installed Ollama model name", placeholder="Enter a model already installed locally"
    )
    if st.button("Compare with and without memory", disabled=not model_name.strip()):
        question = st.session_state["query"]
        try:
            baseline = ask_ollama(question, model_name)
            assisted = ask_ollama(
                f"Relevant user memories:\n{context}\n\nUser request:\n{question}", model_name
            )
            for label, result in [
                ("Without memory", baseline),
                ("With retrieved memory", assisted),
            ]:
                st.markdown(f"### {label}")
                st.write(result.text)
                st.caption(
                    f"Latency: {result.elapsed_seconds:.2f} s · "
                    f"prompt tokens: {result.prompt_tokens} · "
                    f"output tokens: {result.output_tokens}"
                )
            st.warning(
                "This side-by-side display does not score response correctness. "
                "Token counts come from Ollama when provided."
            )
        except Exception as exc:
            st.error(f"Local inference unavailable: {exc}")
