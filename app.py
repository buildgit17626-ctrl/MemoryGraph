"""Transparent local Streamlit demo for editable coding-assistant memories."""

from __future__ import annotations

import streamlit as st
from sentence_transformers import SentenceTransformer

from experiments.config import DEMO_QUERY, MODEL_SNAPSHOT, ROOT
from memorygraph.conflicts import find_conflicts, selected_conflicts
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


def clear_retrieval() -> None:
    """A changed memory must never remain silently inside an old prompt."""
    for key in ("ranked_all", "retrieved_query", "selected_ids", "approved", "comparison"):
        st.session_state.pop(key, None)


st.title("MemoryGraph MVP")
st.caption("Local prototype · fictional memories · inspectable retrieval and model context")
st.info(
    "This demo does not establish real-user benefit or measure energy. "
    "Generated code is shown as text and never executed."
)

with st.sidebar:
    st.header("Memory management")
    st.caption(
        "1. Add or load memories · 2. Inspect candidates · "
        "3. Approve context · 4. Compare responses"
    )
    new_memory = st.text_input(
        "Add a memory", placeholder="The user prefers pytest for Python tests."
    )
    if st.button("Add memory", disabled=not new_memory.strip()):
        store.add(new_memory)
        clear_retrieval()
        st.rerun()
    if st.button("Load fictional sample memories"):
        if not store.list():
            for text in (
                "The user prefers pytest for Python tests.",
                "The user writes TypeScript for web interfaces.",
                "The user prefers concise documentation with examples.",
                "The user prioritizes low response latency.",
            ):
                store.add(text)
            clear_retrieval()
        st.rerun()
    if st.button("Add conflicting sample preference"):
        example = "The user prefers unittest for Python tests."
        if not any(memory.text == example for memory in store.list()):
            store.add(example)
            clear_retrieval()
        st.rerun()
    for memory in store.list():
        with st.expander(memory.text):
            changed = st.text_area("Edit memory", memory.text, key=f"edit_{memory.id}")
            if st.button("Save", key=f"save_{memory.id}"):
                store.update(memory.id, changed)
                clear_retrieval()
                st.rerun()
            if st.button("Delete", key=f"delete_{memory.id}"):
                store.delete(memory.id)
                clear_retrieval()
                st.rerun()

memories = store.list()
conflicts = find_conflicts(memories)
if conflicts:
    for conflict in conflicts:
        st.warning(
            f"Conflicting {conflict.category} preferences: {conflict.first_value} and "
            f"{conflict.second_value}. Edit or delete a stale memory, or select only one below."
        )
    st.caption("The detector covers explicit testing-framework and editor preferences only.")

st.subheader("1 · Ask and inspect candidates")
question = st.text_area(
    "Coding question", value=DEMO_QUERY, key="question", help="The example is fictional."
)
method = st.selectbox(
    "Retrieval method",
    ["tfidf", "semantic", "graph_assisted"],
    format_func=lambda value: {
        "tfidf": "TF-IDF cosine",
        "semantic": "Pretrained embeddings",
        "graph_assisted": "TF-IDF + entity graph",
    }[value],
)
top_k = st.slider("Suggested memory count", 1, 5, 3)
if st.button("Inspect candidate scores", disabled=not question.strip() or not memories):
    try:
        encoder = load_encoder() if method == "semantic" else None
        ranked_all = rank_memories(
            question, memories, method=method, encoder=encoder, k=len(memories)
        )
        st.session_state["ranked_all"] = ranked_all
        st.session_state["retrieved_query"] = question
        st.session_state["selected_ids"] = [item.memory.id for item in ranked_all[:top_k]]
        st.session_state["approved"] = False
        st.session_state.pop("comparison", None)
    except Exception as error:
        st.error(f"Retrieval unavailable: {error}")

ranked_all = st.session_state.get("ranked_all", [])
if ranked_all:
    st.dataframe(
        [
            {"Rank": index, "Candidate memory": item.memory.text, "Score": round(item.score, 3)}
            for index, item in enumerate(ranked_all, 1)
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Scores rank candidates within the selected method; they are not calibrated probabilities."
    )
    st.subheader("2 · Choose and approve context")
    by_id = {item.memory.id: item.memory for item in ranked_all}
    selected_ids = st.multiselect(
        "Memories to include",
        options=list(by_id),
        format_func=lambda memory_id: by_id[memory_id].text,
        key="selected_ids",
    )
    selected = [by_id[memory_id] for memory_id in selected_ids]
    context = "\n".join(f"- {memory.text}" for memory in selected)
    st.code(context or "No memories selected", language="text")
    st.caption("This is the exact memory text sent with the optional local-model request.")
    active_conflicts = selected_conflicts(memories, set(selected_ids))
    if active_conflicts:
        st.error("Select only one side of each flagged preference conflict before using the model.")
    stale = question != st.session_state["retrieved_query"]
    if stale:
        st.warning(
            "The question changed after retrieval. Inspect candidates again before inference."
        )
    approved = st.checkbox(
        "I reviewed and approve sending these selected memories to my local model",
        key="approved",
        disabled=not selected or bool(active_conflicts) or stale,
    )
    st.subheader("3 · Compare local responses")
    model_name = st.text_input("Installed Ollama model name", placeholder="qwen2.5-coder:0.5b")
    if st.button(
        "Compare without and with approved memory",
        disabled=not (approved and model_name.strip()) or bool(active_conflicts) or stale,
    ):
        try:
            baseline = ask_ollama(question, model_name)
            assisted = ask_ollama(
                f"Relevant user memories:\n{context}\n\nUser request:\n{question}", model_name
            )
            st.session_state["comparison"] = (baseline, assisted)
        except Exception as error:
            st.error(f"Local inference unavailable: {error}")
    if "comparison" in st.session_state:
        left, right = st.columns(2)
        for column, label, result in zip(
            (left, right),
            ("Without memory", "With approved memory"),
            st.session_state["comparison"],
            strict=True,
        ):
            with column:
                st.markdown(f"### {label}")
                st.write(result.text)
                st.caption(
                    f"Latency: {result.elapsed_seconds:.2f} s · "
                    f"prompt tokens: {result.prompt_tokens} · "
                    f"output tokens: {result.output_tokens}"
                )
        st.caption(
            "Response correctness is not scored; token counts come from Ollama when provided."
        )
