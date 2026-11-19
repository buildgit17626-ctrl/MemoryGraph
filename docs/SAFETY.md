# Safety and isolation

MemoryGraph's demo sends plain-text prompts to an optional local Ollama HTTP endpoint. Model responses are displayed as text only. The application never evaluates, imports, executes, installs, or runs commands from generated content. The backend URL is fixed to loopback (`127.0.0.1`).

Synthetic data contains fictional preferences and no private conversation history. The editable memory store is a local JSON file under `data/generated/`, which is git-ignored. Users should enter only information they are comfortable keeping on their own machine. No credentials are required or stored.

If a future version tests generated code, it must use a separate restricted subprocess or approved sandbox with hard time and memory limits, no network access, a temporary working directory, and a minimal environment. That capability is intentionally absent from this MVP.
