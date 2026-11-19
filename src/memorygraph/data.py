"""Generate a documented synthetic query-memory relevance dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.config import N_USERS, SCENARIOS, SEED

PREFERENCES = {
    "editor": ["uses VS Code", "prefers Vim keybindings", "works in PyCharm"],
    "testing": ["prefers pytest", "uses unittest", "writes property-based tests"],
    "language": ["writes Python", "writes TypeScript", "writes C++"],
    "performance": ["prioritizes low latency", "prefers low memory use", "profiles CPU hotspots"],
    "documentation": [
        "prefers short docstrings",
        "likes worked examples",
        "maintains detailed README files",
    ],
}
QUERIES = {
    "editor": "Which editor preference should I use for this coding session?",
    "testing": "What test style should the coding assistant follow?",
    "language": "Which programming language does this user prefer?",
    "performance": "What performance constraint matters to this user?",
    "documentation": "How should I document this code for the user?",
}


def make_dataset(n_users: int = N_USERS, seed: int = SEED) -> pd.DataFrame:
    """Create one relevant memory per query among five user-specific memories.

    User IDs are grouped during validation so a persona never appears in both
    train and validation folds. These are fictional templates, not user data.
    """
    if n_users < 5:
        raise ValueError("At least five users are needed for five grouped folds")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for user_number in range(n_users):
        user_id = f"synthetic_user_{user_number:03d}"
        memories = {
            scenario: f"The user {rng.choice(PREFERENCES[scenario])}." for scenario in SCENARIOS
        }
        for scenario in SCENARIOS:
            query_id = f"{user_id}_{scenario}"
            for candidate_scenario, memory_text in memories.items():
                rows.append(
                    {
                        "user_id": user_id,
                        "query_id": query_id,
                        "query_category": scenario,
                        "query_text": QUERIES[scenario],
                        "memory_id": f"{user_id}_{candidate_scenario}",
                        "memory_category": candidate_scenario,
                        "memory_text": memory_text,
                        "relevant": int(candidate_scenario == scenario),
                    }
                )
    return pd.DataFrame.from_records(rows)


def paired_text(frame: pd.DataFrame) -> pd.Series:
    """Keep query and candidate memory together for classification."""
    return frame["query_text"].astype(str) + " [MEMORY] " + frame["memory_text"].astype(str)
