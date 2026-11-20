from memorygraph.conflicts import explicit_preference, find_conflicts, selected_conflicts
from memorygraph.store import Memory


def test_explicit_preferences_and_conflict_pairs():
    memories = [
        Memory("a", "The user prefers pytest for Python tests."),
        Memory("b", "I prefer unittest for Python tests."),
        Memory("c", "The user uses VS Code and pytest."),
    ]
    conflicts = find_conflicts(memories)
    assert len(conflicts) == 1
    assert (conflicts[0].first_id, conflicts[0].second_id) == ("a", "b")
    assert conflicts[0].category == "testing framework"
    assert selected_conflicts(memories, {"a"}) == []
    assert selected_conflicts(memories, {"a", "b"}) == conflicts


def test_narrow_detector_avoids_negation_and_aliases_editor():
    assert explicit_preference("The user does not prefer pytest") is None
    assert explicit_preference("The user prefers VSCode") == ("editor", "vs code")
    assert (
        len(
            find_conflicts(
                [Memory("a", "The user prefers VSCode."), Memory("b", "The user prefers PyCharm.")]
            )
        )
        == 1
    )
