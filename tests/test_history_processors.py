import json
from pathlib import Path

import pytest

from sweagent.agent.history_processors import ClosedWindowHistoryProcessor, LastNObservations, TagToolCallObservations
from sweagent.types import History


def get_history(traj_path: Path):
    return json.loads((traj_path).read_text())["history"]


def count_elided_observations(history: History):
    return len([entry for entry in history if "Old environment output" in entry["content"]])


@pytest.fixture
def test_history(test_trajectories_path: Path):
    return get_history(
        test_trajectories_path
        / "gpt4__swe-agent-test-repo__default_from_url__t-0.00__p-0.95__c-3.00__install-1/6e44b9__sweagenttestrepo-1c2844.traj"
    )


def test_last_n_observations(test_history: History):
    processor = LastNObservations(n=3)
    new_history = processor(test_history)
    total_observations = len([entry for entry in test_history if entry["message_type"] == "observation"])
    # extra -1 because instance template is kept
    expected_elided_observations = total_observations - 3 - 1
    assert count_elided_observations(new_history) == expected_elided_observations


def test_add_tag_to_edits(test_history: History):
    processor = TagToolCallObservations(tags={"test"}, function_names={"edit"})
    new_history = processor(test_history)
    for entry in new_history:
        if entry.get("action", "").startswith("edit "):  # type: ignore
            assert entry.get("tags") == ["test"], entry


@pytest.mark.parametrize("content", ["1: first result\n2: second result\n", "Build failed\n42: error: missing name\n"])
def test_closed_window_preserves_numbered_output_without_file_header(content):
    history = [{"role": "user", "content": content, "message_type": "observation"}]
    assert ClosedWindowHistoryProcessor()(history) == history


def test_closed_window_only_elides_older_windows_for_the_same_file():
    def observation(content):
        return {"role": "user", "content": content, "message_type": "observation"}

    history = [
        observation("[File: example.py (2 lines total)]\n1: old\n2: text\n"),
        observation("1: unrelated command output\n"),
        observation("[File: example.py (2 lines total)]\n1: new\n2: text\n"),
    ]
    result = ClosedWindowHistoryProcessor()(history)
    assert len(result) == len(history)
    assert "Outdated window with 2 lines omitted" in result[0]["content"]
    assert result[1:] == history[1:]
    assert "1: old" in history[0]["content"]
