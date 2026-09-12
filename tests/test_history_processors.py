import copy
import json
from pathlib import Path

import pytest

from sweagent.agent.history_processors import ImageParsingHistoryProcessor, LastNObservations, TagToolCallObservations
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


@pytest.mark.parametrize("role", ["user", "tool"])
def test_image_parsing_preserves_existing_multimodal_content(role):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,b2xk"}}
    history = [
        {
            "role": role,
            "content": [
                {"type": "text", "text": "existing image"},
                image,
                {"type": "text", "text": "new image ![new](data:image/png;base64,bmV3)"},
            ],
            "message_type": "observation",
        }
    ]
    original = copy.deepcopy(history)
    processor = ImageParsingHistoryProcessor()
    result = processor(history)
    images = [item for item in result[0]["content"] if item["type"] == "image_url"]
    assert images == [image, {"type": "image_url", "image_url": {"url": "data:image/png;base64,bmV3"}}]
    assert history == original
    assert processor(result) == result


def test_image_parsing_preserves_image_only_and_empty_content():
    processor = ImageParsingHistoryProcessor()
    for content in ([], [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]):
        history = [{"role": "user", "content": content, "message_type": "observation"}]
        assert processor(history) == history
