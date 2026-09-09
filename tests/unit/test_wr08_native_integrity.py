import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts.analysis import verify_wr08_native_episodes as audit


@pytest.fixture
def episode(tmp_path):
    n = 3
    metadata = {
        "task_id": "WR-08", "seed": 7, "episode_identity": "test-identity",
        "expert_plan": {"metadata": {
            "task_revision": audit.TASK_REVISION,
            "carton_asset_by_item": dict(audit.CARTON_ASSET_BY_ITEM),
        }},
    }
    (tmp_path / "episode.json").write_text(json.dumps({
        "metadata": metadata, "accepted": True, "planner_pass": True,
        "termination": {"success": True}, "step_count": n, "elapsed_s": 1.0,
        "video": {"fps": 20, "frame_stride": 1, "cameras": audit.CAMERAS,
                  "frame_counts": {c: n for c in audit.CAMERAS}},
    }))
    (tmp_path / "planner.json").write_text(json.dumps({
        "episode_result": {"success": True, "stage_count": 17},
        "final_diagnostics": {"success": True},
        "rule_policy": {"operations": [{"success": True}] * 17},
    }))
    with h5py.File(tmp_path / "trajectory.h5", "w") as handle:
        demo = handle.create_group("data/demo_1")
        handle["data"].attrs["env_args"] = json.dumps({"env_kwargs": {"camera_depths": True}})
        demo.create_dataset("actions", data=np.zeros((n, 13), dtype=np.float32))
        demo.create_dataset("obs/robot_state", data=np.zeros((n, 37), dtype=np.float32))
        demo.create_group("obs/depth")
        for name in ("dones", "rewards", "success", "terminated"):
            demo.create_dataset(name, data=np.ones(n))
        demo.create_dataset("truncated", data=np.zeros(n))
        demo.create_dataset("timestamps/sim_time", data=np.arange(n) * 0.05)
        for name in ("diagnostics/frame_json", "oracle/frame_json"):
            demo.create_dataset(name, data=np.zeros(n))
        for camera in audit.CAMERAS:
            demo.create_dataset(f"calibration/{camera}/intrinsic", data=np.zeros((n, 3, 3)))
            demo.create_dataset(f"calibration/{camera}/extrinsic_c2w", data=np.zeros((n, 4, 4)))
            demo.create_dataset(f"video_frame_indices/{camera}", data=np.arange(n))
    (tmp_path / "videos").mkdir()
    for camera in audit.CAMERAS:
        (tmp_path / "videos" / f"{camera}.mp4").write_bytes(b"test-only-placeholder" * 4)
    return tmp_path


def test_empty_depth_group_is_not_recorded_depth(episode):
    result = audit.verify(episode, videos=False)
    assert result["depth_rendered"] is True
    assert result["depth_recorded"] is False
    assert result["steps"] == 3
    assert result["mp4_verified"] is False


@pytest.mark.parametrize("corruption,message", [
    ("nan_action", "nonfinite"),
    ("short_state", "unaligned"),
    ("terminal", "terminal frame"),
    ("depth", "recorded depth datasets"),
    ("timestamps", "timestamps"),
    ("frame_indices", "frame indices"),
])
def test_corrupted_native_data_is_rejected(episode, corruption, message):
    with h5py.File(episode / "trajectory.h5", "a") as handle:
        demo = handle["data/demo_1"]
        if corruption == "nan_action":
            demo["actions"][0, 0] = np.nan
        elif corruption == "short_state":
            del demo["obs/robot_state"]
            demo.create_dataset("obs/robot_state", data=np.zeros((2, 37)))
        elif corruption == "terminal":
            demo["success"][-1] = 0
        elif corruption == "depth":
            demo.create_dataset("obs/depth/camera", data=np.zeros((3, 2, 2)))
        elif corruption == "timestamps":
            demo["timestamps/sim_time"][1] = 0.2
        else:
            demo[f"video_frame_indices/{audit.CAMERAS[0]}"][1] = 0
    with pytest.raises(ValueError, match=message):
        audit.verify(episode, videos=False)


def test_mp4_header_length_is_checked_independently(episode, monkeypatch):
    info = {"nb_frames": "2", "width": 224, "height": 224, "r_frame_rate": "20/1"}
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps({"streams": [info]})))
    with pytest.raises(ValueError, match="actual MP4 frame count mismatch"):
        audit.verify(episode, videos=True)


def test_mp4_decoding_must_match_header_and_hdf5(episode, monkeypatch):
    info = {"nb_frames": "3", "nb_read_frames": "2", "width": 224, "height": 224, "r_frame_rate": "20/1"}
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps({"streams": [info]})))
    with pytest.raises(ValueError, match="decoded MP4 frame count mismatch"):
        audit.verify(episode, videos=True)


def test_mp4_verification_requests_all_frames_to_be_decoded(episode, monkeypatch):
    calls = []
    info = {"nb_frames": "3", "nb_read_frames": "3", "width": 224, "height": 224, "r_frame_rate": "20/1"}

    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({"streams": [info]}))

    monkeypatch.setattr(audit.subprocess, "run", probe)
    result = audit.verify(episode, videos=True)
    assert result["mp4_verified"] and result["mp4_frames_decoded"]
    assert len(calls) == 3 and all("-count_frames" in command for command in calls)


def test_decoder_errors_fail_even_if_frame_counts_match(episode, monkeypatch):
    info = {"nb_frames": "3", "nb_read_frames": "3", "width": 224, "height": 224, "r_frame_rate": "20/1"}
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps({"streams": [info]}), stderr="corrupt decoded frame",
    ))
    with pytest.raises(ValueError, match="MP4 decode errors"):
        audit.verify(episode, videos=True)
