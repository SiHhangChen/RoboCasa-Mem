from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from bench.execution.planning import mplib_planner


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLANNER_SCRIPT = (
    REPOSITORY_ROOT / "bench" / "execution" / "planning" / "mplib_planner.py"
)


def test_planner_repository_root_is_the_project_root() -> None:
    assert mplib_planner.REPOSITORY_ROOT == REPOSITORY_ROOT
    assert (mplib_planner.REPOSITORY_ROOT / "bench").is_dir()


def test_planner_cli_bootstraps_outside_the_repository(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [sys.executable, str(PLANNER_SCRIPT), "--help"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--plan-json" in completed.stdout


def test_planner_reuses_the_active_conda_interpreter(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "robocasa")
    monkeypatch.setattr(mplib_planner.subprocess, "run", fake_run)
    code, result = mplib_planner.run_mplib_planner_subprocess(
        script_path=PLANNER_SCRIPT,
        plan_request={},
        conda_env="robocasa",
    )

    assert code == 0
    assert result == {}
    assert captured["command"][0] == sys.executable
    assert captured["kwargs"]["cwd"] == str(REPOSITORY_ROOT)


def test_planner_defaults_to_isolated_membench_environment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "robocasa")
    monkeypatch.setattr(mplib_planner.subprocess, "run", fake_run)
    code, result = mplib_planner.run_mplib_planner_subprocess(
        script_path=PLANNER_SCRIPT,
        plan_request={},
    )

    assert code == 0
    assert result == {}
    assert captured["command"][:5] == ["conda", "run", "-n", "membench", "python"]


def test_planner_skips_candidate_without_srdf_and_uses_complete_model() -> None:
    calls: list[dict[str, object]] = []
    planner = object()

    class FakeMplib:
        @staticmethod
        def Planner(**kwargs):
            calls.append(kwargs)
            return planner

    candidates = [
        {
            "source": "repo",
            "urdf": "/models/incomplete-panda.urdf",
            "srdf": "/models/incomplete-panda.srdf",
            "urdf_exists": True,
            "srdf_exists": False,
        },
        {
            "source": "mani_skill",
            "urdf": "/models/panda.urdf",
            "srdf": "/models/panda.srdf",
            "urdf_exists": True,
            "srdf_exists": True,
        },
    ]

    result, selected, backend = mplib_planner.instantiate_planner(
        FakeMplib,
        candidates,
    )

    assert result is planner
    assert calls == [
        {
            "urdf": "/models/panda.urdf",
            "srdf": "/models/panda.srdf",
            "move_group": "panda_hand_tcp",
        }
    ]
    assert selected["source"] == "mani_skill"
    assert selected["srdf"] == "/models/panda.srdf"
    assert selected["srdf_mode"] == "file"
    assert backend == "mplib_pinocchio"


def test_planner_reports_skipped_candidates_when_none_can_initialize() -> None:
    class FakeMplib:
        @staticmethod
        def Planner(**kwargs):
            raise AssertionError("a missing URDF must not reach MPLib")

    candidates = [
        {
            "source": "repo",
            "urdf": "/missing/panda.urdf",
            "srdf": "/missing/panda.srdf",
            "urdf_exists": False,
            "srdf_exists": False,
        },
        {
            "source": "repo",
            "urdf": "/models/no-srdf.urdf",
            "srdf": "/models/no-srdf.srdf",
            "urdf_exists": True,
            "srdf_exists": False,
        },
        {
            "source": "mani_skill",
            "error": "ModuleNotFoundError: No module named 'mani_skill'",
        },
    ]

    try:
        mplib_planner.instantiate_planner(FakeMplib, candidates)
    except RuntimeError as exc:
        errors = json.loads(str(exc).split("errors=", 1)[1])
    else:
        raise AssertionError("planner initialization should fail")

    assert [error["error"] for error in errors] == [
        "skipped_missing_urdf",
        "skipped_missing_srdf",
        "ModuleNotFoundError: No module named 'mani_skill'",
    ]
