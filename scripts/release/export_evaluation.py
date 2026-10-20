#!/usr/bin/env python3
"""Reproducibly export the nine-task evaluation candidate without expert code.

The exporter preserves suite bytes. Task input identity changes only when
removing the unused planner resource; provenance records both identities.
WX-01 requires verified setup replay artifacts, generated separately.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("wx01", "wa01", "wa05", "wr01", "wr08", "ts02", "ts03", "tm01", "tm02")
IDS = ("WX-01", "WA-01", "WA-05", "WR-01", "WR-08", "TS02", "TS03", "TM01", "TM02")
EVALUATION_MODULES = (
    "action",
    "adapter",
    "artifacts",
    "canonical_adapter",
    "client",
    "observation",
    "replay",
    "report_profiles",
    "report_stages",
    "rollout",
    "schemas",
    "subtask",
    "suite",
    "summary",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def retain_definitions(path, roots):
    """Keep requested definitions and their module-local dependency closure."""
    tree = ast.parse(path.read_text())
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    missing = set(roots) - definitions.keys()
    if missing:
        raise ValueError(f"{path}: missing definitions {missing}")
    needed = set(roots)
    while True:
        dependencies = {
            n.id
            for name in needed & definitions.keys()
            for n in ast.walk(definitions[name])
            if isinstance(n, ast.Name)
        }
        expanded = needed | dependencies
        if expanded == needed:
            break
        needed = expanded
    kept = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names = [a for a in node.names if (a.asname or a.name) in needed]
            if node.module == "__future__":
                names = node.names
            if names:
                node.names = names
                kept.append(node)
        elif isinstance(node, ast.Import):
            names = [a for a in node.names if (a.asname or a.name.split(".")[0]) in needed]
            if names:
                node.names = names
                kept.append(node)
        elif (
            any(definitions.get(name) is node for name in needed)
            or isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            kept.append(node)
    tree.body = kept
    write(path, ast.unparse(tree) + "\n")


class RemovePlannerResource(ast.NodeTransformer):
    def visit_Set(self, node):
        return self._container(node)

    def visit_List(self, node):
        return self._container(node)

    def visit_Tuple(self, node):
        return self._container(node)

    def _container(self, node):
        self.generic_visit(node)
        node.elts = [
            e
            for e in node.elts
            if not (isinstance(e, ast.Constant) and e.value == "planner_script")
        ]
        return node


def replace_assignment(path, name, expression):
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(t, ast.Name) and t.id == name
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        )
    )
    lines = path.read_text().splitlines(keepends=True)
    write(
        path,
        "".join(lines[: node.lineno - 1])
        + f"{name} = {expression}\n"
        + "".join(lines[node.end_lineno :]),
    )


def export(target: Path):

    if target.exists():
        raise FileExistsError(f"export target already exists: {target}; choose a fresh directory")
    target.mkdir(parents=True)
    sources = json.loads((ROOT / "configs/benchmark/evaluation_suite_sources.json").read_text())
    for subdir in ("core", "env", "runtime", "policy/protocol"):
        for source in (ROOT / "bench" / subdir).rglob("*.py"):
            copy(source, target / source.relative_to(ROOT))
    for file in (
        "registry.py",
        "tasks/__init__.py",
        "tasks/catalog.py",
        "tasks/contracts.py",
        "tasks/episode_support.py",
        "tasks/runtime_support.py",
        "tasks/button_asset.py",
        "tasks/wa02_button_asset.py",
        "policy/checkpoint.py",
        "policy/server.py",
        "policy/backends/base.py",
        "policy/backends/fake.py",
        "policy/backends/lerobot.py",
        "policy/backends/openpi.py",
        "real_robot/__init__.py",
        "real_robot/contracts.py",
        "real_robot/wr07_prompts.py",
        "policy/backends/__init__.py",
    ):
        copy(ROOT / "bench" / file, target / "bench" / file)
    for subdir in (
        "tasks/families",
        "execution/navigation",
        "execution/sampling",
        "execution/planning",
    ):
        for source in (ROOT / "bench" / subdir).rglob("*.py"):
            if source.name == "mplib_planner.py":
                continue
            copy(source, target / source.relative_to(ROOT))
    copy(ROOT / "bench/execution/contracts.py", target / "bench/execution/contracts.py")
    copy(
        ROOT / "bench/execution/manipulation/object_interactions.py",
        target / "bench/execution/manipulation/object_interactions.py",
    )
    retain_definitions(
        target / "bench/execution/manipulation/object_interactions.py",
        [
            "get_object_hold_status",
            "is_object_released",
            "get_object_transport_snapshot",
            "evaluate_object_transport",
            "configure_right_gripper_force_limit",
        ],
    )
    write(
        target / "bench/execution/manipulation/__init__.py",
        '"""Read-only physical grasp and release evidence; no expert controller."""\n',
    )
    for task in TASKS:
        for source in (ROOT / "bench/tasks" / task).rglob("*"):
            if not source.is_file() or source.suffix not in (".py", ".yaml", ".json", ".xml"):
                continue
            if "expert" in source.parts or source.name in ("recording.py", "README.md"):
                continue
            if "execution" in source.parts and source.name not in (
                "__init__.py",
                "navigation.py",
                "geometry.py",
            ):
                continue
            copy(source, target / source.relative_to(ROOT))
        runtime = target / f"bench/tasks/{task}/runtime.py"
        write(
            target / f"bench/tasks/{task}/execution/__init__.py",
            '"""Task-owned initialization configuration."""\n',
        )
        code = runtime.read_text().replace(
            "from bench.runtime.task_local import lazy_expert_runtime\n", ""
        )
        old = f'    expert_runtime=lazy_expert_runtime(\n        "bench.tasks.{task}.expert.operations:build_operation_handlers"\n    ),'
        assert old in code, runtime
        write(runtime, code.replace(old, "    expert_runtime=None,"))
        nav = target / f"bench/tasks/{task}/execution/navigation.py"
        roots = ["build_navigation_config"]
        if task in ("tm01", "tm02"):
            roots += ["sample_episode_path_poses"]
        # Initializers import task-owned yaw/sampling helpers; retain those too.
        for importer in (runtime, target / f"bench/tasks/{task}/initialization.py"):
            tree = ast.parse(importer.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == f"bench.tasks.{task}.execution.navigation"
                ):
                    roots += [a.name for a in node.names]
        retain_definitions(nav, set(roots))
        initial = target / f"bench/tasks/{task}/initialization.py"
        initial_tree = ast.parse(initial.read_text())
        initial_roots = [
            n.name
            for n in initial_tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name != "initialize_episode"
            and (
                task != "wx01"
                or n.name not in ("initialize_short_start_state", "compact_setup_operation_summary")
            )
        ]
        retain_definitions(initial, initial_roots)
        geometry = target / f"bench/tasks/{task}/execution/geometry.py"
        if task in ("wx01", "wr08") and geometry.is_file():
            retain_definitions(geometry, ["resolve_episode_targets"])
        config = target / f"bench/tasks/{task}/config.py"
        write(
            config, ast.unparse(RemovePlannerResource().visit(ast.parse(config.read_text()))) + "\n"
        )
        task_yaml = target / f"bench/tasks/{task}/resources/task.yaml"
        lines = task_yaml.read_text().splitlines(keepends=True)
        write(
            task_yaml, "".join(line for line in lines if not line.startswith("  planner_script:"))
        )
    # WX-01 setup is replayed, so no manipulation or planner executes in the release.
    wx = target / "bench/tasks/wx01/runtime.py"
    code = wx.read_text().replace(
        "    from bench.tasks.wx01.execution.manipulation import build_manipulation_assembly\n\n",
        "",
    )
    code = code.replace(", initialize_short_start_state", "")
    start = code.index("    setup = build_manipulation_assembly(")
    end = code.index("    evaluator = WX01Evaluator()", start)
    code = (
        code[:start]
        + """    from bench.runtime.setup_replay import replay_setup
    updated, short_start = replay_setup(env, descriptor)
"""
        + code[end:]
    )
    write(wx, code)
    copy(ROOT / "bench/runtime/setup_replay.py", target / "bench/runtime/setup_replay.py")
    replay_root = ROOT / "configs/benchmark/setup_replays/wx01"
    if replay_root.is_dir():
        for path in replay_root.glob("*.json"):
            copy(path, target / path.relative_to(ROOT))
    catalog = target / "bench/tasks/catalog.py"
    tree = ast.parse(catalog.read_text())
    entries = next(
        n.value.elts
        for n in tree.body
        if isinstance(n, ast.AnnAssign) and n.target.id == "_ENTRIES"
    )
    ordered = [next(n for n in entries if n.args[0].value == task_id) for task_id in IDS]
    replace_assignment(
        catalog, "_ENTRIES", "(" + ",\n".join(ast.unparse(n) for n in ordered) + ",)"
    )
    write(
        catalog,
        catalog.read_text().replace(
            "build_expert_plan=lambda episode, item=entry: _expert_plan(item, episode),",
            "build_expert_plan=None,",
        ),
    )
    plugins = target / "bench/tasks/__init__.py"
    tree = ast.parse(plugins.read_text())
    entries = next(
        n.value.elts
        for n in tree.body
        if isinstance(n, ast.AnnAssign) and n.target.id == "BUILTIN_TASK_PLUGINS"
    )
    ordered = [next(n for n in entries if n.args[0].value == task_id) for task_id in IDS]
    replace_assignment(
        plugins, "BUILTIN_TASK_PLUGINS", "(" + ",\n".join(ast.unparse(n) for n in ordered) + ",)"
    )
    write(
        plugins,
        plugins.read_text().replace(
            'expert_plan_symbol=f"{package}.expert.spec:build_expert_plan",',
            "expert_plan_symbol=None,",
        ),
    )
    for module in EVALUATION_MODULES:
        copy(ROOT / f"bench/evaluation/{module}.py", target / f"bench/evaluation/{module}.py")
    write(
        target / "bench/evaluation/__init__.py", '"""Independent policy evaluation benchmark."""\n'
    )
    # Keep historical versions for selected tasks; exclude all other task profiles.
    profile = target / "bench/evaluation/report_profiles.py"
    tree = ast.parse(profile.read_text())
    lines = profile.read_text().splitlines(keepends=True)
    assignment = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "REPORT_STAGE_PROFILES" for t in n.targets)
    )
    expression = ast.unparse(assignment.value)
    replace_assignment(
        profile, "REPORT_STAGE_PROFILES", f"tuple(p for p in {expression} if p.task_id in {IDS!r})"
    )
    for path in (ROOT / "configs/benchmark/observation_profiles").glob("*.yaml"):
        copy(path, target / path.relative_to(ROOT))
    copy(
        ROOT / "configs/benchmark/simulator_environment.json",
        target / "configs/benchmark/simulator_environment.json",
    )
    copy(
        ROOT / "configs/benchmark/simulator_assets.json",
        target / "configs/benchmark/simulator_assets.json",
    )
    copy(ROOT / "scripts/release/cli.py", target / "bench/evaluation/release_cli.py")
    for entry in sources["entries"]:
        source = ROOT / entry["suite_file"]
        assert sha(source) == entry["file_sha256"], source
        copy(source, target / entry["suite_file"])
    copy(
        ROOT / "configs/benchmark/evaluation_suite_sources.json",
        target / "configs/benchmark/evaluation_suite_sources.json",
    )
    for script in (
        "eval/run_remote.py",
        "eval/summarize_run.py",
        "eval/replay_run_actions.py",
        "serve/serve_fake.py",
        "serve/serve_lerobot.py",
        "serve/serve_openpi.py",
        "release/repair_ts03_stage_report.py",
        "release/verify_assets.py",
        "release/check_installed_candidate.py",
        "release/validate_installation_evidence.py",
    ):
        copy(ROOT / "scripts" / script, target / "scripts" / script)
    for doc in (
        "membench_evaluation_audit.md", "membench_evaluation_audit.json",
        "evaluation_asset_audit.json",
    ):
        copy(ROOT / "docs" / doc, target / "docs" / doc)
    for path in (ROOT / "docs/asset_evidence").glob("*.json"):
        copy(path, target / path.relative_to(ROOT))
    for path in (ROOT / "docs/installation_evidence").glob("*"):
        if path.is_file():
            copy(path, target / path.relative_to(ROOT))
    for path in (ROOT / "docs/historical_result_repairs").rglob("*"):
        if path.is_file():
            copy(path, target / path.relative_to(ROOT))
    copy(
        ROOT / "docs/evaluation_release_implementation.md",
        target / "docs/IMPLEMENTATION.md",
    )
    for doc in ("INSTALL.md", "THIRD_PARTY.md", "ASSETS.md", "ISOLATED_INSTALL.md"):
        copy(ROOT / "scripts/release" / doc, target / "docs" / doc)
    for test in (
        "evaluation/test_action.py",
        "evaluation/test_observation.py",
        "evaluation/test_suite.py",
        "evaluation/test_artifacts.py",
        "evaluation/test_summary.py",
        "evaluation/test_builtin_suites.py",
        "evaluation/test_remote_e2e.py",
        "policy_protocol/test_codec.py",
        "policy_protocol/test_contracts.py",
        "serving/test_lerobot_act_backend.py",
        "serving/test_lerobot_diffusion_backend.py",
    ):
        copy(ROOT / "tests" / test, target / "tests" / test)
    for test in (ROOT / "tests/release_evaluation").glob("*.py"):
        copy(test, target / test.relative_to(ROOT))
    for file in ("LICENSE",):
        if (ROOT / file).is_file():
            copy(ROOT / file, target / file)
    write(target / "bench/__init__.py", '"""MobileMemBench evaluation release candidate."""\n')
    write(
        target / "MANIFEST.in",
        "recursive-include bench *.py *.yaml *.json *.xml\nrecursive-include configs *.yaml *.json\nrecursive-include scripts *.py\nrecursive-include docs *.md *.json\nrecursive-include docs/installation_evidence *.txt *.log *.py\ninclude README.md LICENSE release_candidate.json\n",
    )
    write(target / "pyproject.toml", (ROOT / "scripts/release/pyproject.toml").read_text())
    write(target / "README.md", (ROOT / "scripts/release/README.md").read_text())
    provenance = {
        "schema_version": "membench.release_candidate/v1",
        "status": "candidate",
        "tasks": list(IDS),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "suite_count": len(sources["entries"]),
        "suite_bytes_preserved": True,
        "resource_identity_change": "Unused planner_script removed; WX-01 setup replay is separate from model actions.",
        "wx01_setup_seeds": [
            int(p.stem)
            for p in sorted((target / "configs/benchmark/setup_replays/wx01").glob("[0-9]*.json"))
        ],
        "release_ready": False,
    }
    provenance["files"] = {
        str(p.relative_to(target)): sha(p) for p in sorted(target.rglob("*")) if p.is_file()
    }
    write(target / "release_candidate.json", json.dumps(provenance, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release/evaluation")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    export(args.output.resolve())
