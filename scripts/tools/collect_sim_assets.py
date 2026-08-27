#!/usr/bin/env python3
"""Collect the RoboCasa / RoboSuite assets used by MobileMemBench.

The collector is deliberately source based: task YAML files provide the fixed
layout/style pairs, while RoboCasa's style and fixture registries determine the
actual model and texture files.  It does not need to construct a MuJoCo
environment, so it also works on machines without a display or MuJoCo Python
bindings.

Examples::

    python scripts/tools/collect_sim_assets.py --manifest /tmp/membench-assets.json
    python scripts/tools/collect_sim_assets.py --copy-to /tmp/membench-assets
    python scripts/tools/collect_sim_assets.py --tar /tmp/membench-assets.tar.gz

The copied tree preserves repository-relative paths.  To use it on another
machine, check out the pinned RoboCasa and RoboSuite commits first, then copy
the bundle over those two source trees (or use the manifest as a download
checklist).  The task code and configuration remain supplied by this repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = Path("robocasa/robocasa/models/assets")
ROBOSUITE_ASSETS_ROOT = Path("robosuite/robosuite/models/assets")

# These are task-owned, fixed model paths.  Keeping this list here makes the
# bundle independent of runtime seed choices and of RoboCasa's object sampler.
FIXED_OBJECT_DIRS = (
    "objects/objaverse/canned_food/canned_food_11",
    "objects/objaverse/canned_food/canned_food_15",
    "objects/objaverse/cup/cup_2",
    "objects/objaverse/cup/cup_5",
    "objects/objaverse/plate/plate_3",
    "objects/objaverse/plate/plate_14",
    "objects/objaverse/sponge/sponge_0",
    "objects/objaverse/sponge/sponge_3",
    "objects/objaverse/waffle/waffle_0",
    "objects/aigen_objs/bar_soap/bar_soap_4",
    "objects/aigen_objs/ketchup/ketchup_3",
    "objects/lightwheel/mustard/Mustard003",
    "objects/lightwheel/oven_tray/OvenTray001",
    "objects/lightwheel/saucepan/Saucepan002",
    "objects/lightwheel/sugar_cube/SugarCube003",
)

# Fixture registries use these aliases for several layout fixture types.
REGISTRY_ALIASES = {
    "hinge_cabinet": "cabinet",
    "single_cabinet": "cabinet",
    "open_cabinet": "cabinet",
    "panel_cabinet": "cabinet",
    "housing_cabinet": "cabinet",
    "stack": "cabinet",
    "box": "cabinet",
    "drawer": "cabinet",
}

XML_ASSET_REFERENCE_RE = re.compile(r"\b(?:file|meshfile)\s*=\s*(['\"])(.*?)\1")


def _repo_path(relative: str | Path) -> Path:
    return ROOT / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_strings(child)


def _asset_path_from_value(value: str) -> str | None:
    value = value.replace("\\", "/")
    prefixes = (
        "fixtures/",
        "textures/",
        "generative_textures/",
        "scenes/",
        "arenas/",
        "objects/",
    )
    if value.startswith(prefixes):
        return value
    return None


def _model_asset_path(value: str, assets_root: Path) -> Path | None:
    """Resolve a registry model value, accepting directory-style values."""
    if not value.startswith(("fixtures/", "objects/")):
        return None
    candidate = assets_root / value
    if candidate.is_dir():
        candidate = candidate / "model.xml"
    return candidate if candidate.exists() else None


def _add_file(files: set[Path], path: Path) -> None:
    if path.is_file():
        files.add(path)


def _add_tree(files: set[Path], root: Path) -> None:
    if root.is_file():
        files.add(root)
    elif root.is_dir():
        files.update(path for path in root.rglob("*") if path.is_file())


def _add_xml_closure(files: set[Path], xml_path: Path, assets_root: Path) -> None:
    """Add an XML and local mesh/texture files referenced by it."""
    if not xml_path.exists():
        return
    assets_root = assets_root.resolve()
    _add_file(files, xml_path)
    try:
        text = xml_path.read_text(errors="replace")
    except OSError:
        return
    for _, raw in XML_ASSET_REFERENCE_RE.findall(text):
        if raw.startswith(("/", "http://", "https://")):
            continue
        local = (xml_path.parent / raw).resolve()
        if local.is_file() and (local == assets_root or assets_root in local.parents):
            _add_file(files, local)


def _task_xml_reference_path(xml_path: Path, raw: str) -> Path | None:
    """Resolve a task XML reference to an external source-tree file.

    References must resolve exactly as RoboSuite resolves them: relative to the
    XML file.  Invalid paths are reported rather than guessed or rewritten.
    """
    normalized = raw.replace("\\", "/")
    candidate = (xml_path.parent / normalized).resolve()
    if candidate.is_file() and candidate.is_relative_to(ROOT):
        return candidate
    return None


def _collect_task_xml_assets(
    files: set[Path], task_local_xmls: list[Path]
) -> dict[str, Any]:
    """Collect external mesh/texture files directly referenced by task XMLs."""
    references: list[dict[str, str | None]] = []
    missing: set[str] = set()
    for xml_path in task_local_xmls:
        try:
            text = xml_path.read_text(errors="replace")
        except OSError:
            continue
        for _, raw in XML_ASSET_REFERENCE_RE.findall(text):
            if raw.startswith(("/", "http://", "https://")):
                continue
            resolved = _task_xml_reference_path(xml_path, raw)
            reference = {
                "xml": xml_path.relative_to(ROOT).as_posix(),
                "reference": raw,
                "resolved": (
                    resolved.relative_to(ROOT).as_posix() if resolved is not None else None
                ),
            }
            references.append(reference)
            if resolved is None:
                missing.add(f"{xml_path.relative_to(ROOT)} -> {raw}")
            elif any(
                resolved.is_relative_to((ROOT / source_name).resolve())
                for source_name in ("robocasa", "robosuite")
            ):
                _add_file(files, resolved)
    return {
        "references": references,
        "missing": sorted(missing),
    }


def _layout_and_style_files(
    task_configs: list[Path],
) -> tuple[set[Path], set[int], set[int], set[tuple[int, int]]]:
    files: set[Path] = set()
    layout_ids: set[int] = set()
    style_ids: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    for config_path in task_configs:
        data = yaml.safe_load(config_path.read_text()) or {}
        scene = data.get("scene") or {}
        if "layout_id" in scene:
            layout_ids.add(int(scene["layout_id"]))
        if "style_id" in scene:
            style_ids.add(int(scene["style_id"]))
        if "layout_id" in scene and "style_id" in scene:
            pairs.add((int(scene["layout_id"]), int(scene["style_id"])))

    for layout_id in sorted(layout_ids):
        folder = "test" if layout_id <= 10 else "train"
        _add_file(files, _repo_path(ASSETS_ROOT / f"scenes/kitchen_layouts/{folder}/layout{layout_id:03d}.yaml"))
    for style_id in sorted(style_ids):
        folder = "test" if style_id <= 10 else "train"
        _add_file(files, _repo_path(ASSETS_ROOT / f"scenes/kitchen_styles/{folder}/style{style_id:03d}.yaml"))
    return files, layout_ids, style_ids, pairs


def _collect_fixture_assets(
    files: set[Path], pairs: set[tuple[int, int]], assets_root: Path
) -> dict[str, Any]:
    """Resolve selected fixture variants from layout/style/registry YAML."""
    selected: dict[str, set[str]] = defaultdict(set)
    registry_names: set[str] = set()
    missing: set[str] = set()
    style_data: dict[int, dict[str, Any]] = {}
    style_ids = {style_id for _, style_id in pairs}
    for style_id in style_ids:
        folder = "test" if style_id <= 10 else "train"
        path = assets_root / f"scenes/kitchen_styles/{folder}/style{style_id:03d}.yaml"
        style_data[style_id] = yaml.safe_load(path.read_text()) or {}

    # First pass: identify registries and the IDs selected by each layout.
    layout_ids = {layout_id for layout_id, _ in pairs}
    for layout_id in layout_ids:
        folder = "test" if layout_id <= 10 else "train"
        path = assets_root / f"scenes/kitchen_layouts/{folder}/layout{layout_id:03d}.yaml"
        layout = yaml.safe_load(path.read_text()) or {}
        for group in layout.values():
            if not isinstance(group, dict):
                continue
            for fixture_list in group.values():
                if not isinstance(fixture_list, list):
                    continue
                for fixture in fixture_list:
                    if not isinstance(fixture, dict) or "type" not in fixture:
                        continue
                    fixture_type = str(fixture["type"])
                    registry = REGISTRY_ALIASES.get(fixture_type, fixture_type)
                    registry_names.add(registry)
                    for pair_layout_id, pair_style_id in pairs:
                        if pair_layout_id != layout_id:
                            continue
                        style = style_data[pair_style_id]
                        style_value = style.get(registry, "default")
                        if "config_name" in fixture:
                            ids: Any = fixture["config_name"]
                        elif "default_config_name" in fixture and isinstance(style_value, dict):
                            ids = style_value.get(
                                fixture["default_config_name"], style_value.get("default")
                            )
                        elif isinstance(style_value, dict):
                            ids = style_value.get("default")
                        else:
                            ids = style_value
                        if not isinstance(ids, list):
                            ids = [ids]
                        selected[registry].update(
                            str(item) for item in ids if item is not None
                        )

    # Registry YAML and the concrete files they point at.
    for registry in sorted(registry_names):
        registry_path = assets_root / f"fixtures/fixture_registry/{registry}.yaml"
        _add_file(files, registry_path)
        if not registry_path.exists():
            continue
        registry_data = yaml.safe_load(registry_path.read_text()) or {}
        merged_ids = {"default", *selected.get(registry, ())}
        for config_id in sorted(merged_ids):
            config = registry_data.get(config_id)
            if not isinstance(config, dict):
                continue
            for value in _walk_strings(config):
                asset_value = _asset_path_from_value(value)
                if asset_value is None:
                    continue
                if asset_value.startswith(("fixtures/", "objects/")):
                    model_path = _model_asset_path(asset_value, assets_root)
                    if model_path is not None:
                        _add_xml_closure(files, model_path, assets_root)
                        # Some fixture XMLs are directory-local models with
                        # several mesh files; keeping the directory guarantees
                        # no hidden relative dependency is lost.
                        _add_tree(files, model_path.parent)
                    else:
                        missing.add(str(assets_root / asset_value))
                else:
                    candidate = assets_root / asset_value
                    if candidate.is_file():
                        _add_file(files, candidate)
                    else:
                        missing.add(str(candidate))

    # Cabinet panel/handle names are Python registry keys rather than paths.
    # Their directories are small compared with the full kitchen download and
    # keeping them all handles both primitive and selected custom variants.
    for directory in ("fixtures/cabinets", "fixtures/handles"):
        _add_tree(files, assets_root / directory)

    return {
        "registry_names": sorted(registry_names),
        "selected_config_ids": {key: sorted(values) for key, values in sorted(selected.items())},
        "missing_registry_assets": sorted(missing),
    }


def collect() -> dict[str, Any]:
    task_configs = sorted(
        path for path in (ROOT / "bench/tasks").glob("*/resources/task.yaml")
    )
    files, layout_ids, style_ids, pairs = _layout_and_style_files(task_configs)
    assets_root = _repo_path(ASSETS_ROOT)
    fixture_info = _collect_fixture_assets(files, pairs, assets_root)

    # Fixed object models and their complete local dependency trees.
    for relative in FIXED_OBJECT_DIRS:
        _add_tree(files, assets_root / relative)

    # RoboCasa arena and metadata used by KitchenArena.
    for relative in (
        "arenas/empty_kitchen_arena.xml",
        "novel_instructions/task_instruction_variants.csv",
        # Procedural cabinet/drawer XMLs keep this legacy fallback texture in
        # their asset section even when a style later replaces it.
        "textures/ceramic.png",
        # Wall backing geoms select this texture in RoboCasa Python code, so it
        # is not discoverable from the selected style registry entries alone.
        "textures/flat/light_gray.png",
    ):
        _add_file(files, assets_root / relative)

    # Task-local XMLs are supplied by this repository, but may directly refer
    # to external RoboCasa / RoboSuite textures and meshes.
    task_local_xmls = sorted((ROOT / "bench/tasks").glob("*/resources/assets/*.xml"))
    task_xml_info = _collect_task_xml_assets(files, task_local_xmls)

    # Minimal RoboSuite closure for robots="PandaOmron".
    robosuite_root = _repo_path(ROBOSUITE_ASSETS_ROOT)
    for relative in (
        "robots/panda",
        "bases/omron_mobile_base.xml",
        "bases/meshes/omron_mobile_base",
        "grippers/panda_gripper.xml",
        "grippers/meshes/panda_gripper",
    ):
        _add_tree(files, robosuite_root / relative)

    # Normalize and retain only files under the two external source trees.
    files = {path.resolve() for path in files if path.is_file()}
    allowed_roots = (_repo_path("robocasa").resolve(), _repo_path("robosuite").resolve())
    files = {path for path in files if any(root == path or root in path.parents for root in allowed_roots)}

    required_paths = [
        assets_root / relative for relative in FIXED_OBJECT_DIRS
    ] + [
        assets_root / "arenas/empty_kitchen_arena.xml",
        assets_root / "novel_instructions/task_instruction_variants.csv",
        assets_root / "textures/ceramic.png",
        assets_root / "textures/flat/light_gray.png",
        robosuite_root / "robots/panda",
        robosuite_root / "bases/omron_mobile_base.xml",
        robosuite_root / "bases/meshes/omron_mobile_base",
        robosuite_root / "grippers/panda_gripper.xml",
        robosuite_root / "grippers/meshes/panda_gripper",
    ]
    missing_required = [str(path) for path in required_paths if not path.exists()]
    missing_required.extend(fixture_info["missing_registry_assets"])
    missing_required.extend(task_xml_info["missing"])

    entries = []
    for path in sorted(files):
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    total_bytes = sum(item["bytes"] for item in entries)
    return {
        "schema_version": 2,
        "generated_by": "scripts/tools/collect_sim_assets.py",
        "repository_root": str(ROOT),
        "task_ids": [path.parent.parent.name for path in task_configs],
        "layout_ids": sorted(layout_ids),
        "style_ids": sorted(style_ids),
        "layout_style_pairs": [list(pair) for pair in sorted(pairs)],
        "robocasa_commit": _git_commit(ROOT / "robocasa"),
        "robosuite_commit": _git_commit(ROOT / "robosuite"),
        "fixture_resolution": fixture_info,
        "task_xml_resolution": task_xml_info,
        "missing_required": sorted(set(missing_required)),
        "fixed_object_directories": list(FIXED_OBJECT_DIRS),
        "task_local_xmls": [path.relative_to(ROOT).as_posix() for path in task_local_xmls],
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / (1024**3), 3),
        "files": entries,
    }


def write_copy(manifest: dict[str, Any], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        source = ROOT / item["path"]
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def write_tar(manifest: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix == ".zst" or destination.name.endswith(".tar.zst"):
        raise ValueError("tar.zst is not portable across the supported Python versions; use .tar.gz")
    mode = "w:gz"
    with tarfile.open(destination, mode) as archive:
        for item in manifest["files"]:
            archive.add(ROOT / item["path"], arcname=item["path"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="write JSON manifest here")
    parser.add_argument("--copy-to", type=Path, help="copy the selected files to this directory")
    parser.add_argument("--tar", type=Path, help="write a tar.gz bundle")
    args = parser.parse_args()
    if not (args.manifest or args.copy_to or args.tar):
        parser.error("choose at least one of --manifest, --copy-to, or --tar")

    manifest = collect()
    if manifest["missing_required"]:
        print("required assets are missing:", file=sys.stderr)
        for path in manifest["missing_required"]:
            print(f"  {path}", file=sys.stderr)
        return 2
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n")
    if args.copy_to:
        write_copy(manifest, args.copy_to)
    if args.tar:
        write_tar(manifest, args.tar)
    print(
        f"selected {manifest['file_count']} files "
        f"({manifest['total_gib']:.3f} GiB) for layouts={manifest['layout_ids']} "
        f"styles={manifest['style_ids']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
