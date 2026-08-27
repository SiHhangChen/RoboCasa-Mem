#!/usr/bin/env python
"""Evaluate an OpenPI checkpoint on a local MemBench LeRobot v3 dataset.

The primary metric is the model's teacher-forced flow-matching loss, using the
same transforms and checkpoint normalization statistics as training. Optional
sampled-action metrics are diagnostic only and are reported in physical action
units; they are not a replacement for closed-loop task success.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OPENPI_ROOT = ROOT / "policy" / "openpi"
for source in (OPENPI_ROOT / "src", OPENPI_ROOT / "packages" / "openpi-client" / "src"):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from openpi import transforms as openpi_transforms  # noqa: E402
from openpi.models import model as model_lib  # noqa: E402
from openpi.shared import nnx_utils  # noqa: E402
from openpi.training import config as training_config  # noqa: E402
from openpi.training import membench_v3_dataset  # noqa: E402


ACTION_GROUPS = {
    "arm": tuple(range(0, 7)),
    "gripper": (7,),
    "base": (8, 9, 10),
    "torso": (11,),
    "base_mode": (12,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_membench_ts02_lora")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loss-values-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss-repeats", type=int, default=1)
    parser.add_argument("--action-stride", type=int, default=50)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--max-loss-frames", type=int)
    parser.add_argument("--max-action-samples", type=int)
    parser.add_argument("--skip-action-metrics", action="store_true")
    parser.add_argument("--video-backend", default="torchcodec")
    return parser.parse_args()


@dataclasses.dataclass
class StreamingStats:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    values: list[np.ndarray] = dataclasses.field(default_factory=list, repr=False)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        self.count += int(array.size)
        self.total += float(array.sum())
        self.total_squared += float(np.square(array).sum())
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))
        self.values.append(array.copy())

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "p90": None,
                "p95": None,
                "p99": None,
                "variance": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        values = np.concatenate(self.values)
        p90, p95, p99 = np.quantile(values, [0.90, 0.95, 0.99], method="linear")
        return {
            "count": self.count,
            "mean": mean,
            "median": float(np.median(values)),
            "p90": float(p90),
            "p95": float(p95),
            "p99": float(p99),
            "variance": variance,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


class TransformedDataset:
    def __init__(self, source: Any, transform: Any):
        self.source = source
        self.transform = transform

    def get_items(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self.transform(self.source[index]) for index in indices]


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="The video decoding and encoding capabilities of torchvision are deprecated.*",
        category=UserWarning,
    )
    _validate_args(args)
    started = time.monotonic()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    loss_values_output = (
        args.loss_values_output.expanduser().resolve()
        if args.loss_values_output is not None
        else None
    )
    train_config = training_config.get_config(args.config)

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    seed_rows = _read_jsonl(dataset_root / "meta" / "seeds.jsonl")
    _validate_contract(info, seed_rows, train_config, checkpoint)

    training_asset_id = train_config.data.assets.asset_id or train_config.data.repo_id
    if not training_asset_id:
        raise ValueError("Training config does not declare a normalization asset id")
    data_factory = dataclasses.replace(
        train_config.data,
        repo_id=dataset_root.name,
        assets=dataclasses.replace(
            train_config.data.assets,
            assets_dir=str(checkpoint / "assets"),
            asset_id=training_asset_id,
        ),
    )
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)

    camera_columns = _dataset_camera_columns(data_factory)
    os.environ["OPENPI_MEMBENCH_CAMERA_KEYS"] = ",".join(camera_columns)
    source_dataset = membench_v3_dataset.MemBenchV3Dataset(
        dataset_root.name,
        root=dataset_root,
        action_horizon=train_config.model.action_horizon,
        video_backend=args.video_backend,
    )
    sample_transforms: list[Any] = []
    if data_config.prompt_from_task:
        sample_transforms.append(openpi_transforms.PromptFromLeRobotTask(source_dataset.tasks))
    dataset = TransformedDataset(
        source_dataset,
        openpi_transforms.compose(
            [
                *sample_transforms,
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                openpi_transforms.Normalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.model_transforms.inputs,
            ]
        ),
    )

    print(f"loading checkpoint={checkpoint}", flush=True)
    params = model_lib.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    model = train_config.model.load(params)
    compute_loss = nnx_utils.module_jit(model.compute_loss, static_argnames=("train",))

    episode_to_seed = {int(row["episode_index"]): int(row["seed"]) for row in seed_rows}
    loss_indices = _loss_indices(source_dataset, args.max_loss_frames)
    print(
        f"flow_loss frames={len(loss_indices)} batch_size={args.batch_size} "
        f"repeats={args.loss_repeats}",
        flush=True,
    )
    flow_loss, flow_loss_values = _evaluate_flow_loss(
        compute_loss=compute_loss,
        dataset=dataset,
        source_dataset=source_dataset,
        indices=loss_indices,
        episode_to_seed=episode_to_seed,
        batch_size=args.batch_size,
        base_seed=args.seed,
        repeats=args.loss_repeats,
    )

    action_metrics = None
    if not args.skip_action_metrics:
        sample_actions = nnx_utils.module_jit(model.sample_actions)
        action_indices = _action_indices(
            source_dataset,
            stride=args.action_stride,
            maximum=args.max_action_samples,
        )
        print(
            f"sampled_actions anchors={len(action_indices)} stride={args.action_stride} "
            f"diffusion_steps={args.diffusion_steps}",
            flush=True,
        )
        output_transform = openpi_transforms.compose(
            [
                *data_config.model_transforms.outputs,
                openpi_transforms.Unnormalize(
                    data_config.norm_stats,
                    use_quantiles=data_config.use_quantile_norm,
                ),
                *data_config.data_transforms.outputs,
            ]
        )
        action_metrics = _evaluate_sampled_actions(
            sample_actions=sample_actions,
            dataset=dataset,
            source_dataset=source_dataset,
            indices=action_indices,
            episode_to_seed=episode_to_seed,
            output_transform=output_transform,
            batch_size=args.batch_size,
            base_seed=args.seed,
            diffusion_steps=args.diffusion_steps,
            stride=args.action_stride,
        )

    result = {
        "schema_version": "membench.openpi_offline_eval/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config": args.config,
        "checkpoint": {
            "path": str(checkpoint),
            "step": checkpoint.name,
            "training_asset_id": training_asset_id,
        },
        "dataset": {
            "path": str(dataset_root),
            "repo_id": dataset_root.name,
            "total_frames": int(info["total_frames"]),
            "total_episodes": int(info["total_episodes"]),
            "seeds": [int(row["seed"]) for row in seed_rows],
            "state_schema": info["features"]["observation.state"]["info"]["schema"],
            "action_schema": info["features"]["action"]["info"]["schema"],
            "camera_columns": camera_columns,
        },
        "settings": {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "loss_repeats": args.loss_repeats,
            "max_loss_frames": args.max_loss_frames,
            "action_stride": args.action_stride,
            "max_action_samples": args.max_action_samples,
            "diffusion_steps": args.diffusion_steps,
            "video_backend": args.video_backend,
            "loss_values_output": str(loss_values_output) if loss_values_output else None,
        },
        "flow_matching_loss": flow_loss,
        "sampled_action_diagnostics": action_metrics,
        "notes": {
            "primary_metric": "teacher-forced flow-matching MSE in normalized model action space",
            "sampled_action_metric": "diagnostic MAE/RMSE in raw 13D action units",
            "closed_loop_required": True,
        },
        "runtime": {
            "seconds": time.monotonic() - started,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "membench_revision": _git_revision(ROOT),
            "openpi_revision": _git_revision(OPENPI_ROOT),
        },
    }
    if loss_values_output is not None:
        loss_values_output.parent.mkdir(parents=True, exist_ok=True)
        loss_values_temporary = loss_values_output.with_suffix(loss_values_output.suffix + ".tmp")
        with loss_values_temporary.open("wb") as stream:
            np.savez_compressed(stream, loss=flow_loss_values.astype(np.float32))
        loss_values_temporary.replace(loss_values_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"output={output}", flush=True)
    print(f"flow_matching_loss={flow_loss['overall']['mean']:.8f}", flush=True)
    if action_metrics is not None:
        print(f"sampled_action_mae={action_metrics['overall']['mae']:.8f}", flush=True)


def _evaluate_flow_loss(
    *,
    compute_loss: Any,
    dataset: Any,
    source_dataset: Any,
    indices: list[int],
    episode_to_seed: dict[int, int],
    batch_size: int,
    base_seed: int,
    repeats: int,
) -> tuple[dict[str, Any], np.ndarray]:
    overall = StreamingStats()
    per_episode: dict[int, StreamingStats] = defaultdict(StreamingStats)
    frame_count: dict[int, int] = defaultdict(int)
    total_batches = math.ceil(len(indices) / batch_size)
    base_key = jax.random.key(base_seed)
    for batch_number, batch_indices in enumerate(_batches(indices, batch_size)):
        observation, actions, valid = _load_batch(dataset, batch_indices, batch_size)
        repeat_losses = []
        for repeat in range(repeats):
            rng = jax.random.fold_in(jax.random.fold_in(base_key, batch_number), repeat)
            losses = compute_loss(rng, observation, actions, train=False)
            repeat_losses.append(np.asarray(losses[:valid], dtype=np.float64))
        losses = np.mean(np.stack(repeat_losses, axis=0), axis=0)
        overall.update(losses)
        episode_indices = np.asarray(source_dataset._episode_indices)[batch_indices]
        for row, episode_index in enumerate(episode_indices):
            per_episode[int(episode_index)].update(losses[row])
            frame_count[int(episode_index)] += 1
        if batch_number == 0 or (batch_number + 1) % 25 == 0 or batch_number + 1 == total_batches:
            print(f"flow_loss batch={batch_number + 1}/{total_batches}", flush=True)
    result = {
        "frames": len(indices),
        "horizon": int(actions.shape[-2]),
        "overall": overall.result(),
        "by_episode": [
            {
                "episode_index": episode_index,
                "seed": episode_to_seed[episode_index],
                "frames": frame_count[episode_index],
                **per_episode[episode_index].result(),
            }
            for episode_index in sorted(per_episode)
        ],
    }
    return result, np.concatenate(overall.values)


def _evaluate_sampled_actions(
    *,
    sample_actions: Any,
    dataset: Any,
    source_dataset: Any,
    indices: list[int],
    episode_to_seed: dict[int, int],
    output_transform: Any,
    batch_size: int,
    base_seed: int,
    diffusion_steps: int,
    stride: int,
) -> dict[str, Any]:
    absolute = StreamingStats()
    squared = StreamingStats()
    per_dimension_abs = [StreamingStats() for _ in range(13)]
    per_group_abs = {name: StreamingStats() for name in ACTION_GROUPS}
    per_episode_abs: dict[int, StreamingStats] = defaultdict(StreamingStats)
    anchor_count: dict[int, int] = defaultdict(int)
    total_batches = math.ceil(len(indices) / batch_size)
    base_key = jax.random.key(base_seed + 1)
    for batch_number, batch_indices in enumerate(_batches(indices, batch_size)):
        observation, expert_normalized, valid = _load_batch(dataset, batch_indices, batch_size)
        rng = jax.random.fold_in(base_key, batch_number)
        predicted_normalized = np.asarray(
            sample_actions(rng, observation, num_steps=diffusion_steps)[:valid]
        )
        expert_normalized = np.asarray(expert_normalized[:valid])
        predicted = []
        expert = []
        for row in range(valid):
            state = np.asarray(observation.state[row])
            predicted.append(
                output_transform({"state": state.copy(), "actions": predicted_normalized[row].copy()})["actions"]
            )
            expert.append(
                output_transform({"state": state.copy(), "actions": expert_normalized[row].copy()})["actions"]
            )
        predicted_array = np.asarray(predicted, dtype=np.float64)
        expert_array = np.asarray(expert, dtype=np.float64)
        error = predicted_array - expert_array
        abs_error = np.abs(error)
        absolute.update(abs_error)
        squared.update(np.square(error))
        for dimension in range(13):
            per_dimension_abs[dimension].update(abs_error[..., dimension])
        for name, dimensions in ACTION_GROUPS.items():
            per_group_abs[name].update(abs_error[..., dimensions])
        episode_indices = np.asarray(source_dataset._episode_indices)[batch_indices]
        for row, episode_index in enumerate(episode_indices):
            per_episode_abs[int(episode_index)].update(abs_error[row])
            anchor_count[int(episode_index)] += 1
        if batch_number == 0 or (batch_number + 1) % 10 == 0 or batch_number + 1 == total_batches:
            print(f"sampled_actions batch={batch_number + 1}/{total_batches}", flush=True)

    overall_abs = absolute.result()
    overall_squared = squared.result()
    return {
        "diagnostic_only": True,
        "anchors": len(indices),
        "stride": stride,
        "action_values": absolute.count,
        "overall": {
            "mae": overall_abs["mean"],
            "rmse": math.sqrt(float(overall_squared["mean"])),
            "max_absolute_error": overall_abs["max"],
        },
        "by_action_dimension": [
            {"dimension": dimension, **stats.result()}
            for dimension, stats in enumerate(per_dimension_abs)
        ],
        "by_action_group": {
            name: stats.result() for name, stats in per_group_abs.items()
        },
        "by_episode": [
            {
                "episode_index": episode_index,
                "seed": episode_to_seed[episode_index],
                "anchors": anchor_count[episode_index],
                "mae": per_episode_abs[episode_index].result()["mean"],
            }
            for episode_index in sorted(per_episode_abs)
        ],
    }


def _load_batch(dataset: Any, indices: list[int], batch_size: int) -> tuple[Any, Any, int]:
    items = dataset.get_items(indices)
    valid = len(items)
    items.extend(items[-1] for _ in range(batch_size - valid))
    batch = jax.tree.map(
        lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values], axis=0)),
        *items,
    )
    return model_lib.Observation.from_dict(batch), batch["actions"], valid


def _loss_indices(source_dataset: Any, maximum: int | None) -> list[int]:
    all_indices = list(range(len(source_dataset)))
    if maximum is None or maximum >= len(all_indices):
        return all_indices
    return _stratified_indices(source_dataset, all_indices, maximum)


def _action_indices(source_dataset: Any, *, stride: int, maximum: int | None) -> list[int]:
    episode_indices = np.asarray(source_dataset._episode_indices)
    selected: list[int] = []
    for episode_index in sorted(set(int(value) for value in episode_indices)):
        positions = np.flatnonzero(episode_indices == episode_index)
        selected.extend(int(value) for value in positions[::stride])
    if maximum is not None and maximum < len(selected):
        return _stratified_indices(source_dataset, selected, maximum)
    return selected


def _stratified_indices(source_dataset: Any, candidates: list[int], maximum: int) -> list[int]:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    episodes = np.asarray(source_dataset._episode_indices)
    by_episode: dict[int, list[int]] = defaultdict(list)
    for index in candidates:
        by_episode[int(episodes[index])].append(index)
    chosen: list[int] = []
    remaining = maximum
    episode_ids = sorted(by_episode)
    for position, episode_id in enumerate(episode_ids):
        slots = max(1, remaining // (len(episode_ids) - position))
        values = by_episode[episode_id]
        take = min(slots, len(values), remaining)
        offsets = np.linspace(0, len(values) - 1, take, dtype=int)
        chosen.extend(values[int(offset)] for offset in offsets)
        remaining -= take
        if remaining == 0:
            break
    return sorted(chosen)


def _batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _dataset_camera_columns(data_factory: Any) -> list[str]:
    for transform in data_factory.repack_transforms.inputs:
        structure = getattr(transform, "structure", None)
        if isinstance(structure, dict) and isinstance(structure.get("images"), dict):
            return [str(value) for value in structure["images"].values()]
    raise ValueError("Could not derive dataset camera columns from config repack transforms")


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "loss_repeats", "action_stride", "diffusion_steps"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_loss_frames is not None and args.max_loss_frames <= 0:
        raise ValueError("--max-loss-frames must be positive")
    if args.max_action_samples is not None and args.max_action_samples <= 0:
        raise ValueError("--max-action-samples must be positive")


def _validate_contract(
    info: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    train_config: Any,
    checkpoint: Path,
) -> None:
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {checkpoint / 'params'}")
    required_features = {
        "observation.state",
        "action",
        "observation.images.robot0_agentview_right",
        "observation.images.robot0_eye_in_hand",
    }
    missing = required_features - set(info.get("features", {}))
    if missing:
        raise ValueError(f"Dataset is missing required features: {sorted(missing)}")
    state_dim = int(info["features"]["observation.state"]["shape"][0])
    action_dim = int(info["features"]["action"]["shape"][0])
    if state_dim != 37 or action_dim != 13:
        raise ValueError(f"Expected state/action dimensions 37/13, got {state_dim}/{action_dim}")
    if int(info["total_episodes"]) != len(seed_rows):
        raise ValueError("Seed registry does not match dataset episode count")
    seeds = [int(row["seed"]) for row in seed_rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Validation seeds are not unique: {seeds}")
    if int(train_config.model.action_horizon) <= 0:
        raise ValueError("Training config has an invalid action horizon")
    asset_id = train_config.data.assets.asset_id or train_config.data.repo_id
    if not (checkpoint / "assets" / str(asset_id) / "norm_stats.json").is_file():
        raise FileNotFoundError(f"Checkpoint normalization stats missing for asset {asset_id}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


if __name__ == "__main__":
    main()
