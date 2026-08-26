#!/usr/bin/env python
"""Offline ACT checkpoint validation on a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
LEROBOT_SRC = ROOT / "lerobot" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


DEFAULT_CHECKPOINT_ROOT = ROOT / "ckpts" / "act" / "act_hs_wx_01_chunk50" / "checkpoints"


def parse_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    result: list[int] = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            result.extend(range(start, end + step, step))
        else:
            result.append(int(item))
    return result


def resolve_checkpoint_paths(values: list[Path] | None) -> list[Path]:
    if values:
        candidates = values
    else:
        candidates = [
            DEFAULT_CHECKPOINT_ROOT / "050000",
            DEFAULT_CHECKPOINT_ROOT / "100000",
        ]

    resolved: list[Path] = []
    for path in candidates:
        path = path.expanduser().resolve()
        if (path / "pretrained_model").is_dir():
            path = path / "pretrained_model"
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(f"checkpoint model.safetensors not found under {path}")
        resolved.append(path)
    return resolved


def repo_id_from_train_config(checkpoint: Path) -> str | None:
    config_path = checkpoint / "train_config.json"
    if not config_path.exists():
        return None
    with config_path.open() as f:
        payload = json.load(f)
    dataset = payload.get("dataset") or {}
    repo_id = dataset.get("repo_id")
    return str(repo_id) if repo_id else None


def dataset_root_from_train_config(checkpoint: Path) -> Path | None:
    config_path = checkpoint / "train_config.json"
    if not config_path.exists():
        return None
    with config_path.open() as f:
        payload = json.load(f)
    dataset = payload.get("dataset") or {}
    root = dataset.get("root")
    return Path(root).expanduser() if root else None


def load_dataset(
    *,
    repo_id: str,
    root: Path,
    policy_cfg: PreTrainedConfig,
    episodes: list[int] | None,
    video_backend: str,
    tolerance_s: float,
) -> LeRobotDataset:
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy_cfg, ds_meta)
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
        return_uint8=True,
        tolerance_s=tolerance_s,
    )


def select_eval_subset(dataset: torch.utils.data.Dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return torch.utils.data.Subset(dataset, indices)


def tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().tolist()]


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    dataset: torch.utils.data.Dataset,
    camera_keys: list[str],
    batch_size: int,
    num_workers: int,
    device: str,
    max_batches: int | None,
) -> dict[str, Any]:
    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.device = device
    policy = ACTPolicy.from_pretrained(checkpoint, config=policy_cfg).to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )

    norm_l1_sum = 0.0
    raw_l1_sum = 0.0
    valid_scalar_count = 0
    valid_step_count = 0
    per_dim_l1_sum: torch.Tensor | None = None
    pred_sum: torch.Tensor | None = None
    pred_sq_sum: torch.Tensor | None = None
    gt_sum: torch.Tensor | None = None
    gt_sq_sum: torch.Tensor | None = None

    batches = 0
    samples = 0
    with torch.inference_mode():
        for raw_batch in loader:
            if max_batches is not None and batches >= max_batches:
                break
            batches += 1
            samples += int(raw_batch[ACTION].shape[0])

            for camera_key in camera_keys:
                if camera_key in raw_batch and raw_batch[camera_key].dtype == torch.uint8:
                    raw_batch[camera_key] = raw_batch[camera_key].to(dtype=torch.float32) / 255.0

            raw_action = raw_batch[ACTION].clone()
            batch = preprocessor(raw_batch)
            pred_norm = policy.predict_action_chunk(batch)
            gt_norm = batch[ACTION]
            valid_mask = ~batch["action_is_pad"].unsqueeze(-1)

            norm_abs = (pred_norm - gt_norm).abs() * valid_mask
            norm_l1_sum += float(norm_abs.sum().detach().cpu())

            pred_raw = postprocessor(pred_norm)
            raw_abs = (pred_raw - raw_action).abs() * valid_mask.cpu()
            raw_l1_sum += float(raw_abs.sum())

            valid_scalar_count += int(valid_mask.sum().item()) * int(pred_norm.shape[-1])
            valid_step_count += int(valid_mask.sum().item())

            raw_abs_by_dim = raw_abs.sum(dim=(0, 1)).cpu()
            masked_pred = pred_raw * valid_mask.cpu()
            masked_gt = raw_action * valid_mask.cpu()
            pred_by_dim = masked_pred.sum(dim=(0, 1)).cpu()
            pred_sq_by_dim = (masked_pred.square()).sum(dim=(0, 1)).cpu()
            gt_by_dim = masked_gt.sum(dim=(0, 1)).cpu()
            gt_sq_by_dim = (masked_gt.square()).sum(dim=(0, 1)).cpu()

            if per_dim_l1_sum is None:
                per_dim_l1_sum = raw_abs_by_dim
                pred_sum = pred_by_dim
                pred_sq_sum = pred_sq_by_dim
                gt_sum = gt_by_dim
                gt_sq_sum = gt_sq_by_dim
            else:
                per_dim_l1_sum += raw_abs_by_dim
                pred_sum += pred_by_dim  # type: ignore[operator]
                pred_sq_sum += pred_sq_by_dim  # type: ignore[operator]
                gt_sum += gt_by_dim  # type: ignore[operator]
                gt_sq_sum += gt_sq_by_dim  # type: ignore[operator]

    if valid_scalar_count == 0 or valid_step_count == 0:
        raise RuntimeError("no valid action steps were evaluated")

    assert per_dim_l1_sum is not None
    assert pred_sum is not None and pred_sq_sum is not None
    assert gt_sum is not None and gt_sq_sum is not None

    pred_mean = pred_sum / valid_step_count
    gt_mean = gt_sum / valid_step_count
    pred_std = torch.sqrt(torch.clamp(pred_sq_sum / valid_step_count - pred_mean.square(), min=0.0))
    gt_std = torch.sqrt(torch.clamp(gt_sq_sum / valid_step_count - gt_mean.square(), min=0.0))

    return {
        "checkpoint": str(checkpoint),
        "batches": batches,
        "samples": samples,
        "valid_action_steps": valid_step_count,
        "normalized_action_mae": norm_l1_sum / valid_scalar_count,
        "raw_action_mae": raw_l1_sum / valid_scalar_count,
        "raw_action_mae_by_dim": tensor_to_float_list(per_dim_l1_sum / valid_step_count),
        "pred_action_mean_by_dim": tensor_to_float_list(pred_mean),
        "pred_action_std_by_dim": tensor_to_float_list(pred_std),
        "gt_action_mean_by_dim": tensor_to_float_list(gt_mean),
        "gt_action_std_by_dim": tensor_to_float_list(gt_std),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--episodes", default=None, help="Comma/range syntax, e.g. 0,1,5-8")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = resolve_checkpoint_paths(args.checkpoint)
    first_checkpoint = checkpoints[0]

    dataset_root = args.dataset_root or dataset_root_from_train_config(first_checkpoint)
    if dataset_root is None or not dataset_root.exists():
        raise FileNotFoundError(
            "LeRobot dataset root was not found. Pass --dataset-root /path/to/membench_hs_wx_01"
        )

    repo_id = args.dataset_repo_id or repo_id_from_train_config(first_checkpoint)
    if repo_id is None:
        raise ValueError("dataset repo_id was not found. Pass --dataset-repo-id local/membench_hs_wx_01")

    first_policy_cfg = PreTrainedConfig.from_pretrained(first_checkpoint)
    episodes = parse_int_list(args.episodes)
    dataset = load_dataset(
        repo_id=repo_id,
        root=dataset_root,
        policy_cfg=first_policy_cfg,
        episodes=episodes,
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
    )
    camera_keys = list(dataset.meta.camera_keys)
    dataset = select_eval_subset(dataset, args.max_samples, args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    results = {
        "dataset_root": str(dataset_root),
        "dataset_repo_id": repo_id,
        "episodes": episodes,
        "num_eval_frames": len(dataset),
        "device": args.device,
        "results": [
            evaluate_checkpoint(
                checkpoint=checkpoint,
                dataset=dataset,
                camera_keys=camera_keys,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                max_batches=args.max_batches,
            )
            for checkpoint in checkpoints
        ],
    }

    text = json.dumps(results, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
