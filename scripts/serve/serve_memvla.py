#!/usr/bin/env python
"""Serve MemoryVLA checkpoint over membench-policy/v2."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MEMVLA_SRC = ROOT / "policy" / "MemoryVLA"
if str(MEMVLA_SRC) not in sys.path:
    sys.path.insert(0, str(MEMVLA_SRC))

from bench.evaluation.action import validate_action_chunk
from bench.policy.protocol import (
    ActionContract,
    CheckpointIdentity,
    InferenceRequest,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
)
from bench.policy.server import WebsocketPolicyServer
from bench.runtime.data_contract import ACTION_DIM, ACTION_SCHEMA, STATE_DIM, STATE_SCHEMA

logger = logging.getLogger(__name__)


class MemoryVLABackend:
    """Backend wrapping a MemoryVLA model for the membench-policy/v2 WebSocket protocol.

    The model's ``predict_action`` returns a chunk of shape ``[future_window+1, action_dim]``
    (e.g. ``[16, 13]``).  The backend declares this as the action horizon and the eval suite's
    ``replan_steps`` should match (use a copy of the suite with ``replan_steps: 16``, or whatever
    the model's actual output length is).
    """

    def __init__(
        self,
        vla,
        *,
        unnorm_key: str,
        action_horizon: int,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        server_id: str | None = None,
        checkpoint: CheckpointIdentity | None = None,
        code_revision: dict[str, str] | None = None,
    ) -> None:
        self.vla = vla
        self.unnorm_key = unnorm_key
        self.cfg_scale = cfg_scale
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self._contract = PolicyContract(
            server_id=server_id or "memvla",
            policy={"family": "memvla", "name": "memory_vla"},
            checkpoint=checkpoint or CheckpointIdentity(
                uri="file://unknown",
                fingerprint="sha256:unknown",
            ),
            observation=ObservationContract(
                state_schema=STATE_SCHEMA,
                state_dim=STATE_DIM,
                required_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
                image_size=(224, 224),
            ),
            action=ActionContract(
                schema=ACTION_SCHEMA,
                dim=ACTION_DIM,
                horizon=action_horizon,
            ),
            control_frequency_hz=20.0,
            code_revision=dict(code_revision or {}),
        )
        self._active_episode: str | None = None

    @property
    def contract(self) -> PolicyContract:
        return self._contract

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device: str = "cuda",
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        use_bf16: bool = False,
        checkpoint_uri: str | None = None,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
        hf_token: str | None = None,
    ) -> "MemoryVLABackend":
        """Load a MemoryVLA checkpoint and return a configured backend.

        The checkpoint directory must have the standard structure::

            <run_dir>/
                config.json
                dataset_statistics.json
                checkpoints/
                    step-XXXXXX-epoch-YY-loss=Z.ZZ.pt

        ``checkpoint_path`` should point to the ``.pt`` file under ``checkpoints/``.
        """
        ckpt = checkpoint_path.expanduser().resolve()
        if not ckpt.is_file() or ckpt.suffix != ".pt":
            raise FileNotFoundError(f"MemoryVLA checkpoint must be a .pt file: {ckpt}")

        run_dir = ckpt.parents[1]
        config_json = run_dir / "config.json"
        dataset_statistics_json = run_dir / "dataset_statistics.json"
        if not config_json.is_file():
            raise FileNotFoundError(f"Missing config.json in {run_dir}")
        if not dataset_statistics_json.is_file():
            raise FileNotFoundError(f"Missing dataset_statistics.json in {run_dir}")

        with open(config_json) as f:
            train_cfg = json.load(f)

        # Read dataset statistics to determine unnorm_key and action horizon
        with open(dataset_statistics_json) as f:
            norm_stats = json.load(f)
        unnorm_keys = list(norm_stats.keys())
        if not unnorm_keys:
            raise ValueError("dataset_statistics.json contains no keys")
        unnorm_key = unnorm_keys[0]

        # Determine action horizon from training config
        future_window = int(train_cfg.get("future_action_window_size", 15))
        action_horizon = future_window + 1  # e.g. 16 for future_action_window_size=15

        logger.info(
            "Loading MemoryVLA from %s (unnorm_key=%s, horizon=%d, device=%s)",
            ckpt, unnorm_key, action_horizon, device,
        )
        vla = cls._load_vla(
            ckpt, run_dir, config_json, device=device,
            use_bf16=use_bf16, hf_token=hf_token,
        )

        ckpt_identity = CheckpointIdentity(
            uri=checkpoint_uri or f"file://{ckpt}",
            fingerprint=f"sha256:{ckpt.name}",
            step=_parse_step_from_name(ckpt.name),
        )

        return cls(
            vla,
            unnorm_key=unnorm_key,
            action_horizon=action_horizon,
            cfg_scale=cfg_scale,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            server_id=server_id or f"memvla-{run_dir.name}",
            checkpoint=ckpt_identity,
            code_revision=code_revision,
        )

    @staticmethod
    def _load_vla(
        ckpt_path: Path,
        run_dir: Path,
        config_json: Path,
        *,
        device: str = "cuda",
        use_bf16: bool = False,
        hf_token: str | None = None,
    ):
        """Import and load the MemoryVLA model, adding the memvla source to sys.path."""
        # Import inside the method so the module is only required when actually loading
        import torch
        import json

        # Load the config to get the base_vlm type
        with open(config_json) as f:
            cfg = json.load(f)

        base_vlm_id = cfg["vla"]["base_vlm"]
        action_dim = int(cfg.get("action_dim", 13))
        future_window = int(cfg.get("future_action_window_size", 15))
        use_lora = bool(cfg.get("use_lora", True))
        lora_r = int(cfg.get("lora_r", 16))
        lora_alpha = float(cfg.get("lora_alpha", 32.0))
        lora_dropout = float(cfg.get("lora_dropout", 0.1))
        lora_target_modules = tuple(cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
        mem_length = int(cfg.get("mem_length", 16))
        retrieval_layers = int(cfg.get("retrieval_layers", 2))
        use_timestep_pe = bool(cfg.get("use_timestep_pe", True))
        fusion_type = cfg.get("fusion_type", "gate")
        consolidate_type = cfg.get("consolidate_type", "tome")
        per_token_size = int(cfg.get("per_token_size", 256))
        action_model_type = cfg.get("action_model_type", "DiT-L")
        use_ema = bool(cfg.get("use_ema", False))

        # Load dataset statistics
        ds_stats_path = run_dir / "dataset_statistics.json"
        with open(ds_stats_path) as f:
            norm_stats = json.load(f)

        # Import MemoryVLA modules.  The vision/LLM backbones resolve their pretrained
        # weights from ``pretrained/<name>`` relative to the MemoryVLA repo root, so we
        # chdir there before importing (mirrors how train_wr03.sh runs from that root).
        memvla_root = Path(MEMVLA_SRC)
        cwd = Path.cwd()
        if cwd != memvla_root:
            os.chdir(memvla_root)

        from prismatic.conf import ModelConfig
        from prismatic.models.materialize import (
            get_llm_backbone_and_tokenizer,
            get_vision_backbone_and_transform,
        )
        from vla import MemoryVLA

        # Load base VLM ModelConfig
        model_cfg = ModelConfig.get_choice_class(base_vlm_id)()

        logger.info(
            "Loading vision backbone: %s", model_cfg.vision_backbone_id,
        )
        vision_backbone, _ = get_vision_backbone_and_transform(
            model_cfg.vision_backbone_id,
            model_cfg.image_resize_strategy,
        )

        logger.info(
            "Loading LLM backbone: %s", model_cfg.llm_backbone_id,
        )
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            model_cfg.llm_backbone_id,
            llm_max_length=model_cfg.llm_max_length,
            hf_token=hf_token,
            inference_mode=True,
        )

        logger.info("Building MemoryVLA from checkpoint: %s", ckpt_path)
        vla = MemoryVLA.from_pretrained(
            ckpt_path,
            model_cfg.model_id,
            vision_backbone,
            llm_backbone,
            arch_specifier=model_cfg.arch_specifier,
            freeze_weights=True,
            norm_stats=norm_stats,
            image_resize_strategy=model_cfg.image_resize_strategy,
            action_dim=action_dim,
            future_action_window_size=future_window,
            action_model_type=action_model_type,
            use_ema=use_ema,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            mem_length=mem_length,
            retrieval_layers=retrieval_layers,
            use_timestep_pe=use_timestep_pe,
            fusion_type=fusion_type,
            consolidate_type=consolidate_type,
            per_token_size=per_token_size,
        )

        if use_bf16:
            logger.info("Converting model to bfloat16")
            vla = vla.to(torch.bfloat16)
        else:
            vla = vla.to(torch.float32)

        vla = vla.to(device).eval()
        logger.info("MemoryVLA loaded successfully")
        return vla

    def reset(self, episode_id: str) -> None:
        """Reset memory banks for a new episode."""
        self._active_episode = episode_id
        # Reset the model's memory banks eagerly so state is clean even if the
        # first infer step is not step 0 or the client never issues step 0.
        try:
            self.vla.cog_mem_bank.reset()
            self.vla.per_mem_bank.reset()
            self.vla.cur_timestep = 0
        except AttributeError:
            # Some memory bank implementations expose reset() via the outer model only.
            reset = getattr(self.vla, "reset", None)
            if callable(reset):
                reset()

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference: convert observations, call predict_action, return actions."""
        if request.episode_id != self._active_episode:
            from bench.policy.protocol.errors import ErrorCode, ProtocolError
            raise ProtocolError(
                ErrorCode.EPISODE_MISMATCH,
                f"Reset episode {request.episode_id!r} before infer",
                request_id=request.request_id,
            )

        obs = request.observation

        # The model was trained on robot0_agentview_left (primary camera).
        # Use the first camera available; fallback: agentview_left.
        camera = "agentview_left"
        if camera not in obs.images:
            camera = next(iter(obs.images))

        # Convert np.ndarray (HWC, uint8) to PIL Image
        rgb_array = np.asarray(obs.images[camera])
        pil_image = Image.fromarray(rgb_array)

        # The first step of the episode triggers memory bank reset
        ep_first = "True" if request.step == 0 else "False"

        start = time.monotonic()
        actions, _ = self.vla.predict_action(
            image=pil_image,
            instruction=obs.prompt,
            unnorm_key=self.unnorm_key,
            cfg_scale=self.cfg_scale,
            use_ddim=self.use_ddim,
            num_ddim_steps=self.num_ddim_steps,
            episode_first_frame=ep_first,
        )
        model_ms = (time.monotonic() - start) * 1000.0

        # actions shape: [future_window+1, action_dim] e.g. [16, 13]
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]

        # Validate and return
        actions = validate_action_chunk(
            actions,
            expected_horizon=self.contract.action.horizon,
        )

        return InferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            actions=actions,
            timing={"model_ms": model_ms},
        )

    def close(self) -> None:
        self._active_episode = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_step_from_name(name: str) -> int | None:
    """Extract step number from checkpoint filename like ``step-050000-epoch-14-loss=0.0032.pt``."""
    import re
    m = re.search(r"step-(\d+)", name)
    return int(m.group(1)) if m else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve MemoryVLA checkpoint over membench-policy/v2 WebSocket",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the .pt checkpoint file (e.g. runs/memvla_wr03/checkpoints/step-050000-*.pt)",
    )
    parser.add_argument("--checkpoint-uri", help="Optional URI override for the checkpoint identity")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--cfg-scale", type=float, default=1.5, help="CFG scale (default: 1.5)")
    parser.add_argument("--use-bf16", action="store_true", help="Use bfloat16 inference")
    parser.add_argument("--no-ddim", action="store_true", help="Use DDPM instead of DDIM sampling")
    parser.add_argument("--num-ddim-steps", type=int, default=10, help="Number of DDIM steps (default: 10)")
    parser.add_argument("--api-key-env", default="MEMBENCH_POLICY_API_KEY", help="Env var for API key")
    parser.add_argument("--inference-timeout-s", type=float, default=60.0, help="Inference timeout (default: 60)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated model access")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()

    logger.info("Loading MemoryVLA from %s (device=%s)", args.checkpoint, args.device)
    backend = MemoryVLABackend.from_checkpoint(
        args.checkpoint,
        device=args.device,
        cfg_scale=args.cfg_scale,
        use_ddim=not args.no_ddim,
        num_ddim_steps=args.num_ddim_steps,
        use_bf16=args.use_bf16,
        checkpoint_uri=args.checkpoint_uri,
        code_revision={
            "membench": _git_revision(ROOT),
            "memvla": _git_revision(MEMVLA_SRC),
        },
        hf_token=args.hf_token,
    )

    logger.info(
        "MemoryVLA ready: horizon=%d, unnorm_key=%s, action_dim=%d",
        backend.contract.action.horizon,
        backend.unnorm_key,
        backend.contract.action.dim,
    )

    # Print a reminder about the suite config
    _print_suite_hint(backend.contract.action.horizon)

    server = WebsocketPolicyServer(
        backend,
        host=args.host,
        port=args.port,
        api_key=os.environ.get(args.api_key_env),
        inference_timeout_s=args.inference_timeout_s,
    )
    server.serve_forever()


def _print_suite_hint(horizon: int) -> None:
    """Print a reminder about matching the eval suite's replan_steps to the model's horizon."""
    print(
        f"\n  [IMPORTANT] The model's action horizon is {horizon}. "
        f"Create a copy of your eval suite YAML with:\n"
        f"    rollout:\n"
        f"      replan_steps: {horizon}\n"
        f"  and use that copy for evaluation.\n"
    )


def _git_revision(path: Path) -> str:
    import subprocess
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


if __name__ == "__main__":
    main()