from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .seci_net import HybridTextClassifier


CHECKPOINT_FORMAT_VERSION = 2


def save_checkpoint_bundle(
    checkpoint_path: str | Path,
    model: HybridTextClassifier,
    vocab: dict[str, Any],
    label_to_index: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": model.architecture_name,
        "state_dict": model.state_dict(),
        "model_config": model.get_model_config(),
        "vocab": vocab,
        "label_to_index": label_to_index,
        "extra": extra or {},
    }
    torch.save(payload, Path(checkpoint_path))
    return payload


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location=map_location or "cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload or "model_config" not in payload:
        raise ValueError(
            "Checkpoint is not a self-describing SECI-Net v2 bundle. Please retrain with main/train.py."
        )
    return payload


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[HybridTextClassifier, dict[str, Any]]:
    payload = load_checkpoint_bundle(checkpoint_path, map_location=map_location)
    model = HybridTextClassifier(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
