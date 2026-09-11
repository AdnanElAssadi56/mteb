"""Smoke-test model wrappers before spending GPU hours on them.

For each model, loads it and encodes one tiny synthetic batch per modality it
claims to support. Catches the class of bug that silently ruins a campaign:
wrappers that crash on a modality, return the wrong shape, or return degenerate
embeddings (all-zero, NaN, or identical for different inputs).

    python scripts/moeb_smoke.py --models Qwen/Qwen2.5-Omni-3B --device cuda
    python scripts/moeb_smoke.py --omni --device cuda --out smoke.json

Exit code is non-zero if any model fails, so it can gate a campaign.
"""

from __future__ import annotations

import argparse
import json
import traceback
from typing import Any

import numpy as np

import mteb

MODALITIES = ("text", "image", "audio", "video")


def _probe_batch(modality: str, n: int = 4) -> list[dict[str, Any]]:
    """Two distinguishable items per modality, so we can tell embeddings apart."""
    rng = np.random.default_rng(0)
    if modality == "text":
        return [{"text": t} for t in ("a dog barking", "a violin solo")][:n]
    if modality == "image":
        from PIL import Image

        return [
            {"image": Image.fromarray(
                rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) if i else
                np.zeros((224, 224, 3), dtype=np.uint8)
            )}
            for i in range(2)
        ]
    if modality == "audio":
        sr = 16_000
        t = np.linspace(0, 2, 2 * sr, dtype=np.float32)
        return [
            {"audio": {"array": np.sin(2 * np.pi * f * t), "sampling_rate": sr}}
            for f in (220.0, 880.0)
        ]
    if modality == "video":
        import torch

        return [
            {"video": torch.from_numpy(
                rng.integers(0, 255, (8, 3, 224, 224), dtype=np.uint8)
            )}
            for _ in range(2)
        ]
    raise ValueError(modality)


_TASK_CACHE: dict[str, Any] = {}


def _task_for(modality: str):
    """A task whose declared modalities match, so the dataloader keeps the column."""
    if modality not in _TASK_CACHE:
        for task in mteb.get_tasks(exclude_beta=False):
            if set(task.metadata.modalities or []) == {modality}:
                _TASK_CACHE[modality] = task
                break
        else:
            raise RuntimeError(f"no single-modality task found for {modality!r}")
    return _TASK_CACHE[modality]


def _check_embeddings(emb: Any) -> str | None:
    """Return a failure reason, or None if the embeddings look usable."""
    arr = np.asarray(emb)
    if arr.ndim < 2:
        return f"expected 2-D embeddings, got shape {arr.shape}"
    if arr.shape[0] < 2:
        return f"expected one row per input, got shape {arr.shape}"
    if not np.isfinite(arr).all():
        return "embeddings contain NaN or inf"
    if np.allclose(arr, 0):
        return "all-zero embeddings"
    a, b = arr[0].ravel(), arr[1].ravel()
    if np.allclose(a, b):
        return "identical embeddings for different inputs (encoder likely ignored input)"
    return None


def smoke_one(name: str, device: str, batch_size: int) -> dict[str, Any]:
    result: dict[str, Any] = {"model": name, "modalities": {}, "load_error": None}
    try:
        meta = mteb.get_model_meta(name)
        declared = [m for m in MODALITIES if m in (meta.modalities or [])]
        result["declared"] = declared
        model = mteb.get_model(name, device=device)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        result["load_error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=3)
        return result

    for modality in declared:
        entry: dict[str, Any] = {}
        try:
            from mteb._create_dataloaders import create_dataloader
            from datasets import Dataset

            rows = _probe_batch(modality)
            task = _task_for(modality)
            ds = Dataset.from_list(rows)
            loader = create_dataloader(
                ds, task_metadata=task.metadata, batch_size=batch_size
            )
            emb = model.encode(
                loader, task_metadata=task.metadata, hf_split="test", hf_subset="default"
            )
            problem = _check_embeddings(emb)
            entry["shape"] = list(np.asarray(emb).shape)
            entry["ok"] = problem is None
            if problem:
                entry["problem"] = problem
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["problem"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc(limit=3)
        result["modalities"][modality] = entry
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", help="model names to smoke-test")
    p.add_argument("--omni", action="store_true",
                   help="test every model declaring all four modalities")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--out", default=None, help="write JSON report here")
    args = p.parse_args()

    names = list(args.models or [])
    if args.omni:
        names += [
            m.name for m in mteb.get_model_metas()
            if set(MODALITIES) <= set(m.modalities or [])
            and not m.name.startswith("mteb/baseline")
        ]
    names = sorted(dict.fromkeys(names))
    if not names:
        p.error("pass --models and/or --omni")

    reports, failures = [], 0
    for name in names:
        rep = smoke_one(name, args.device, args.batch_size)
        reports.append(rep)
        if rep["load_error"]:
            failures += 1
            print(f"FAIL  {name}\n        load: {rep['load_error']}", flush=True)
            continue
        bad = [m for m, e in rep["modalities"].items() if not e.get("ok")]
        status = "FAIL" if bad else "ok  "
        failures += bool(bad)
        detail = " ".join(
            f"{m}:{'ok' if e.get('ok') else 'X'}" for m, e in rep["modalities"].items()
        )
        print(f"{status}  {name:48s} {detail}", flush=True)
        for m in bad:
            print(f"        {m}: {rep['modalities'][m]['problem']}", flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(reports, fh, indent=2, default=str)
        print(f"\nreport -> {args.out}")
    print(f"\n{len(names) - failures}/{len(names)} models passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
