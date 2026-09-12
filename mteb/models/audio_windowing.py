"""Windowed encoding for audio models that declare no maximum input length.

Several speech encoders (wav2vec2, HuBERT, WavLM, data2vec, UniSpeech, MMS,
SEW-D) declare no input-length limit: their positional encoding is convolutional,
so the architecture is length-agnostic and only memory bounds the input. Feeding
a long clip whole is therefore quadratic in attention and OOMs, while truncating
it discards most of the audio.

Every framework that evaluates these models windows the clip and pools instead:

- pyannote.audio sets the window to the model's own training duration
  (`duration = duration or training_duration`) and slides it with overlap.
- s3prl / SUPERB chunk at 2000 frames (20 s at a 10 ms shift).
- X-ARES reference encoders split at 10 s and concatenate the frame embeddings.
- HEAR defines `get_scene_embeddings` as `torch.mean()` over the per-timestamp
  embeddings of the whole clip.

This module implements that: split into windows, encode each, mean-pool the
windows belonging to one clip. Windows are non-overlapping, which matches s3prl
and X-ARES; pyannote's overlap matters for frame-level tasks, not for a single
clip embedding.

Following pyannote's rule, each wrapper sets its window to the duration the model
was actually trained on -- see the reference on each wrapper's `window_seconds`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mteb.types import Array

__all__ = ["pool_windows", "split_into_windows"]


def split_into_windows(
    arrays: list[Array],
    window_samples: int | None,
    min_samples: int = 1,
) -> tuple[list[Array], list[int]]:
    """Split each waveform into non-overlapping windows.

    Args:
        arrays: One waveform per clip.
        window_samples: Window length in samples. ``None`` disables windowing,
            in which case each clip is returned unchanged.
        min_samples: Drop a trailing window shorter than this, unless it is the
            only window for that clip (in which case it is kept and the caller's
            padding applies).

    Returns:
        ``(windows, owner)`` where ``windows`` is the flattened list of windows
        across all clips and ``owner[i]`` is the index of the clip that
        ``windows[i]`` came from.
    """
    windows: list[Array] = []
    owner: list[int] = []
    for index, array in enumerate(arrays):
        arr = np.asarray(array)
        n = arr.shape[-1]
        if window_samples is None or n <= window_samples:
            windows.append(arr)
            owner.append(index)
            continue
        produced = 0
        for start in range(0, n, window_samples):
            window = arr[..., start : start + window_samples]
            if window.shape[-1] < min_samples and produced:
                continue  # drop a too-short tail, but never drop the only window
            windows.append(window)
            owner.append(index)
            produced += 1
    return windows, owner


def pool_windows(
    embeddings: Array,
    owner: list[int],
    n_clips: int,
    normalize: bool = False,
) -> Array:
    """Mean-pool the window embeddings belonging to each clip.

    Args:
        embeddings: One embedding per window, in the order produced by
            :func:`split_into_windows`.
        owner: Clip index for each window.
        n_clips: Number of clips the windows came from.
        normalize: L2-normalise each pooled embedding. Use this when the model's
            own embeddings are normalised, so that pooling does not shrink them.

    Returns:
        One embedding per clip, in the original clip order.
    """
    emb = np.asarray(embeddings)
    if len(owner) != emb.shape[0]:
        raise ValueError(f"got {emb.shape[0]} embeddings for {len(owner)} windows")
    pooled = np.zeros((n_clips, *emb.shape[1:]), dtype=emb.dtype)
    counts = np.zeros(n_clips, dtype=np.int64)
    for row, index in enumerate(owner):
        pooled[index] += emb[row]
        counts[index] += 1
    if (counts == 0).any():
        raise ValueError("every clip must produce at least one window")
    pooled /= counts.reshape(-1, *([1] * (pooled.ndim - 1)))
    if normalize:
        norms = np.linalg.norm(pooled, axis=-1, keepdims=True)
        pooled /= np.maximum(norms, 1e-12)
    return pooled
