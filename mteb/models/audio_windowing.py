"""Window long audio and mean-pool, for encoders that declare no length limit.

wav2vec2, HuBERT, WavLM, data2vec, UniSpeech, MMS and SEW-D have convolutional
positional encoding, so only memory bounds the input. Every framework windows
and pools rather than truncating:

- pyannote.audio: window = the model's training duration (`duration or training_duration`)
- s3prl/SUPERB: 2000 frames = 20 s (UnfoldChunkByFrame)
- X-ARES: 10 s (example/data2vec/data2vec_encoder.py)
- HEAR: get_scene_embeddings = torch.mean() over per-timestamp embeddings
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mteb.types import Array


def split_into_windows(
    arrays: list[Array],
    window_samples: int | None,
    min_samples: int = 1,
) -> tuple[list[Array], list[int]]:
    """Split each waveform into non-overlapping windows.

    Args:
        arrays: One waveform per clip.
        window_samples: Window length in samples, or None to disable windowing.
        min_samples: Drop a trailing window shorter than this, unless it is the
            clip's only window.

    Returns:
        The windows across all clips, and the clip index each window came from.
    """
    windows: list[Array] = []
    owner: list[int] = []
    for index, array in enumerate(arrays):
        arr = np.asarray(array)
        if window_samples is None or arr.shape[-1] <= window_samples:
            windows.append(arr)
            owner.append(index)
            continue
        for start in range(0, arr.shape[-1], window_samples):
            window = arr[..., start : start + window_samples]
            if start and window.shape[-1] < min_samples:
                continue  # drop a short tail, but never the first window
            windows.append(window)
            owner.append(index)
    return windows, owner


def pool_windows(embeddings: Array, owner: list[int], n_clips: int) -> Array:
    """Mean-pool the window embeddings belonging to each clip.

    Args:
        embeddings: One embedding per window, ordered as `split_into_windows`.
        owner: Clip index for each window.
        n_clips: Number of clips the windows came from.

    Returns:
        One embedding per clip, in the original clip order.
    """
    emb = np.asarray(embeddings)
    if len(owner) != len(emb):
        raise ValueError(f"got {len(emb)} embeddings for {len(owner)} windows")
    index = np.asarray(owner)
    counts = np.bincount(index, minlength=n_clips)
    if (counts == 0).any():
        raise ValueError("every clip must produce at least one window")
    summed = np.zeros((n_clips, emb.shape[-1]), dtype=emb.dtype)
    np.add.at(summed, index, emb)
    return summed / counts[:, None]
