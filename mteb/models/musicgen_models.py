from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MusicgenForConditionalGeneration, AutoProcessor

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class MusicgenWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device

        self.model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_rate = 32000  # MusicGen uses 32kHz audio

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
        processed_audio = []

        if isinstance(audio, DataLoader):
            for batch in audio:
                processed_audio.extend(self._handle_batch(batch))
        else:
            processed_audio = self._handle_batch(audio)

        return processed_audio

    def _handle_batch(
        self, batch: AudioData | Iterable[tuple[AudioData, str]]
    ) -> list[torch.Tensor]:
        waveforms = []

        if isinstance(batch, tuple):
            for audio, _ in batch:
                waveforms.append(self._convert_audio(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        audio = item["array"]
                        audio = (
                            torch.from_numpy(audio).float()
                            if isinstance(audio, np.ndarray)
                            else audio.float()
                        )
                        if item["sampling_rate"] != self.sampling_rate:
                            resampler = torchaudio.transforms.Resample(
                                item["sampling_rate"], self.sampling_rate
                            )
                            audio = resampler(audio)
                        waveforms.append(self._convert_audio(audio))
                    elif "path" in item:
                        waveforms.append(self._load_audio_file(item["path"]))
                elif isinstance(item, (np.ndarray, torch.Tensor)):
                    waveforms.append(self._convert_audio(item))
                elif isinstance(item, str):
                    waveforms.append(self._load_audio_file(item))

        return waveforms

    def _convert_audio(self, audio: AudioData) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def _pad_audio_batch(self, batch: list[torch.Tensor]) -> torch.Tensor:
        max_length = max(audio.shape[0] for audio in batch)
        padded_batch = [
            torch.nn.functional.pad(audio, (0, max_length - audio.shape[0]))
            for audio in batch
        ]
        return torch.stack(padded_batch)

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 4,
        **kwargs: Any,
    ) -> torch.Tensor:
        processed_audio = self._process_audio(audio)
        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(processed_audio), batch_size)):
                batch = processed_audio[i : i + batch_size]
                batch_tensor = self._pad_audio_batch(batch)

                print(batch_tensor.shape)

                # Convert each row of batch_tensor into a 1D numpy array
                audio_list: list[np.ndarray] = []
                np_batch = batch_tensor.cpu().numpy()  # (B, T)
                for arr in np_batch:
                    if arr.ndim > 1:
                        arr = arr.squeeze()
                    audio_list.append(arr)

                # Pass a list of 1D arrays into the processor
                inputs = self.processor(
                    audio=audio_list,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Extract embeddings from the decoder
                outputs = self.model.encoder(
                    inputs.input_values,
                    output_hidden_states=True,
                    return_dict=True,
                )
                
                # Use last hidden state for embeddings
                last_hidden_state = outputs.hidden_states[-1]
                # Mean pooling over sequence dimension
                embeddings = torch.mean(last_hidden_state, dim=1)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode(
        self,
        inputs: AudioBatch,
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.get_audio_embeddings(inputs, task_name=task_name, **kwargs).numpy()


musicgen_small = ModelMeta(
    loader=partial(
        MusicgenWrapper,
        model_name="facebook/musicgen-small",
    ),
    name="facebook/musicgen-small",
    languages=["eng-Latn"],
    open_weights=True,
    revision="4c8334b02c6ec4e8664a91979669a501ec497792",
    release_date="2023-06-09",
    max_tokens=None,
    n_parameters=300_000_000,  # 300M parameters
    memory_usage_mb=1200,  # Approximate memory usage
    embed_dim=1024,  # Hidden dimension
    license="mit",
    reference="https://huggingface.co/facebook/musicgen-small",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/audiocraft",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
) 

musicgen_medium = ModelMeta(
    loader=partial(
        MusicgenWrapper,
        model_name="facebook/musicgen-medium",
    ),
    name="facebook/musicgen-medium",
    languages=["eng-Latn"],
    open_weights=True,
    revision="d3bd7b00761b78ad7a8a05145ee31e7832e9916c",
    release_date="2023-06-09",
    max_tokens=None,
    n_parameters=1_500_000_000,  # 1.5B parameters
    memory_usage_mb=5800,  # Approximate memory usage
    embed_dim=1024,  # Hidden dimension
    license="mit",
    reference="https://huggingface.co/facebook/musicgen-medium",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/facebookresearch/audiocraft",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
) 