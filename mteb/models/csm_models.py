from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, CsmForConditionalGeneration

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class CsmWrapper(Wrapper):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device

        self.model = CsmForConditionalGeneration.from_pretrained(model_name).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_rate = 24000  # CSM uses 24kHz audio

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

                # Create a conversation context with just the audio
                conversation_batch = []
                for audio_sample in batch_tensor:
                    conversation = [
                        {
                            "role": "0",  # Use speaker ID 0
                            "content": [
                                {"type": "audio", "path": audio_sample.cpu().numpy()}
                            ],
                        }
                    ]
                    conversation_batch.append(conversation)

                # Process through CSM
                inputs = self.processor.apply_chat_template(
                    conversation_batch,
                    tokenize=True,
                    return_dict=True,
                ).to(self.device)

                # Get encoder hidden states from the backbone decoder
                outputs = self.model.backbone_decoder(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
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


csm_1b = ModelMeta(
    loader=partial(
        CsmWrapper,
        model_name="sesame/csm-1b",
    ),
    name="sesame/csm-1b",
    languages=["eng-Latn"],
    open_weights=True,
    revision="ce2aeb48f203aa7f3ff4b228bc50c1af1a909bf7",  # Use specific revision for reproducibility
    release_date="2024-07-03",
    max_tokens=None,
    n_parameters=1_200_000_000,  # 1.2B parameters
    memory_usage_mb=4800,  # Approximate memory usage
    embed_dim=1024,  # Typical hidden dimension
    license="mit",
    reference="https://huggingface.co/sesame/csm-1b",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/kyutai/sesame",
    public_training_data=None,
    training_datasets=None,
    modalities=["audio"],
)
