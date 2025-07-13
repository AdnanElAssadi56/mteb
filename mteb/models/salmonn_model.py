from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from mteb.encoder_interface import AudioBatch, AudioData, PromptType
from mteb.model_meta import ModelMeta
from mteb.models.wrapper import Wrapper


class SalmonnWrapper(Wrapper):
    def __init__(
        self,
        model_name: str = "tsinghua-ee/SALMONN-7B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        
        # Load SALMONN model and processor
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # SALMONN typically uses 16kHz sampling rate
        self.sampling_rate = getattr(self.processor, 'sampling_rate', 16000)
        
        # Set model to evaluation mode
        self.model.eval()

    def _process_audio(self, audio: AudioBatch) -> list[torch.Tensor]:
        """Process audio batch similar to CLAP wrapper"""
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
        """Handle different batch formats"""
        waveforms = []

        if isinstance(batch, tuple):  # Handle (audio, metadata) tuples
            for audio, _ in batch:
                waveforms.append(self._convert_audio(audio))
        else:
            for item in batch:
                if isinstance(item, dict):
                    if "array" in item:
                        audio = item["array"]
                        # Convert to torch tensor and ensure float32
                        audio = (
                            torch.from_numpy(audio).float()
                            if isinstance(audio, np.ndarray)
                            else audio.float()
                        )
                        if item.get("sampling_rate", self.sampling_rate) != self.sampling_rate:
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
        """Convert audio to proper tensor format"""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)
        return audio.squeeze().float()

    def _load_audio_file(self, path: str) -> torch.Tensor:
        """Load audio file and resample if necessary"""
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def get_audio_embeddings(
        self,
        audio: AudioBatch,
        **kwargs: Any,
    ) -> np.ndarray:
        """Extract audio embeddings using SALMONN"""
        all_features = []

        if isinstance(audio, DataLoader):
            # Process all batches
            for batch in tqdm(audio, desc="Processing audio batches"):
                batch_features = []
                for item in batch:
                    if isinstance(item, torch.Tensor):
                        item = {"array": item.numpy()}
                    
                    # Process audio through SALMONN
                    audio_tensor = torch.from_numpy(item["array"]).float()
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    
                    audio_tensor = audio_tensor.to(self.device)
                    
                    with torch.no_grad():
                        # Use SALMONN's audio encoder to get embeddings
                        audio_features = self.model.encode_audio(audio_tensor)
                        
                        # Normalize embeddings
                        audio_features = audio_features / audio_features.norm(
                            dim=-1, keepdim=True
                        )
                        batch_features.append(audio_features.cpu().numpy())

                all_features.extend(batch_features)

            return np.vstack(all_features)
        else:
            # Process single batch
            batch_features = []
            for item in audio:
                audio_tensor = torch.from_numpy(item["array"]).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                audio_tensor = audio_tensor.to(self.device)
                
                with torch.no_grad():
                    audio_features = self.model.encode_audio(audio_tensor)
                    audio_features = audio_features / audio_features.norm(
                        dim=-1, keepdim=True
                    )
                    batch_features.append(audio_features.cpu().numpy())

            return np.vstack(batch_features)

    def get_text_embeddings(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> np.ndarray:
        """Extract text embeddings using SALMONN"""
        all_features = []
        
        # Process texts in batches to avoid memory issues
        batch_size = kwargs.get('batch_size', 8)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize texts
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Use SALMONN's text encoder to get embeddings
                text_features = self.model.encode_text(**inputs)
                
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_features.append(text_features.cpu().numpy())
        
        return np.vstack(all_features)

    def encode(
        self,
        inputs: AudioBatch | list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Unified encoding interface"""
        if isinstance(inputs[0], str):
            return self.get_text_embeddings(inputs, **kwargs)
        return self.get_audio_embeddings(inputs, **kwargs)

    def compute_similarity(
        self,
        audio_embeddings: np.ndarray,
        text_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between audio and text embeddings"""
        # Normalize embeddings
        audio_embeddings = audio_embeddings / np.linalg.norm(
            audio_embeddings, axis=-1, keepdims=True
        )
        text_embeddings = text_embeddings / np.linalg.norm(
            text_embeddings, axis=-1, keepdims=True
        )
        
        # Compute cosine similarity
        return np.dot(audio_embeddings, text_embeddings.T)


# Model metadata for SALMONN
salmonn_7b = ModelMeta(
    loader=partial(SalmonnWrapper, model_name="tsinghua-ee/SALMONN-7B"),
    name="tsinghua-ee/SALMONN-7B",
    languages=["eng-Latn"],
    revision="ed583f2c565c1110b3bf5772ed42d6744f3cc443",  # Update with specific revision if needed
    release_date="2023-10-01",  # Update with actual release date
    modalities=["audio", "text"],
    n_parameters=7_000_000_000,  # Approximate for 7B model
    memory_usage_mb=14000,  # Approximate for 7B model
    max_tokens=512,  # Typical max sequence length
    embed_dim=4096,  # Typical embedding dimension for 7B model
    license="apache-2.0",  # Update with actual license
    open_weights=True,
    public_training_code="https://github.com/bytedance/SALMONN",
    public_training_data="Mixed audio-text datasets",  # Update with specifics
    framework=["PyTorch"],
    reference="https://huggingface.co/tsinghua-ee/SALMONN-7B",
    similarity_fn_name="cosine",
    use_instructions=False,
    training_datasets={
        # Add specific training datasets if known
    },
)
