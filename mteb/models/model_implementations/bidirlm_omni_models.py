from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator, VideoCollator
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing_extensions import Unpack

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs

from .bidirlm_models import (
    BIDIRLM_CITATION,
    BIDIRLM_LANGUAGES,
    bidirlm_task_prompts,
    bidirlm_training_data,
)

BIDIRLM_OMNI_TRAINING_DATASETS = bidirlm_training_data | {
    "LAION-Audio-300M",
    "MS_COCO",
    "colpali_train_set",
    "natcap",
    "librispeech_asr",
}


TASK_PROMPTS: dict[str, str | dict[str, str]] = {
    **bidirlm_task_prompts,
    # MTEB tasks
    "ArguAna": {
        "query": "Given a claim, retrieve documents that support or refute the claim",
        "document": "Given a claim, retrieve documents that support or refute the claim",
    },
    "CQADupstackGamingRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    "CQADupstackUnixRetrieval": {
        "query": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
        "document": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question",
    },
    # MIEB tasks
    "AROCocoOrder": "Compositionality Evaluation of images to their captions.Each capation has four hard negatives created by order permutations.",
    "AROFlickrOrder": "Compositionality Evaluation of images to their captions.Each capation has four hard negatives created by order permutations.",
    "BLINKIT2IMultiChoice": "Retrieve images based on images and specific retrieval instructions.",
    "Country211ZeroShot": "Classifying images of 211 countries.",
    "CVBenchRelation": "decide the relation of the objects in the image.",
    "FER2013ZeroShot": "Classifying facial emotions.",
    "VidoreDocVQARetrieval": "Retrieve associated pages according to questions.",
    "VidoreShiftProjectRetrieval": "Retrieve associated pages according to questions.",
    "VidoreSyntheticDocQAAIRetrieval": "Retrieve associated pages according to questions.",
    "VidoreTabfquadRetrieval": "Retrieve associated pages according to questions.",
    "VidoreTatdqaRetrieval": "Retrieve associated pages according to questions.",
    "VQA2IT2TRetrieval": "Retrieve the correct answer for a question about an image.",
    "WebQAT2ITRetrieval": "Retrieve sources of information based on questions.",
    "WITT2IRetrieval": "Retrieve images based on multilingual descriptions.",
    "XM3600T2IRetrieval": "Retrieve images based on multilingual descriptions.",
    # MAEB tasks
    "CommonLanguageAgeDetection": "Age Classification. This is a stratified subsampled version of the original CommonLanguage dataset.",
    "CommonVoiceMini21T2ARetrieval": "Speech recordings with corresponding text transcriptions from CommonVoice dataset.",
    "FSD2019Kaggle": "Multilabel Audio Classification.",
    "JamAltArtistA2ARetrieval": "Given audio clip of a song (query), retrieve all songs from the same artist in the Jam-Alt-Lines dataset.",
    "JamAltLyricA2TRetrieval": "From audio clips of songs (query), retrieve corresponding textual lyric from the Jam-Alt-Lines dataset.",
    "SpeechCommandsZeroshotv0.02": "Sound Classification/Keyword Spotting Dataset. This is a set of one-second audio clips containing a single spoken English word or background noise. These words are from a small set of commands such as 'yes', 'no', and 'stop' spoken by various speakers. With a total of 10 labels/commands for keyword spotting and a total of 30 labels for other auxiliary tasks.",
    "VehicleSoundClustering": "Clustering vehicle sounds recorded from smartphones (0 (car class), 1 (truck, bus and van class), 2 (motorcycle class)).",
    "VoxPopuliAccentPairClassification": "Classifying same or different regional accent of English.",
}

POOLING_DIM = 2048


class BidirLMOmniEncoder(AbsEncoder):
    """MTEB-compatible multimodal encoder for BidirLM-Omni (text / image / audio)."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code: bool = True,
        max_text_length: int = 1024,
        # Frame budget: fps=2 is the vendor default -- Qwen ships `FPS = 2.0`
        # (with FPS_MIN_FRAMES=4, FPS_MAX_FRAMES=768) in qwen-vl-utils /
        # qwen-omni-utils, which is this family's own preprocessing path.
        # `max_frames` is an mteb-side cost cap well below Qwen's 768: video
        # encoding dominates benchmark runtime, and frames are sampled uniformly
        # across the whole clip, so the cap reduces temporal resolution rather
        # than coverage. Note the field has two conventions -- MVEB uses
        # fps=2/max 64 for variable-length models (arXiv:2606.14958) while
        # UVRB and MMEB-V3 force a uniform 8 frames on every model
        # (arXiv:2510.27571) -- so this is a defensible choice, not the only one.
        # https://github.com/QwenLM/Qwen2.5-Omni/blob/main/qwen-omni-utils/src/qwen_omni_utils/v2_5/vision_process.py
        fps: float | None = 2.0,
        max_frames: int | None = 64,
        num_frames: int | None = None,
        # SHARED BACKBONE, INCONSISTENT CAPS -- see note. This model's audio
        # tower is the Qwen2.5-Omni (Whisper-lineage) encoder, whose feature
        # pipeline maxes out at 1500 log-mel frames = 30 s, and whose docs advise
        # keeping clips "under 30 seconds". mteb currently gives the five
        # wrappers built on that same tower three different limits:
        #   qwen_omni_lm 300 s | bidirlm 30 s | jina 30 s | lco None | colqwen None
        # so scores on long-audio tasks partly reflect the wrapper, not the model.
        # Left as-is pending a check of whether the processor actually consumes
        # >30 s or silently truncates; do not "fix" one of these in isolation.
        # https://huggingface.co/docs/transformers/model_doc/qwen2_5_omni
        max_samples: int | None = 30 * 16_000,
        **kwargs: Any,
    ) -> None:
        from transformers import AutoVideoProcessor

        processor_kwargs = kwargs.get("processor_kwargs", {})

        # VideoCollator already samples frames; skip inner sampling.
        video_processor = AutoVideoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            do_sample_frames=False,
            **kwargs,
        )
        processor_kwargs |= {
            "video_processor": video_processor,
        }

        kwargs["processor_kwargs"] = processor_kwargs
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self.model.eval()
        self.max_text_length = max_text_length
        self.fps = fps
        self.max_frames = max_frames
        self.num_frames = num_frames

        self.task_prompts = TASK_PROMPTS
        self.sampling_rate = 16_000
        self.max_samples = max_samples

    def _get_instruction(
        self,
        task_metadata: TaskMetadata,
        prompt_type: PromptType | None,
    ) -> str | None:
        task_type = task_metadata.type

        if task_type == "Summarization":
            return None

        entry = self.task_prompts.get(task_metadata.name)

        # Asymmetric retrieval: documents get no instruction unless the prompt
        # dict explicitly provides a "document" key.
        if (
            task_metadata.simplified_task_type == "retrieval"
            and prompt_type == PromptType.document
            and not (isinstance(entry, dict) and "document" in entry)
        ):
            return None

        if entry is not None:
            if isinstance(entry, dict):
                key = prompt_type.value if prompt_type else "query"
                instruction = entry.get(key, entry.get("query")) or None
            else:
                instruction = entry
            return instruction

        if task_type in {"STS", "PairClassification"}:
            return "Retrieve semantically similar text"
        if task_type == "BitextMining":
            return "Retrieve parallel sentences"

        return None

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Unpack[EncodeKwargs],
    ) -> Array:
        """Implements AbsEncoder.encode with multimodal support (text, image, audio).

        Builds conversation messages from whichever modalities are present and
        delegates to SentenceTransformer.encode() via the native 'message' modality.
        """
        ds_features = inputs.dataset.features
        has_text = "text" in ds_features
        has_image = "image" in ds_features
        has_audio = "audio" in ds_features
        has_video = "video" in ds_features

        if has_video:
            inputs.collate_fn = VideoCollator(
                target_sampling_rate=self.sampling_rate,
                fps=self.fps,
                max_frames=self.max_frames,
                num_frames=self.num_frames,
                max_samples=self.max_samples,
            )
        elif has_audio:
            inputs.collate_fn = AudioCollator(
                target_sampling_rate=self.sampling_rate,
                max_samples=self.max_samples,
            )
        instruction = self._get_instruction(task_metadata, prompt_type)

        # Truncate only when the schema is pure text; for multimodal schemas
        # we keep the full context to avoid chopping off special tokens.
        is_text_only_schema = has_text and not (has_image or has_audio or has_video)
        self.model.max_seq_length = (
            self.max_text_length if is_text_only_schema else 32768
        )

        modality_keys = ("image", "audio", "text", "video")
        all_embeddings: list = []
        for batch in inputs:
            batch_size = len(next(iter(batch.values())))
            batch_inputs: list[dict[str, Any]] = []
            for i in range(batch_size):
                row = {
                    key: batch[key][i]
                    for key in modality_keys
                    if key in batch and batch[key][i] is not None
                }
                batch_inputs.append(row)

            embeddings = self.model.encode(
                batch_inputs,
                prompt=instruction,
                **kwargs,
            )
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().detach().float()
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)


bidirlm_omni_2_5b = ModelMeta(
    name="BidirLM/BidirLM-Omni-2.5B-Embedding",
    loader=BidirLMOmniEncoder,
    loader_kwargs=dict(
        trust_remote_code=True,
        max_text_length=1024,
    ),
    languages=BIDIRLM_LANGUAGES,
    open_weights=True,
    revision="4d8a7d34095ef8c4ab760744015825b809756dfe",
    release_date="2026-04-07",
    n_parameters=2_445_009_536,
    n_embedding_parameters=315_098_112,
    memory_usage_mb=4663,
    max_tokens=32768,
    embed_dim=POOLING_DIM,
    license="apache-2.0",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    modalities=["text", "image", "audio", "video"],
    model_type=["dense"],
    reference="https://huggingface.co/BidirLM/BidirLM-Omni-2.5B-Embedding",
    public_training_code=None,
    public_training_data="https://huggingface.co/datasets/BidirLM/BidirLM-Omni-Contrastive",
    training_datasets=BIDIRLM_OMNI_TRAINING_DATASETS,
    citation=BIDIRLM_CITATION,
)
