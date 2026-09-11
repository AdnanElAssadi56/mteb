from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm.auto import tqdm

from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.modality_collators import AudioCollator

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType
    from mteb.types._encoder_io import AudioInput


class CNN14Wrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # NOTE: CNN14 is trained on 10-second clips (PANNs pads/crops AudioSet to 10 s, arXiv:1912.10211), so a
        # 10 s cap looks "more native" than 30 s -- but lowering a cap
        # removes signal, and that reasoning has already failed once in this
        # file's history: setting CNN14's rate to the value its config declares
        # measurably hurt (see cnn14_model.py). Left at 30 s until A/B'd on a task
        # with clips longer than 10 s; the two small tasks available here
        # are all-short and cannot detect the difference.
        max_audio_length_s: float = 30.0,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.max_audio_length_s = max_audio_length_s

        from speechbrain.inference.classifiers import AudioClassifier

        # Load the SpeechBrain model
        self.model = AudioClassifier.from_hparams(
            source=model_name,
            savedir="pretrained_models/cnn14-esc50",
            run_opts={"device": device},
        )

        # 44.1 kHz, matching the checkpoint. Evidence, including the parts that
        # point the other way, because this one is genuinely mixed:
        #
        # For 44.1 kHz (decisive):
        #  - hyperparams.yaml declares `sample_rate: 44100`, and the mel frontend
        #    (n_fft 1024, hop 11.61 ms, win 23.22 ms) is derived from it.
        #  - The authors' own `example_dogbark.wav` is 44.1 kHz.
        #  - Running the authors' own sanity check, the model's classifier head
        #    predicts "dog" when fed 44.1 kHz and "hand_saw" when fed the same
        #    audio resampled to 16 kHz. Wrong rate => wrong prediction.
        #
        # Against (why this looks tempting to revert): mteb's audio classification
        # does not use that head -- it trains a fresh probe on embeddings, and
        # mangled-but-consistent features can still probe well. Measured on CPU at
        # mteb 2.20.12 (accuracy, 16 kHz vs 44.1 kHz):
        #    GunshotTriangulation (n=88)   0.8320  vs  0.5333   <- favours 16 kHz
        #    BeijingOpera         (n=236)  0.8088  vs  0.7458   <- favours 16 kHz
        #    FSDD                 (n=300)  0.1523  vs  0.2900   <- favours 44.1 kHz
        #    VoxPopuliGenderID    (n=500)  0.6920  vs  0.7360   <- favours 44.1 kHz
        # The two tasks preferring 16 kHz are the two smallest. We follow the rate
        # the model actually works at rather than the one that probes higher on
        # small tasks.
        #
        # NOTE: published CNN14 results (e.g. BeijingOpera 0.8386 at mteb 2.4.2)
        # were produced at 16 kHz and are NOT comparable to runs after this change.
        self.sampling_rate = 44_100

    def _pad_audio_batch(self, batch: list[torch.Tensor]) -> torch.Tensor:  # noqa: PLR6301
        max_len = max(w.shape[0] for w in batch)
        padded = [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in batch]
        return torch.stack(padded)

    def get_audio_embeddings(
        self,
        inputs: DataLoader[AudioInput],
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> Array:
        inputs.collate_fn = AudioCollator(target_sampling_rate=self.sampling_rate)

        all_embeddings = []

        for batch in tqdm(
            inputs,
            disable=not show_progress_bar,
        ):
            audio_tensors = []
            for a in batch["audio"]:
                array = torch.tensor(a["array"], dtype=torch.float32)

                array = array.squeeze()

                # Apply audio truncation (configurable limit)
                max_length = int(self.max_audio_length_s * self.sampling_rate)
                if array.shape[-1] > max_length:
                    array = array[..., :max_length]

                audio_tensors.append(array)

            with torch.no_grad():
                # Convert batch to tensors and move to device
                batch_tensor = self._pad_audio_batch(audio_tensors).to(self.device)

                feats = self.model.mods.compute_features(batch_tensor)
                b, f, t = feats.shape
                if f < 64 or t < 80:
                    # zero-pad in the frequency or time dimension until it's at least [64, 80]
                    pad_freq = max(0, 64 - f)
                    pad_time = max(0, 80 - t)
                    feats = torch.nn.functional.pad(feats, (0, pad_time, 0, pad_freq))
                embeddings = self.model.mods.embedding_model(feats)
                # Apply mean pooling over time dimension if needed
                if embeddings.dim() > 2:
                    embeddings = torch.mean(embeddings, dim=1)

                all_embeddings.append(embeddings.cpu().detach())

        return torch.cat(all_embeddings, dim=0).numpy()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        if "audio" not in inputs.dataset.features:
            raise ValueError("ASTWrapper only supports audio inputs.")
        return self.get_audio_embeddings(inputs, **kwargs)


cnn14_esc50 = ModelMeta(
    loader=CNN14Wrapper,
    name="speechbrain/cnn14-esc50",
    languages=["eng-Latn"],
    open_weights=True,
    revision="422a112e9a22a5fac0d37571aacaee5caf154395",
    release_date="2022-11-26",
    max_tokens=None,
    n_parameters=80_753_615,
    n_embedding_parameters=0,
    memory_usage_mb=308,
    embed_dim=2048,
    license="apache-2.0",
    reference="https://huggingface.co/speechbrain/cnn14-esc50",
    similarity_fn_name="cosine",
    framework=["PyTorch"],
    use_instructions=False,
    public_training_code="https://github.com/speechbrain/speechbrain",
    public_training_data=None,
    training_datasets={
        "ESC50",
        # "VGGSound",  # not in MTEB
    },
    modalities=["audio"],
    citation="""
@inproceedings{wang2022CRL,
    title={Learning Representations for New Sound Classes With Continual Self-Supervised Learning},
    author={Zhepei Wang, Cem Subakan, Xilin Jiang, Junkai Wu, Efthymios Tzinis, Mirco Ravanelli, Paris Smaragdis},
    year={2022},
    booktitle={Accepted to IEEE Signal Processing Letters}
}
""",
    extra_requirements_groups=["speechbrain"],
)
