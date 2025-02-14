from __future__ import annotations

from functools import partial

from mteb.model_meta import (
    ModelMeta,
    ScoringFunction,
    sentence_transformers_loader,
)

mxbai_embed_large_v1 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        revision="990580e27d329c7408b3741ecff85876e128e203",
        model_prompts={
            "query": "Represent this sentence for searching relevant passages: "
        },
    ),
    name="mixedbread-ai/mxbai-embed-large-v1",
    languages=["eng_Latn"],
    open_weights=True,
    revision="990580e27d329c7408b3741ecff85876e128e203",
    release_date="2024-03-07",  # initial commit of hf model.
    n_parameters=335_000_000,
    memory_usage_mb=639,
    max_tokens=512,
    embed_dim=1024,
    license="apache-2.0",
    reference="https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1",
    similarity_fn_name=ScoringFunction.COSINE,
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    citation="""
    @online{emb2024mxbai,
      title={Open Source Strikes Bread - New Fluffy Embeddings Model},
      author={Sean Lee and Aamir Shakir and Darius Koenig and Julius Lipp},
      year={2024},
      url={https://www.mixedbread.ai/blog/mxbai-embed-large-v1},
    }
    
    @article{li2023angle,
      title={AnglE-optimized Text Embeddings},
      author={Li, Xianming and Li, Jing},
      journal={arXiv preprint arXiv:2309.12871},
      year={2023}
    }
    """,
    public_training_code=None,
    public_training_data=None,
    training_datasets=None,
)
