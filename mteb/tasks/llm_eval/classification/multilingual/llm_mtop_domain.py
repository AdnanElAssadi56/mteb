from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMMTOPDomainClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LLMMTOPDomainClassification",
        description="MTOP domain classification — LLM eval subset (500 test samples, seed 42).",
        reference="https://arxiv.org/abs/2008.09335",
        dataset={
            "path": "mteb/llm-eval-mtop_domain",
            "revision": "14315a9fd4305bf99cf0e400e06fb54a9b815f9c",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "en": ["eng-Latn"],
            "de": ["deu-Latn"],
            "fr": ["fra-Latn"],
        },
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Spoken", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{li-etal-2021-mtop,
  author = {Li, Haoran and Arora, Abhinav and Chen, Shuohui and Gupta, Anchit and Gupta, Sonal and Mehdad, Yashar},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics},
  doi = {10.18653/v1/2021.eacl-main.257},
  month = apr,
  pages = {2950--2962},
  publisher = {Association for Computational Linguistics},
  title = {{MTOP}: A Comprehensive Multilingual Task-Oriented Semantic Parsing Benchmark},
  url = {https://aclanthology.org/2021.eacl-main.257},
  year = {2021},
}
""",
        prompt="Classify the intent domain of the given utterance in task-oriented conversation",
        adapted_from=["MTOPDomainClassification"],
    )
