from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES_STS17 = {
    "en-de": ["eng-Latn", "deu-Latn"],
    "en-en": ["eng-Latn"],
    "es-es": ["spa-Latn"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "it-en": ["ita-Latn", "eng-Latn"],
}


class LLMSTS17(AbsTaskSTS):
    fast_loading = False
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="LLMSTS17",
        description="SemEval-2017 STS multilingual — LLM eval subset (en, de, es, fr, it).",
        reference="https://alt.qcri.org/semeval2017/task1/",
        dataset={
            "path": "mteb/llm-eval-sts17",
            "revision": "fe4f4e1b9fdafeae22df69e66bdc3b634be30e9d",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES_STS17,
        main_score="cosine_spearman",
        date=("2014-01-01", "2017-12-31"),
        domains=["News", "Web", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{cer-etal-2017-semeval,
  author = {Cer, Daniel and Diab, Mona and Agirre, Eneko and Lopez-Gazpio, I{\~n}igo and Specia, Lucia},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)},
  doi = {10.18653/v1/S17-2001},
  month = aug,
  pages = {1--14},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation},
  url = {https://aclanthology.org/S17-2001},
  year = {2017},
}
""",
        adapted_from=["STS17"],
    )
