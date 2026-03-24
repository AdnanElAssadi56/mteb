from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES_STS22 = {
    "en": ["eng-Latn"],
    "de": ["deu-Latn"],
    "es": ["spa-Latn"],
    "fr": ["fra-Latn"],
    "ru": ["rus-Cyrl"],
    "zh": ["cmn-Hans"],
}


class LLMSTS22v2(AbsTaskSTS):
    fast_loading = True
    min_score = 1
    max_score = 4

    metadata = TaskMetadata(
        name="LLMSTS22v2",
        description="SemEval-2022 Task 8 multilingual STS v2 — LLM eval subset (en, de, es, fr, ru, zh).",
        reference="https://competitions.codalab.org/competitions/33835",
        dataset={
            "path": "mteb/llm-eval-sts22_v2",
            "revision": "7a79fd41b024091522d04e84d8d9dc93d223cf8c",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES_STS22,
        main_score="cosine_spearman",
        date=("2020-01-01", "2020-06-11"),
        domains=["News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{chen-etal-2022-semeval,
  author = {Chen, Xi and Zeynali, Ali and Camargo, Chico and Fl{\"o}ck, Fabian and Gaffney, Devin and Grabowicz, Przemyslaw and Hale, Scott and Jurgens, David and Samory, Mattia},
  booktitle = {Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)},
  doi = {10.18653/v1/2022.semeval-1.155},
  month = jul,
  pages = {1094--1106},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2022 Task 8: Multilingual news article similarity},
  url = {https://aclanthology.org/2022.semeval-1.155},
  year = {2022},
}
""",
        adapted_from=["STS22.v2"],
    )
