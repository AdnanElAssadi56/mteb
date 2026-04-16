from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata

_LANGUAGES_RTE3 = {
    "de": ["deu-Latn"],
    "en": ["eng-Latn"],
    "fr": ["fra-Latn"],
    "it": ["ita-Latn"],
}


class LLMRTE3PC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="LLMRTE3PC",
        description="Recognising Textual Entailment Challenge (RTE-3) — LLM eval subset (de, en, fr, it).",
        reference="https://aclanthology.org/W07-1401/",
        dataset={
            "path": "mteb/llm-eval-rte3",
            "revision": "5d745bd9e435cc190599f916f54c0ab197ddeb73",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=_LANGUAGES_RTE3,
        main_score="max_ap",
        date=("2023-03-25", "2024-04-15"),
        domains=["News", "Web", "Encyclopaedic", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{giampiccolo-etal-2007-third,
  address = {Prague},
  author = {Giampiccolo, Danilo  and
Magnini, Bernardo  and
Dagan, Ido  and
Dolan, Bill},
  booktitle = {Proceedings of the {ACL}-{PASCAL} Workshop on Textual Entailment and Paraphrasing},
  month = jun,
  pages = {1--9},
  publisher = {Association for Computational Linguistics},
  title = {The Third {PASCAL} Recognizing Textual Entailment Challenge},
  url = {https://aclanthology.org/W07-1401},
  year = {2007},
}
""",
        adapted_from=["RTE3"],
    )

    def dataset_transform(self, num_proc=None):
        for lang in self.dataset:
            self.dataset[lang] = self.dataset[lang].rename_column("label", "labels")
