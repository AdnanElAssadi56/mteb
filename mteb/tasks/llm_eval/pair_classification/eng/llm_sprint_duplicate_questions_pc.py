from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMSprintDuplicateQuestionsPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="LLMSprintDuplicateQuestionsPC",
        description="Duplicate questions from the Sprint community — LLM eval subset.",
        reference="https://www.aclweb.org/anthology/D18-1131/",
        dataset={
            "path": "mteb/llm-eval-sprint_duplicate_questions",
            "revision": "d1c6be04a5f84b606b758024ed9d3b3fb7e4029a",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2018-10-01", "2018-12-30"),
        domains=["Programming", "Written"],
        task_subtypes=["Duplicate Detection"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve duplicate questions from Sprint forum",
        bibtex_citation=r"""
@inproceedings{shah-etal-2018-adversarial,
  address = {Brussels, Belgium},
  author = {Shah, Darsh  and
Lei, Tao  and
Moschitti, Alessandro  and
Romeo, Salvatore  and
Nakov, Preslav},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1131},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {1056--1063},
  publisher = {Association for Computational Linguistics},
  title = {Adversarial Domain Adaptation for Duplicate Question Detection},
  url = {https://aclanthology.org/D18-1131},
  year = {2018},
}
""",
        adapted_from=["SprintDuplicateQuestions"],
    )

    def dataset_transform(self, num_proc=None):
        self.dataset = self.dataset.rename_column("label", "labels")
