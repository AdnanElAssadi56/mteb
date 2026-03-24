from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMTwitterURLCorpusPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="LLMTwitterURLCorpusPC",
        description="Paraphrase-Pairs of Tweets — LLM eval subset.",
        reference="https://languagenet.github.io/",
        dataset={
            "path": "mteb/llm-eval-twitter_url_corpus",
            "revision": "741fbe5cf43a594d8c37073d66baf8ab879281ce",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2017-01-01", "2017-12-31"),
        domains=["Social", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        prompt="Retrieve tweets that are semantically similar to the given tweet",
        bibtex_citation=r"""
@inproceedings{lan-etal-2017-continuously,
  address = {Copenhagen, Denmark},
  author = {Lan, Wuwei  and
Qiu, Siyu  and
He, Hua  and
Xu, Wei},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D17-1126},
  editor = {Palmer, Martha  and
Hwa, Rebecca  and
Riedel, Sebastian},
  month = sep,
  pages = {1224--1234},
  publisher = {Association for Computational Linguistics},
  title = {A Continuously Growing Dataset of Sentential Paraphrases},
  url = {https://aclanthology.org/D17-1126},
  year = {2017},
}
""",
        adapted_from=["TwitterURLCorpus"],
    )

    def dataset_transform(self, num_proc=None):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
