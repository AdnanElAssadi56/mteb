from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMTwitterURLCorpusPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="LLMTwitterURLCorpusPC",
        dataset={
            "path": "mteb/llm-eval-twitter_url_corpus",
            "revision": "741fbe5cf43a594d8c37073d66baf8ab879281ce",
        },
        description="Paraphrase-Pairs of Tweets.",
        reference="https://languagenet.github.io/",
        category="t2t",
        modalities=["text"],
        type="PairClassification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_ap",
        date=("2017-01-01", "2017-12-31"),  # publication year
        domains=["Social", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
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
        prompt="Retrieve tweets that are semantically similar to the given tweet",
        adapted_from=["TwitterURLCorpus"],
    )

    label_column_name = "label"
    instruction = (
        "Determine whether two tweets convey the same message (are paraphrases of each other). "
        "Output json with fields 'reasoning' and 'output' (1 if paraphrase, 0 if not)."
    )

    def dataset_transform(self, num_proc: int | None = None) -> None:
        pass
