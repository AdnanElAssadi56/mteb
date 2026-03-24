from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMImdbClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LLMImdbClassification",
        description="Large Movie Review Dataset — LLM eval subset (500 test samples, seed 42).",
        dataset={
            "path": "mteb/llm-eval-imdb",
            "revision": "c7cd15a51954e6862a5d29508c8d8db61cb8f1e8",
        },
        reference="http://www.aclweb.org/anthology/P11-1015",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2000-01-01", "2010-12-31"),
        domains=["Reviews", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{maas-etal-2011-learning,
  author = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month = jun,
  pages = {142--150},
  publisher = {Association for Computational Linguistics},
  title = {Learning Word Vectors for Sentiment Analysis},
  url = {https://aclanthology.org/P11-1015},
  year = {2011},
}
""",
        prompt="Classify the sentiment expressed in the given movie review text from the IMDB dataset",
        adapted_from=["ImdbClassification"],
    )
