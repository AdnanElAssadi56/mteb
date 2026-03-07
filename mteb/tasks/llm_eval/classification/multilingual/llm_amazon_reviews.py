from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMAmazonReviewsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LLMAmazonReviewsClassification",
        dataset={
            "path": "mteb/llm-eval-amazon_reviews",
            "revision": "a2cb45438645ef5ce5014511a1652f09d1ef5025",
        },
        description="A collection of Amazon reviews specifically designed to aid research in multilingual text classification.",
        reference="https://arxiv.org/abs/2010.02573",
        category="t2c",
        modalities=["text"],
        type="Classification",
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2015-11-01", "2019-11-01"),
        domains=["Reviews", "Written"],
        task_subtypes=[],
        license="https://docs.opendata.aws/amazon-reviews-ml/license.txt",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{keung2020multilingual,
  archiveprefix = {arXiv},
  author = {Phillip Keung and Yichao Lu and György Szarvas and Noah A. Smith},
  eprint = {2010.02573},
  primaryclass = {cs.CL},
  title = {The Multilingual Amazon Reviews Corpus},
  year = {2020},
}
""",
        prompt="Classify the given Amazon review into its appropriate rating category",
        adapted_from=["AmazonReviewsClassification"],
    )
