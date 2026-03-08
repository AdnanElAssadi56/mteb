from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class LLMTwentyNewsgroupsClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="LLMTwentyNewsgroupsClustering",
        description="Clustering of the 20 Newsgroups dataset (subject only).",
        reference="https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
        dataset={
            "path": "mteb/llm-eval-twenty_newsgroups",
            "revision": "e8714f5e20fc679b53b9f980187f4102522aa149",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1995-01-01", "1995-01-01"),
        domains=["News", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@incollection{LANG1995331,
  address = {San Francisco (CA)},
  author = {Ken Lang},
  booktitle = {Machine Learning Proceedings 1995},
  doi = {https://doi.org/10.1016/B978-1-55860-377-6.50048-7},
  editor = {Armand Prieditis and Stuart Russell},
  isbn = {978-1-55860-377-6},
  pages = {331-339},
  publisher = {Morgan Kaufmann},
  title = {NewsWeeder: Learning to Filter Netnews},
  url = {https://www.sciencedirect.com/science/article/pii/B9781558603776500487},
  year = {1995},
}
""",
        prompt="Identify the topic or theme of the given news articles",
        adapted_from=["TwentyNewsgroupsClustering"],
    )
    instruction = "Cluster the following Usenet newsgroup posts by topic."
