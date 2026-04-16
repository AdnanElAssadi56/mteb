from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class LLMArxivClusteringS2S(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="LLMArxivClusteringS2S",
        description="Clustering of titles from arxiv — LLM eval pre-sampled subset.",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/llm-eval-arxiv_clustering_s2s",
            "revision": "28c899f0bb82d9001804d0f408d66932effc0c1c",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1991-01-01", "2021-01-01"),
        domains=["Academic", "Written"],
        task_subtypes=[],
        license="cc0-1.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{arxiv_org_submitters_2024,
  author = {arXiv.org submitters},
  doi = {10.34740/KAGGLE/DSV/7548853},
  publisher = {Kaggle},
  title = {arXiv Dataset},
  url = {https://www.kaggle.com/dsv/7548853},
  year = {2024},
}
""",
        prompt="Identify the main and secondary category of Arxiv papers based on the titles",
        adapted_from=["ArxivClusteringS2S"],
    )
