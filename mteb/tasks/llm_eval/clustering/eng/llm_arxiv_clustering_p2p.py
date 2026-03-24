from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class LLMArxivClusteringP2P(AbsTaskClustering):
    metadata = TaskMetadata(
        name="LLMArxivClusteringP2P",
        description="Clustering of titles+abstract from arxiv — LLM eval pre-sampled subset.",
        reference="https://www.kaggle.com/Cornell-University/arxiv",
        dataset={
            "path": "mteb/llm-eval-arxiv_clustering_p2p",
            "revision": "3ce50711e47553e7c9f47b53819ccd64cd0f5b9b",
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
        prompt="Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
        adapted_from=["ArxivClusteringP2P"],
    )
