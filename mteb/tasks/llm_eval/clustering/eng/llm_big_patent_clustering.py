from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class LLMBigPatentClustering(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="LLMBigPatentClustering",
        description="Clustering of documents from the Big Patent dataset — LLM eval pre-sampled subset.",
        reference="https://huggingface.co/datasets/NortheasternUniversity/big_patent",
        dataset={
            "path": "mteb/llm-eval-big_patent_clustering",
            "revision": "fbb01d02cb8308089d99223cbcc35c584a67ba77",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("1971-01-01", "2019-06-10"),
        domains=["Legal", "Written"],
        task_subtypes=["Thematic clustering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{DBLP:journals/corr/abs-1906-03741,
  author = {Eva Sharma and
Chen Li and
Lu Wang},
  eprint = {1906.03741},
  eprinttype = {arXiv},
  journal = {CoRR},
  title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
  url = {http://arxiv.org/abs/1906.03741},
  volume = {abs/1906.03741},
  year = {2019},
}
""",
        adapted_from=["BigPatentClustering"],
    )
