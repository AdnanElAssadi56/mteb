from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class LLMBigPatentClustering(AbsTaskClustering):
    max_depth = 1
    metadata = TaskMetadata(
        name="LLMBigPatentClustering",
        description="Clustering of documents from the Big Patent dataset. Test set only includes documents belonging to a single category, with a total of 9 categories.",
        reference="https://huggingface.co/datasets/NortheasternUniversity/big_patent",
        dataset={
            "path": "mteb/llm-eval-big_patent_clustering",
            "revision": "05b54214c5abac981d97173657590e0211c9de52",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=(
            "1971-01-01",
            "2019-06-10",
        ),  # start date from paper, end date - paper publication
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
  bibsource = {dblp computer science bibliography, https://dblp.org},
  biburl = {https://dblp.org/rec/journals/corr/abs-1906-03741.bib},
  eprint = {1906.03741},
  eprinttype = {arXiv},
  journal = {CoRR},
  timestamp = {Wed, 26 Jun 2019 07:14:58 +0200},
  title = {{BIGPATENT}: A Large-Scale Dataset for Abstractive and Coherent Summarization},
  url = {http://arxiv.org/abs/1906.03741},
  volume = {abs/1906.03741},
  year = {2019},
}
""",
        adapted_from=["BigPatentClustering.v2"],
    )
