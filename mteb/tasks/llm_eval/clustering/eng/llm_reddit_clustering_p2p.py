from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class LLMRedditClusteringP2P(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="LLMRedditClusteringP2P",
        description="Clustering of title+posts from reddit — LLM eval pre-sampled subset.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/llm-eval-reddit_clustering_p2p",
            "revision": "298518f04f9f2a5849282dd20e43672172428790",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-04-14"),
        domains=["Web", "Social", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{geigle:2021:arxiv,
  archiveprefix = {arXiv},
  author = {Gregor Geigle and
Nils Reimers and
Andreas R{\"u}ckl{\'e} and
Iryna Gurevych},
  eprint = {2104.07081},
  journal = {arXiv preprint},
  title = {TWEAC: Transformer with Extendable QA Agent Classifiers},
  url = {http://arxiv.org/abs/2104.07081},
  volume = {abs/2104.07081},
  year = {2021},
}
""",
        prompt="Identify the topic or theme of Reddit posts based on the titles and posts",
        adapted_from=["RedditClusteringP2P"],
    )
