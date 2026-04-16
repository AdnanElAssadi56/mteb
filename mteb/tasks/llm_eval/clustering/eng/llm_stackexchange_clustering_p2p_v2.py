from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata

_GEIGLE_BIBTEX = r"""
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
"""


class LLMStackExchangeClusteringP2PV2(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="LLMStackExchangeClusteringP2PV2",
        description="Clustering of title+body from StackExchange — LLM eval pre-sampled subset.",
        reference="https://arxiv.org/abs/2104.07081",
        dataset={
            "path": "mteb/llm-eval-stackexchange_clustering_p2p_v2",
            "revision": "5de039af34921939fb4a1c8b2e4f9f0a6ed6cd50",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2021-04-14"),
        domains=["Web", "Written"],
        task_subtypes=["Thematic clustering"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=_GEIGLE_BIBTEX,
        prompt="Identify the topic or theme of StackExchange posts based on the given paragraphs",
        adapted_from=["StackExchangeClusteringP2P"],
    )
