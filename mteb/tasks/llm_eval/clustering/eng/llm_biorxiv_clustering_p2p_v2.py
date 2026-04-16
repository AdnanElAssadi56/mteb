from mteb.abstasks.clustering_legacy import AbsTaskClusteringLegacy
from mteb.abstasks.task_metadata import TaskMetadata


class LLMBiorxivClusteringP2PV2(AbsTaskClusteringLegacy):
    metadata = TaskMetadata(
        name="LLMBiorxivClusteringP2PV2",
        description="Clustering of titles+abstract from biorxiv across 26 categories — LLM eval pre-sampled subset.",
        reference="https://api.biorxiv.org/",
        dataset={
            "path": "mteb/llm-eval-biorxiv_clustering_p2p_v2",
            "revision": "9e11c95384ef78952ba754f9d8942084ddbb61a7",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        domains=["Academic", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://www.biorxiv.org/content/about-biorxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        prompt="Identify the main category of Biorxiv papers based on the titles and abstracts",
        adapted_from=["BiorxivClusteringP2P"],
    )
