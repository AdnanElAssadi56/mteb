from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class LLMMedrxivClusteringP2PV2(AbsTaskClustering):
    metadata = TaskMetadata(
        name="LLMMedrxivClusteringP2PV2",
        description="Clustering of titles+abstract from medrxiv across 51 categories — LLM eval pre-sampled subset.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "mteb/llm-eval-medrxiv_clustering_p2p_v2",
            "revision": "63c8e6cfbcab3f986291141799fe646c60bb441c",
        },
        type="Clustering",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="v_measure",
        date=("2021-01-01", "2022-05-10"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Thematic clustering"],
        license="https://www.medrxiv.org/content/about-medrxiv",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation="",
        prompt="Identify the main category of Medrxiv papers based on the titles and abstracts",
        adapted_from=["MedrxivClusteringP2P"],
    )
