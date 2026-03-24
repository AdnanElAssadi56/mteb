from mteb.abstasks.clustering import AbsTaskClustering
from mteb.abstasks.task_metadata import TaskMetadata


class LLMMedrxivClusteringS2SV2(AbsTaskClustering):
    metadata = TaskMetadata(
        name="LLMMedrxivClusteringS2SV2",
        description="Clustering of titles from medrxiv across 51 categories — LLM eval pre-sampled subset.",
        reference="https://api.medrxiv.org/",
        dataset={
            "path": "mteb/llm-eval-medrxiv_clustering_s2s_v2",
            "revision": "c565eea82f4a8728b8fe5181388b407d305f2647",
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
        prompt="Identify the main category of Medrxiv papers based on the titles",
        adapted_from=["MedrxivClusteringS2S"],
    )
