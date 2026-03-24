from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMAmazonCounterfactualClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LLMAmazonCounterfactualClassification",
        description="Amazon customer reviews annotated for counterfactual detection — LLM eval subset (500 test samples, seed 42).",
        reference="https://arxiv.org/abs/2104.06893",
        dataset={
            "path": "mteb/llm-eval-amazon_counterfactual",
            "revision": "8df4b672d55146368ad9d82a498ac0f16b8f177f",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "en": ["eng-Latn"],
            "de": ["deu-Latn"],
            "ja": ["jpn-Jpan"],
        },
        main_score="accuracy",
        date=("2018-01-01", "2021-12-31"),
        domains=["Reviews", "Written"],
        task_subtypes=["Counterfactual Detection"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{oneill-etal-2021-wish,
  author = {O{'}Neill, James and Rozenshtein, Polina and Kiryo, Ryuichi and Kubota, Motoko and Bollegala, Danushka},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/2021.emnlp-main.568},
  month = nov,
  pages = {7092--7108},
  publisher = {Association for Computational Linguistics},
  title = {{I} Wish {I} Would Have Loved This One, But {I} Didn{'}t -- A Multilingual Dataset for Counterfactual Detection in Product Review},
  url = {https://aclanthology.org/2021.emnlp-main.568},
  year = {2021},
}
""",
        prompt="Classify a given Amazon customer review text as either counterfactual or not-counterfactual",
        adapted_from=["AmazonCounterfactualClassification"],
    )

    samples_per_label = 32
