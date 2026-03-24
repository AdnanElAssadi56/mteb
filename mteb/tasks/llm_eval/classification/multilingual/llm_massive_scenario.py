from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMMassiveScenarioClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LLMMassiveScenarioClassification",
        description="MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages — LLM eval subset (500 test samples, seed 42).",
        reference="https://arxiv.org/abs/2204.08582",
        dataset={
            "path": "mteb/llm-eval-massive_scenario",
            "revision": "ba1521289b080d967f820b289dd285f22fb968a8",
        },
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "en": ["eng-Latn"],
            "de": ["deu-Latn"],
            "fr": ["fra-Latn"],
            "ja": ["jpn-Jpan"],
        },
        main_score="accuracy",
        date=("2022-01-01", "2022-04-22"),
        domains=["Spoken"],
        task_subtypes=[],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated and localized",
        bibtex_citation=r"""
@inproceedings{fitzgerald-etal-2023-massive,
  author = {FitzGerald, Jack and Hench, Christopher and Peris, Charith and Mackie, Scott and Rottmann, Kay and Sanchez, Ana and Nash, Aaron and Urbach, Liam and Kakarala, Vishesh and Singh, Richa and Ranganath, Swetha and Crist, Laurie and Britan, Misha and Leeuwis, Wouter and Tur, Gokhan and Natarajan, Prem},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  doi = {10.18653/v1/2023.acl-long.235},
  month = jul,
  pages = {4277--4302},
  publisher = {Association for Computational Linguistics},
  title = {{MASSIVE}: A 1{M}-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
  url = {https://aclanthology.org/2023.acl-long.235},
  year = {2023},
}
""",
        prompt="Given a user utterance, classify its scenario domain",
        adapted_from=["MassiveScenarioClassification"],
    )
