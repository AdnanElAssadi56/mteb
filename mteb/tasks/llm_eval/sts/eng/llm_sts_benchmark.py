from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class LLMSTSBenchmark(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="LLMSTSBenchmark",
        description="Semantic Textual Similarity Benchmark — LLM eval subset.",
        reference="https://github.com/PhilipMay/stsb-multi-mt/",
        dataset={
            "path": "mteb/llm-eval-stsbenchmark",
            "revision": "86bbaf4470f501ee381411836b3a22f112bfe42a",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2018-12-31"),
        domains=["Blog", "News", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{cer-etal-2017-semeval,
  author = {Cer, Daniel and Diab, Mona and Agirre, Eneko and Lopez-Gazpio, I{\~n}igo and Specia, Lucia},
  booktitle = {Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017)},
  doi = {10.18653/v1/S17-2001},
  month = aug,
  pages = {1--14},
  publisher = {Association for Computational Linguistics},
  title = {{S}em{E}val-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation},
  url = {https://aclanthology.org/S17-2001},
  year = {2017},
}
""",
        adapted_from=["STSBenchmark"],
    )
