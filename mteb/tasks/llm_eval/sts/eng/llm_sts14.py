from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class LLMSTS14(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="LLMSTS14",
        description="SemEval STS 2014 dataset. Currently only the English dataset — LLM eval subset.",
        reference="https://www.aclweb.org/anthology/S14-1002",
        dataset={
            "path": "mteb/llm-eval-sts14",
            "revision": "6237db5b2c4be5e7d4aacad8e850ba8d550093d6",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-08-31"),
        domains=["Blog", "Web", "Spoken"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{bandhakavi-etal-2014-generating,
  address = {Dublin, Ireland},
  author = {Bandhakavi, Anil  and
Wiratunga, Nirmalie  and
P, Deepak  and
Massie, Stewart},
  booktitle = {Proceedings of the Third Joint Conference on Lexical and Computational Semantics (*{SEM} 2014)},
  doi = {10.3115/v1/S14-1002},
  editor = {Bos, Johan  and
Frank, Anette  and
Navigli, Roberto},
  month = aug,
  pages = {12--21},
  publisher = {Association for Computational Linguistics and Dublin City University},
  title = {Generating a Word-Emotion Lexicon from {{\#}}Emotional Tweets},
  url = {https://aclanthology.org/S14-1002},
  year = {2014},
}
""",
        adapted_from=["STS14"],
    )
