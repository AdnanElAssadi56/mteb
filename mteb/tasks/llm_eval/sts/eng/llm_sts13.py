from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class LLMSTS13(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="LLMSTS13",
        description="SemEval STS 2013 dataset. — LLM eval subset.",
        reference="https://www.aclweb.org/anthology/S13-1004/",
        dataset={
            "path": "mteb/llm-eval-sts13",
            "revision": "e0251ba4e151f033f74bd5543c5d32706e810eab",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2012-01-01", "2012-12-31"),
        domains=["Web", "News", "Non-fiction", "Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{Agirre2013SEM2S,
  author = {Eneko Agirre and Daniel Matthew Cer and Mona T. Diab and Aitor Gonzalez-Agirre and Weiwei Guo},
  booktitle = {International Workshop on Semantic Evaluation},
  title = {*SEM 2013 shared task: Semantic Textual Similarity},
  url = {https://api.semanticscholar.org/CorpusID:10241043},
  year = {2013},
}
""",
        adapted_from=["STS13"],
    )
