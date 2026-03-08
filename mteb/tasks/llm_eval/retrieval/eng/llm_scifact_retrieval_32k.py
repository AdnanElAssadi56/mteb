from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLMSciFactRetrieval32k(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMSciFactRetrieval32k",
        dataset={
            "path": "mteb/loft-scifact-32k",
            "revision": "1541ed7823c2bb464b00590cd51d59fe8fc01869",
        },
        description="SciFact verifies scientific claims using evidence from the research literature containing scientific paper abstracts.",
        reference="https://github.com/allenai/scifact",
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="recall_at_1",
        date=("2020-01-01", "2020-12-31"),
        domains=["Academic", "Medical", "Written"],
        task_subtypes=["Claim verification"],
        license="cc-by-nc-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@inproceedings{specter2020cohan,
  author = {Arman Cohan and Sergey Feldman and Iz Beltagy and Doug Downey and Daniel S. Weld},
  booktitle = {ACL},
  title = {SPECTER: Document-level Representation Learning using Citation-informed Transformers},
  year = {2020},
}
""",
        prompt={
            "query": "Given a scientific claim, retrieve documents that support or refute the claim"
        },
        adapted_from=["SciFact"],
    )
