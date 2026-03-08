from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_quora_metadata = dict(
    reference="https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="recall_at_1",
    date=("2017-01-01", "2017-12-31"),  # original publication year
    domains=["Written", "Web", "Blog"],
    task_subtypes=["Question answering"],
    license="not specified",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@misc{quora-question-pairs,
  author = {DataCanary, hilfialkaff, Lili Jiang, Meg Risdal, Nikhil Dandekar, tomtung},
  publisher = {Kaggle},
  title = {Quora Question Pairs},
  url = {https://kaggle.com/competitions/quora-question-pairs},
  year = {2017},
}
""",
)


class LLMQuoraRetrieval32k(AbsTaskRetrieval):
    ignore_identical_ids = True

    metadata = TaskMetadata(
        name="LLMQuoraRetrieval32k",
        dataset={
            "path": "mteb/loft-quora-32k",
            "revision": "main",
        },
        description=(
            "QuoraRetrieval is based on questions that are marked as duplicates on the Quora platform. Given a"
            + " question, find other (duplicate) questions."
        ),
        prompt={
            "query": "Given a question, retrieve questions that are semantically equivalent to the given question"
        },
        adapted_from=["QuoraRetrieval"],
        **_quora_metadata,
    )
