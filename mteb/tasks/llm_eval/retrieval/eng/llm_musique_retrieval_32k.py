from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_musique_metadata = dict(
    reference="https://arxiv.org/abs/2108.00573",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="recall_at_1",
    date=("2021-01-01", "2021-12-31"),
    domains=["Web", "Written"],
    task_subtypes=["Question answering"],
    license="cc-by-4.0",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@article{trivedi2022musique,
  author = {Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal = {Transactions of the Association for Computational Linguistics},
  title = {MuSiQue: Multihop Questions via Single-hop Question Composition},
  volume = {10},
  year = {2022},
}
""",
)


class LLMMusiqueRetrieval32k(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMMusiqueRetrieval32k",
        dataset={
            "path": "mteb/loft-musique-32k",
            "revision": "0e0c4975d1b6a4df4b51f10d66d26d28fb96e6e6",
        },
        description=(
            "MuSiQue is a multi-hop question answering dataset where questions require reasoning over multiple documents."
        ),
        prompt={"query": "Given a multi-hop question, retrieve documents that can help answer the question"},
        adapted_from=["MusiQue"],
        **_musique_metadata,
    )
