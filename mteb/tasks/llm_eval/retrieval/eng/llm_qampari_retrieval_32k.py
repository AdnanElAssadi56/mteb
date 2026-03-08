from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_qampari_metadata = dict(
    reference="https://arxiv.org/abs/2205.12644",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="recall_at_1",
    date=("2022-01-01", "2022-12-31"),
    domains=["Web", "Written"],
    task_subtypes=["Question answering"],
    license="cc-by-4.0",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@article{rubin2022qampari,
  author = {Rubin, Ohad and Herzig, Jonathan and Berant, Jonathan},
  journal = {arXiv preprint arXiv:2205.12644},
  title = {QAMPARI: A Benchmark for Open-domain Questions with Many Answers},
  year = {2022},
}
""",
)


class LLMQAMPARIRetrieval32k(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMQAMPARIRetrieval32k",
        dataset={
            "path": "mteb/loft-qampari-32k",
            "revision": "d89a5c90a7dc1e5e47fa7e8ee553d1d28cde5f11",
        },
        description=(
            "QAMPARI is an open-domain question answering benchmark with questions that have multiple answers "
            "spread across multiple documents."
        ),
        prompt={"query": "Given a question with multiple answers, retrieve documents that contain the answers"},
        adapted_from=["QAMPARI"],
        **_qampari_metadata,
    )
