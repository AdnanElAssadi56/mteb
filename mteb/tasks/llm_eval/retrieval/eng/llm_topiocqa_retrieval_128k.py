from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_topiocqa_metadata = dict(
    reference="https://arxiv.org/abs/2110.00768",
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
@article{adlakha2022topiocqa,
  author = {Adlakha, Vaibhav and Dhuliawala, Shehzaad and Suleman, Kaheer and de Vries, Harm and Reddy, Siva},
  journal = {Transactions of the Association for Computational Linguistics},
  title = {TopiOCQA: Open-domain Conversational Question Answering with Topic Switching},
  volume = {10},
  year = {2022},
}
""",
)


class LLMTopiOCQARetrieval128k(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMTopiOCQARetrieval128k",
        dataset={
            "path": "mteb/loft-topiocqa-128k",
            "revision": "9bd8e7636ce11669d1771800dabce1c8c58baace",
        },
        description=(
            "TopiOCQA is a conversational question answering dataset with topic switching questions "
            "that require retrieval of multiple documents."
        ),
        prompt={"query": "Given a conversational question, retrieve documents that answer the question"},
        adapted_from=["TopiOCQA"],
        **_topiocqa_metadata,
    )
