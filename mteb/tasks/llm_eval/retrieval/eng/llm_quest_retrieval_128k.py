from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_quest_metadata = dict(
    reference="https://arxiv.org/abs/2309.03028",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="recall_at_1",
    date=("2023-01-01", "2023-12-31"),
    domains=["Web", "Written"],
    task_subtypes=["Question answering"],
    license="cc-by-4.0",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@article{maekawa2023quest,
  author = {Maekawa, Tetsuya and Hayashi, Koki and Okazaki, Naoaki and Yoda, Taro},
  journal = {arXiv preprint arXiv:2309.03028},
  title = {QUEST: A Retrieval Dataset of Entity-Seeking Queries with Implicit Set Operations},
  year = {2023},
}
""",
)


class LLMQUESTRetrieval128k(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMQUESTRetrieval128k",
        dataset={
            "path": "mteb/loft-quest-128k",
            "revision": "fa0b5d677bfb3a129f16f15b78e4ad7b19ab078d",
        },
        description=(
            "QUEST is a retrieval dataset with entity-seeking queries requiring implicit set operations."
        ),
        prompt={"query": "Given a question, retrieve documents that answer the question"},
        adapted_from=["QUEST"],
        **_quest_metadata,
    )
