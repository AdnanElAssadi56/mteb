from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_hotpot_qa_metadata = dict(
    reference="https://hotpotqa.github.io/",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="recall_at_1",
    date=("2018-01-01", "2018-12-31"),  # best guess: based on publication date
    domains=["Web", "Written"],
    task_subtypes=["Question answering"],
    license="cc-by-sa-4.0",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@inproceedings{yang-etal-2018-hotpotqa,
  address = {Brussels, Belgium},
  author = {Yang, Zhilin  and
Qi, Peng  and
Zhang, Saizheng  and
Bengio, Yoshua  and
Cohen, William  and
Salakhutdinov, Ruslan  and
Manning, Christopher D.},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  doi = {10.18653/v1/D18-1259},
  editor = {Riloff, Ellen  and
Chiang, David  and
Hockenmaier, Julia  and
Tsujii, Jun{'}ichi},
  month = oct # {-} # nov,
  pages = {2369--2380},
  publisher = {Association for Computational Linguistics},
  title = {{H}otpot{QA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  url = {https://aclanthology.org/D18-1259},
  year = {2018},
}
""",
)


class LLMHotpotQARetrieval128k(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMHotpotQARetrieval128k",
        dataset={
            "path": "mteb/loft-hotpotqa-128k",
            "revision": "8c3c58a1167bc2134ce23511062fc211533b266a",
        },
        description=(
            "HotpotQA is a question answering dataset featuring natural, multi-hop questions, with strong "
            "supervision for supporting facts to enable more explainable question answering systems."
        ),
        prompt={
            "query": "Given a multi-hop question, retrieve documents that can help answer the question"
        },
        adapted_from=["HotpotQA"],
        **_hotpot_qa_metadata,
    )
