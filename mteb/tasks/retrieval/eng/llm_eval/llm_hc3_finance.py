from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLMHC3FinanceRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMHC3FinanceRetrieval",
        description="Retrieve the relevant finance Q&A answer for an open-ended finance question, derived from the HC3 Finance corpus — LLM eval subset (100 queries, 415 docs).",
        reference="https://huggingface.co/datasets/embedding-benchmark/HC3Finance",
        dataset={
            "path": "mteb/llm-eval-hc3-finance",
            "revision": "8733760a6f3e9ddb8bacaaf020906c2ba5a5e30b",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2023-01-01", "2023-12-31"),
        domains=["Financial"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{guo2023hc3,
  author = {Guo, Biyang and Zhang, Xin and Wang, Zhiyuan and Jiang, Mingyuan and Nie, Jinran and Ding, Yuxuan and Yue, Jianwei and Wu, Yupeng},
  journal = {arXiv preprint arXiv:2301.07597},
  title = {How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection},
  year = {2023},
}
""",
        adapted_from=["HC3FinanceRetrieval"],
    )
