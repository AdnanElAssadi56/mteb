from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLMPublicHealthQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMPublicHealthQA",
        description="Retrieve the relevant public-health / COVID-19 information passage for a public-health question — LLM eval subset, English subset only (100 queries, 172 docs).",
        reference="https://huggingface.co/datasets/xhluca/publichealth-qa",
        dataset={
            "path": "mteb/llm-eval-public-health-qa",
            "revision": "b05938525381b7aebc079f88fc3ed8f572a80bb9",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2020-12-31"),
        domains=["Medical", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{xhluca_publichealthqa,
  author = {Lu, Xing Han},
  howpublished = {\url{https://huggingface.co/datasets/xhluca/publichealth-qa}},
  title = {Public Health QA: A Multilingual Public Health Questions Dataset},
  year = {2020},
}
""",
        adapted_from=["PublicHealthQA"],
    )
