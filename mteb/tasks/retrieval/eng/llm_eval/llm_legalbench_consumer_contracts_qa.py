from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLMLegalBenchConsumerContractsQA(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMLegalBenchConsumerContractsQA",
        description="Retrieve the relevant Terms-of-Service clause for a consumer-contract question from LegalBench — LLM eval subset (100 queries, 154 docs).",
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/consumer_contracts_qa",
        dataset={
            "path": "mteb/llm-eval-legalbench-consumer-contracts",
            "revision": "642870c78f65ed31a2ab0f8ab113f2957ff2df27",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2023-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-nc-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{hendrycks2021cuad,
  author = {Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal = {arXiv preprint arXiv:2103.06268},
  title = {Cuad: An expert-annotated nlp dataset for legal contract review},
  year = {2021},
}

@article{koreeda2021contractnli,
  author = {Koreeda, Yuta and Manning, Christopher D},
  journal = {arXiv preprint arXiv:2110.01799},
  title = {ContractNLI: A dataset for document-level natural language inference for contracts},
  year = {2021},
}
""",
        adapted_from=["LegalBenchConsumerContractsQA"],
    )
