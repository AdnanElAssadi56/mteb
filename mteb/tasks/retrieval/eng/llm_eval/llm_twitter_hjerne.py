from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLMTwitterHjerneRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMTwitterHjerneRetrieval",
        description="Danish question asked on Twitter with the Hashtag #Twitterhjerne ('Twitter brain') and their corresponding answer — LLM eval subset (78 queries, 262 docs).",
        reference="https://huggingface.co/datasets/sorenmulli/da-hashtag-twitterhjerne",
        dataset={
            "path": "mteb/llm-eval-twitter-hjerne",
            "revision": "31f9b918c30ef94e15a168cb95ca4ebf291396eb",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["train"],
        eval_langs=["dan-Latn"],
        main_score="ndcg_at_10",
        date=("2006-01-01", "2024-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Question answering"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{holm2024gllms,
  author = {Holm, Soren Vejlgaard},
  title = {Are GLLMs Danoliterate? Benchmarking Generative NLP in Danish},
  year = {2024},
}
""",
        prompt={"query": "Retrieve answers to questions asked in Danish tweets"},
        adapted_from=["TwitterHjerneRetrieval"],
    )
