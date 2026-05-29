from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLMFQuADRetrieval(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMFQuADRetrieval",
        description="Retrieve the French Wikipedia paragraph that answers a French question — LLM eval subset (100 queries, 269 docs).",
        reference="https://huggingface.co/datasets/manu/fquad2_test",
        dataset={
            "path": "mteb/llm-eval-fquad",
            "revision": "dc5443dbfad5cd88c9507d27db4cde4caaede178",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["fra-Latn"],
        main_score="ndcg_at_10",
        date=("2019-11-01", "2020-05-01"),
        domains=["Encyclopaedic", "Written"],
        task_subtypes=["Question answering"],
        license="apache-2.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{dhoffschmidt-etal-2020-fquad,
  address = {Online},
  author = {d{'}Hoffschmidt, Martin  and
Belblidia, Wacim  and
Heinrich, Quentin  and
Brendl{\'e}, Tom  and
Vidal, Maxime},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  doi = {10.18653/v1/2020.findings-emnlp.107},
  editor = {Cohn, Trevor  and
He, Yulan  and
Liu, Yang},
  month = nov,
  pages = {1193--1208},
  publisher = {Association for Computational Linguistics},
  title = {{FQ}u{AD}: {F}rench Question Answering Dataset},
  url = {https://aclanthology.org/2020.findings-emnlp.107},
  year = {2020},
}
""",
        adapted_from=["FQuADRetrieval"],
    )
