from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLMAILAStatutes(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMAILAStatutes",
        description="Identifying the most relevant statutes for a given situation — LLM eval subset (50 queries, 82 docs).",
        reference="https://zenodo.org/records/4063986",
        dataset={
            "path": "mteb/llm-eval-aila-statutes",
            "revision": "a2acf12d293ea934823ca26752a05c5bfab24dff",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2020-10-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@dataset{paheli_bhattacharya_2020_4063986,
  author = {Paheli Bhattacharya and
Kripabandhu Ghosh and
Saptarshi Ghosh and
Arindam Pal and
Parth Mehta and
Arnab Bhattacharya and
Prasenjit Majumder},
  doi = {10.5281/zenodo.4063986},
  month = oct,
  publisher = {Zenodo},
  title = {AILA 2019 Precedent \& Statute Retrieval Task},
  url = {https://doi.org/10.5281/zenodo.4063986},
  year = {2020},
}
""",
        adapted_from=["AILAStatutes"],
    )
