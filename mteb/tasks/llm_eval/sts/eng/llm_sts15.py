from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class LLMSTS15(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="LLMSTS15",
        description="SemEval STS 2015 dataset — LLM eval subset.",
        reference="https://www.aclweb.org/anthology/S15-2010",
        dataset={
            "path": "mteb/llm-eval-sts15",
            "revision": "6283e03ad9bd28cb1ed3d3a472d08a0e9df58fc1",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2008-01-01", "2014-07-28"),
        domains=["Blog", "News", "Web", "Written", "Spoken"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{bicici-2015-rtm,
  address = {Denver, Colorado},
  author = {Bi{\c{c}}ici, Ergun},
  booktitle = {Proceedings of the 9th International Workshop on Semantic Evaluation ({S}em{E}val 2015)},
  doi = {10.18653/v1/S15-2010},
  editor = {Nakov, Preslav  and
Zesch, Torsten  and
Cer, Daniel  and
Jurgens, David},
  month = jun,
  pages = {56--63},
  publisher = {Association for Computational Linguistics},
  title = {{RTM}-{DCU}: Predicting Semantic Similarity with Referential Translation Machines},
  url = {https://aclanthology.org/S15-2010},
  year = {2015},
}
""",
        adapted_from=["STS15"],
    )
