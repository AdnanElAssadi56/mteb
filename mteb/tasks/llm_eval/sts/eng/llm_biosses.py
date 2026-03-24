from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class LLMBIOSSES(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="LLMBIOSSES",
        description="Biomedical Semantic Similarity Estimation. — LLM eval subset.",
        reference="https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
        dataset={
            "path": "mteb/llm-eval-biosses",
            "revision": "cf968edb41fa17a96392f6b373819efac1c2d6d6",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2017-07-01", "2017-12-31"),
        domains=["Medical"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{10.1093/bioinformatics/btx238,
  author = {Soğancıoğlu, Gizem and Öztürk, Hakime and Özgür, Arzucan},
  doi = {10.1093/bioinformatics/btx238},
  eprint = {https://academic.oup.com/bioinformatics/article-pdf/33/14/i49/50315066/bioinformatics\_33\_14\_i49.pdf},
  issn = {1367-4803},
  journal = {Bioinformatics},
  month = {07},
  number = {14},
  pages = {i49-i58},
  title = {{BIOSSES: a semantic sentence similarity estimation system for the biomedical domain}},
  url = {https://doi.org/10.1093/bioinformatics/btx238},
  volume = {33},
  year = {2017},
}
""",
        adapted_from=["BIOSSES"],
    )
