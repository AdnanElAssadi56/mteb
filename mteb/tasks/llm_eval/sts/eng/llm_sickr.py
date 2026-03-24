from mteb.abstasks.sts import AbsTaskSTS
from mteb.abstasks.task_metadata import TaskMetadata


class LLMSICKR(AbsTaskSTS):
    min_score = 0
    max_score = 5

    metadata = TaskMetadata(
        name="LLMSICKR",
        description="Semantic Textual Similarity SICK-R dataset — LLM eval subset.",
        reference="https://aclanthology.org/L14-1314/",
        dataset={
            "path": "mteb/llm-eval-sickr",
            "revision": "82eb9939fa177ddd94d5ddf4668e011d7446c1da",
        },
        type="STS",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=("2014-01-01", "2014-12-31"),
        domains=["Web", "Written"],
        task_subtypes=["Textual Entailment"],
        license="cc-by-nc-sa-3.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="created",
        bibtex_citation=r"""
@inproceedings{marelli-etal-2014-sick,
  address = {Reykjavik, Iceland},
  author = {Marelli, Marco  and
Menini, Stefano  and
Baroni, Marco  and
Bentivogli, Luisa  and
Bernardi, Raffaella  and
Zamparelli, Roberto},
  booktitle = {Proceedings of the Ninth International Conference on Language Resources and Evaluation ({LREC}'14)},
  editor = {Calzolari, Nicoletta  and
Choukri, Khalid  and
Declerck, Thierry  and
Loftsson, Hrafn  and
Maegaard, Bente  and
Mariani, Joseph  and
Moreno, Asuncion  and
Odijk, Jan  and
Piperidis, Stelios},
  month = may,
  pages = {216--223},
  publisher = {European Language Resources Association (ELRA)},
  title = {A {SICK} cure for the evaluation of compositional distributional semantic models},
  url = {http://www.lrec-conf.org/proceedings/lrec2014/pdf/363_Paper.pdf},
  year = {2014},
}
""",
        adapted_from=["SICKR"],
    )
