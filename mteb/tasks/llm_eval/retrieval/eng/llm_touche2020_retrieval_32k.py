from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata

_touche2020_metadata = dict(
    reference="https://webis.de/events/touche-20/shared-task-1.html",
    type="Retrieval",
    category="t2t",
    modalities=["text"],
    eval_splits=["test"],
    eval_langs=["eng-Latn"],
    main_score="recall_at_1",
    date=("2020-01-01", "2020-12-31"),
    domains=["Web", "Written"],
    task_subtypes=["Question answering"],
    license="cc-by-4.0",
    annotations_creators="human-annotated",
    dialect=[],
    sample_creation="found",
    bibtex_citation=r"""
@inproceedings{bondarenko2020overview,
  author    = {Alexander Bondarenko and
               Maik Fr{\"{o}}be and
               Meriem Beloucif and
               Lukas Gienapp and
               Yamen Ajjour and
               Alexander Panchenko and
               Chris Biemann and
               Benno Stein and
               Henning Wachsmuth and
               Martin Potthast and
               Matthias Hagen},
  title     = {Overview of Touch{\'{e}} 2020: Argument Retrieval},
  booktitle = {Experimental {IR} Meets Multilinguality, Multimodality, and Interaction
               - 11th International Conference of the {CLEF} Association, {CLEF}
               2020, Thessaloniki, Greece, September 22-25, 2020, Proceedings},
  series    = {Lecture Notes in Computer Science},
  volume    = {12260},
  pages     = {384--395},
  publisher = {Springer},
  year      = {2020},
}
""",
)


class LLMTouche2020Retrieval32k(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMTouche2020Retrieval32k",
        dataset={
            "path": "mteb/loft-webis-touche2020-32k",
            "revision": "main",
        },
        description=(
            "Touché-2020 argument retrieval (LOFT 32k scale, 10 queries)."
        ),
        prompt={"query": "Given a question, retrieve detailed and persuasive arguments that answer the question"},
        adapted_from=["Touche2020"],
        **_touche2020_metadata,
    )
