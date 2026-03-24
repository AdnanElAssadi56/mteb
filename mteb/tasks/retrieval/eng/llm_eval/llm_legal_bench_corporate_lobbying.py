from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.task_metadata import TaskMetadata


class LLMLegalBenchCorporateLobbying(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="LLMLegalBenchCorporateLobbying",
        description="The dataset includes bill titles and bill summaries related to corporate lobbying — LLM eval subset (100 queries, 319 docs).",
        reference="https://huggingface.co/datasets/nguha/legalbench/viewer/corporate_lobbying",
        dataset={
            "path": "mteb/llm-eval-legalbench-corporate-lobbying",
            "revision": "300cb175d1b608cfc3fdb94fd06b3ceb937f9b6a",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2019-01-01", "2023-12-31"),
        domains=["Legal", "Written"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{guha2023legalbench,
  archiveprefix = {arXiv},
  author = {Neel Guha and Julian Nyarko and Daniel E. Ho and Christopher Ré and Adam Chilton and Aditya Narayana and Alex Chohlas-Wood and Austin Peters and Brandon Waldon and Daniel N. Rockmore and Diego Zambrano and Dmitry Talisman and Enam Hoque and Faiz Surani and Frank Fagan and Galit Sarfaty and Gregory M. Dickinson and Haggai Porat and Jason Hegland and Jessica Wu and Joe Nudell and Joel Niklaus and John Nay and Jonathan H. Choi and Kevin Tobia and Margaret Hagan and Megan Ma and Michael Livermore and Nikon Rasumov-Rahe and Nils Holzenberger and Noam Kolt and Peter Henderson and Sean Rehaag and Sharad Goel and Shang Gao and Spencer Williams and Sunny Gandhi and Tom Zur and Varun Iyer and Zehua Li},
  eprint = {2308.11462},
  primaryclass = {cs.CL},
  title = {LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models},
  year = {2023},
}
""",
        adapted_from=["LegalBenchCorporateLobbying"],
    )
