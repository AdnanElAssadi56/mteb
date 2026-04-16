from mteb.abstasks.pair_classification import AbsTaskPairClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMLegalBenchPC(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="LLMLegalBenchPC",
        description="This LegalBench pair classification task is a combination of the following datasets: Citation Prediction Classification, Consumer Contracts QA, Contract QA, Hearsay, Privacy Policy Entailment, Privacy Policy QA. — LLM eval subset.",
        reference="https://huggingface.co/datasets/nguha/legalbench",
        dataset={
            "path": "mteb/llm-eval-legal_bench_pc",
            "revision": "a0217dc60ec5a45077e213a9538e845533523ed0",
        },
        type="PairClassification",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="max_accuracy",
        date=("2000-01-01", "2023-08-23"),
        domains=["Legal", "Written"],
        task_subtypes=[],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
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

@article{kolt2022predicting,
  author = {Kolt, Noam},
  journal = {Berkeley Tech. LJ},
  pages = {71},
  publisher = {HeinOnline},
  title = {Predicting consumer contracts},
  volume = {37},
  year = {2022},
}

@article{ravichander2019question,
  author = {Ravichander, Abhilasha and Black, Alan W and Wilson, Shomir and Norton, Thomas and Sadeh, Norman},
  journal = {arXiv preprint arXiv:1911.00841},
  title = {Question answering for privacy policies: Combining computational and legal perspectives},
  year = {2019},
}

@article{zimmeck2019maps,
  author = {Zimmeck, Sebastian and Story, Peter and Smullen, Daniel and Ravichander, Abhilasha and Wang, Ziqi and Reidenberg, Joel R and Russell, N Cameron and Sadeh, Norman},
  journal = {Proc. Priv. Enhancing Tech.},
  pages = {66},
  title = {Maps: Scaling privacy compliance analysis to a million apps},
  volume = {2019},
  year = {2019},
}
""",
        adapted_from=["LegalBenchPC"],
    )

    def dataset_transform(self, num_proc=None):
        self.dataset = self.dataset.rename_column("label", "labels")
