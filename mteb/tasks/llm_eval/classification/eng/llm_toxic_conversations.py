from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMToxicConversationsClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LLMToxicConversationsClassification",
        description="Toxic Conversations dataset — LLM eval subset (500 test samples, seed 42).",
        dataset={
            "path": "mteb/llm-eval-toxic_conversations",
            "revision": "34eeb6105ca217433c04b207ea66810a2ff42625",
        },
        reference="https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2019-01-01", "2019-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="cc-by-4.0",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{jigsaw-unintended-bias-in-toxicity-classification,
  author = {Daniel Borkan and Lucas Dixon and Jeffrey Sorensen and Nithum Thain and Lucy Vasserman},
  title = {Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification},
  year = {2019},
  url = {https://arxiv.org/abs/1903.04561},
}
""",
        prompt="Classify the given comment as either toxic or not toxic",
        adapted_from=["ToxicConversationsClassification"],
    )
