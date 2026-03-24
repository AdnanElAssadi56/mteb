from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata


class LLMTweetSentimentExtractionClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="LLMTweetSentimentExtractionClassification",
        description="Tweet Sentiment Extraction dataset — LLM eval subset (500 test samples, seed 42).",
        dataset={
            "path": "mteb/llm-eval-tweet_sentiment",
            "revision": "5e847f1f41ec9089cfdb7ea7b2852a23bd5ea7c8",
        },
        reference="https://www.kaggle.com/competitions/tweet-sentiment-extraction",
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="accuracy",
        date=("2020-01-01", "2020-12-31"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@misc{tweet-sentiment-extraction-2020,
  author = {Kaggle},
  title = {Tweet Sentiment Extraction},
  year = {2020},
  url = {https://www.kaggle.com/competitions/tweet-sentiment-extraction},
}
""",
        prompt="Classify the sentiment of a tweet as positive, negative, or neutral",
        adapted_from=["TweetSentimentExtractionClassification"],
    )
