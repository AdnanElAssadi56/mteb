from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset
from scipy.stats import kendalltau

from ..core import HumanEvalTask

logger = logging.getLogger(__name__)

class RerankingHumanEval(HumanEvalTask):
    """Human evaluation for reranking tasks."""
    
    def __init__(
        self,
        task_name: str,
        sample_size: int = 100,
        output_dir: str = "human_evaluations",
        random_seed: int = 42,
    ):
        super().__init__(
            task_name=task_name,
            task_type="reranking",
            sample_size=sample_size,
            output_dir=output_dir,
            random_seed=random_seed,
        )
    
    def prepare_for_evaluation(self, dataset):
        """Convert a reranking dataset into a format suitable for human annotation."""
        samples = []
        
        # Handle the custom format from sample_dataset
        if isinstance(dataset, list) and dataset and 'query' in dataset[0]:
            for idx, example in enumerate(dataset):
                sample = {
                    'id': example.get('id', idx),
                    'query': example['query'],
                    'candidates': example['candidates'],
                    # Add any other fields needed
                }
                samples.append(sample)
        else:
            # Original code for standard dataset format
            for idx, example in enumerate(dataset):
                # Handle different dataset formats
                query = example.get("query", example.get("question", ""))
                candidates = example.get("texts", example.get("documents", example.get("passages", [])))
                relevance = example.get("relevance", example.get("labels", []))
                
                # Ensure we have necessary data
                if not query or not candidates:
                    logger.warning(f"Skipping example {idx}: missing query or candidates")
                    continue
                    
                sample = {
                    "id": idx,
                    "query": query,
                    "candidates": candidates,
                    "true_relevance": relevance,  # Used for evaluation, not shown to human annotators
                }
                samples.append(sample)
        
        instructions = (
            "Rank the candidate documents based on their relevance to the query. "
            "Assign rank 1 to the most relevant document, 2 to the second most relevant, and so on."
        )
        
        return {
            "task_type": "reranking",
            "task_name": self.task_name,
            "instructions": instructions,
            "samples": samples,
        }
    
    def compute_metrics(self, gold_data: Optional[Dataset] = None) -> Dict[str, float]:
        """Compute performance metrics for human reranking evaluations."""
        if not self.results or "annotations" not in self.results:
            raise ValueError("No results available. Call load_results first.")
            
        # Prepare for metrics calculation
        all_kendall_tau = []
        all_precision_at_1 = []
        all_ndcg_at_3 = []
        all_ndcg_at_10 = []
        
        for annotation in self.results["annotations"]:
            sample_id = annotation["sample_id"]
            human_rankings = annotation["rankings"]
            
            # Find the corresponding gold data
            gold_sample = next((s for s in gold_data if s["id"] == sample_id), None)
            if not gold_sample or "true_relevance" not in gold_sample:
                logger.warning(f"Missing gold data for sample {sample_id}")
                continue
                
            gold_relevance = gold_sample["true_relevance"]
            
            # Calculate Kendall's Tau
            # Convert rankings to comparable format
            human_ranks = np.array([int(r) for r in human_rankings])
            # For gold data, higher relevance should get lower ranks
            gold_ranks = (-np.array(gold_relevance)).argsort().argsort() + 1
            
            kt, _ = kendalltau(human_ranks, gold_ranks)
            all_kendall_tau.append(kt)
            
            # Calculate P@1
            top1_human = human_ranks.argmin()
            p_at_1 = gold_relevance[top1_human] == max(gold_relevance)
            all_precision_at_1.append(float(p_at_1))
            
            # Calculate NDCG@3 and NDCG@10
            # CURRENTLY NOT IMPLEMENTED
        
        return {
            "kendall_tau": np.mean(all_kendall_tau),
            "precision_at_1": np.mean(all_precision_at_1),
            "ndcg_at_3": np.mean(all_ndcg_at_3),
            "ndcg_at_10": np.mean(all_ndcg_at_10),
        }
