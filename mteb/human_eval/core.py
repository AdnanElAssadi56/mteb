from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset

logger = logging.getLogger(__name__)

class HumanEvalTask:
    """Base class for human evaluation tasks in MTEB."""
    
    def __init__(
        self, 
        task_name: str,
        task_type: str,
        sample_size: int = 100,
        output_dir: str = "human_evaluations",
        random_seed: int = 42,
    ):
        self.task_name = task_name
        self.task_type = task_type
        self.sample_size = sample_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.random_seed = random_seed
        self.results = {}
        
    def sample_dataset(self, dataset, split: str = "test", mteb_task=None):
        """Sample a subset of examples for human evaluation."""
        # Special handling for reranking tasks
        if dataset is None and mteb_task is not None:
            # Reranking tasks store data differently - access through the task object
            queries = mteb_task.queries.get(split, {})
            corpus = mteb_task.corpus.get(split, {})
            top_ranked = mteb_task.top_ranked.get(split, {})
            
            # Create samples with query and ranked documents
            samples = []
            for query_id, query in queries.items():
                if query_id in top_ranked:
                    doc_ids = top_ranked[query_id]
                    documents = [corpus[doc_id]['text'] for doc_id in doc_ids if doc_id in corpus]
                    
                    if documents:  # Only add if we have documents
                        samples.append({
                            'id': query_id,
                            'query': query['text'],
                            'candidates': documents,
                            'doc_ids': doc_ids
                        })
            
            # Sample from the created dataset
            if len(samples) <= self.sample_size:
                return samples
            
            random.seed(self.random_seed)
            sampled_indices = random.sample(range(len(samples)), self.sample_size)
            return [samples[i] for i in sampled_indices]
        
        # Original code for standard datasets with splits
        if dataset is not None:
            if isinstance(dataset, dict) and split in dataset:
                if len(dataset[split]) <= self.sample_size:
                    return dataset[split]
                    
                random.seed(self.random_seed)
                indices = random.sample(range(len(dataset[split])), self.sample_size)
                return dataset[split].select(indices)
        
        raise ValueError(f"Dataset format not recognized or split '{split}' not found.")
    
    def prepare_for_evaluation(self, dataset: Dataset) -> Dict[str, Any]:
        """Convert a dataset into a format suitable for human annotation."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def export_task(self, task_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export the task to a JSON file for distribution to human evaluators."""
        if filename is None:
            filename = f"{self.task_name}_human_eval.json"
        
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(task_data, f, indent=2)
            
        return str(filepath)
    
    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Load human evaluation results from a file."""
        with open(results_file, "r") as f:
            self.results = json.load(f)
        return self.results
    
    def compute_metrics(self, gold_data: Optional[Dataset] = None) -> Dict[str, float]:
        """Compute performance metrics for human evaluations."""
        raise NotImplementedError("Subclasses must implement this method")
