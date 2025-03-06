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
        
    def sample_dataset(self, dataset: Dataset, split: str = "test") -> Dataset:
        """Sample a subset of examples for human evaluation."""
        if len(dataset[split]) <= self.sample_size:
            return dataset[split]
            
        random.seed(self.random_seed)
        indices = random.sample(range(len(dataset[split])), self.sample_size)
        return dataset[split].select(indices)
    
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
