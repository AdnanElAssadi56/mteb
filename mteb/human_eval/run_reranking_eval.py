import argparse
import logging
from pathlib import Path

import datasets
from mteb import get_task
from mteb.human_eval.tasks.reranking import RerankingHumanEval
from mteb.human_eval.interfaces.gradio_app import launch_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run human evaluation for MTEB reranking tasks")
    parser.add_argument("--task", type=str, required=True, help="Name of the MTEB task to evaluate")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of samples to use")
    parser.add_argument("--output-dir", type=str, default="human_eval_data", help="Directory to save evaluation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, choices=["prepare", "evaluate"], default="prepare", 
                      help="Mode of operation: prepare tasks or run evaluation interface")
    parser.add_argument("--task-file", type=str, help="Path to the prepared task file (for evaluate mode)")
    
    args = parser.parse_args()
    
    if args.mode == "prepare":
        # Load the MTEB task
        mteb_task = get_task(args.task)
        logger.info(f"Loaded task: {mteb_task.metadata.name}")
        
        # Initialize dataset
        if not mteb_task.data_loaded:
            logger.info("Loading dataset...")
            mteb_task.load_data()
        
        # Initialize human evaluator
        evaluator = RerankingHumanEval(
            task_name=mteb_task.metadata.name,
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            random_seed=args.seed,
        )
        
        # Sample dataset
        logger.info(f"Sampling {args.sample_size} examples...")
        split = mteb_task.metadata.eval_splits[0]  # Use first available evaluation split
        sampled_dataset = evaluator.sample_dataset(mteb_task.dataset, split=split)
        
        # Prepare task for human evaluation
        logger.info("Preparing task for human evaluation...")
        task_data = evaluator.prepare_for_evaluation(sampled_dataset)
        
        # Export task
        task_file = evaluator.export_task(task_data)
        logger.info(f"Task exported to: {task_file}")
        
        print(f"\nTo run the evaluation interface, use: python -m mteb.human_eval.run_reranking_eval --mode evaluate --task-file {task_file}")
    
    elif args.mode == "evaluate":
        if not args.task_file:
            parser.error("--task-file is required in evaluate mode")
        
        logger.info(f"Launching evaluation interface for: {args.task_file}")
        launch_app(args.task_file)

if __name__ == "__main__":
    main()
