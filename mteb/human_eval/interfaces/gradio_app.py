import gradio as gr
import json
import numpy as np
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

def create_reranking_interface(task_data: Dict[str, Any]):
    """Create a Gradio interface for reranking evaluation."""
    samples = task_data["samples"]
    results = {"task_name": task_data["task_name"], "task_type": "reranking", "annotations": []}
    
    def save_ranking(rankings, sample_id):
        """Save the current set of rankings."""
        # Convert rankings to integers where possible
        processed_rankings = []
        for r in rankings:
            try:
                processed_rankings.append(int(r) if r else None)
            except (ValueError, TypeError):
                processed_rankings.append(None)
                
        # Store this annotation
        results["annotations"].append({
            "sample_id": sample_id,
            "rankings": processed_rankings
        })
        
        current_idx = samples.index(next(s for s in samples if s["id"] == sample_id))
        progress = f"Progress: {current_idx + 1}/{len(samples)}"
        return progress
    
    with gr.Blocks() as demo:
        gr.Markdown(f"# {task_data['task_name']} - Reranking Evaluation")
        gr.Markdown(task_data["instructions"])
        
        current_sample_id = gr.State(value=samples[0]["id"])
        progress_text = gr.Textbox(label="Progress", value=f"Progress: 1/{len(samples)}")
        
        with gr.Group():
            query_text = gr.Textbox(label="Query", value=samples[0]["query"])
            
            # Create a table for documents and rankings
            headers = ["Document", "Rank"]
            candidate_docs = samples[0]["candidates"]
            
            # Create rows for each document
            table_rows = []
            for i, doc in enumerate(candidate_docs):
                table_rows.append([doc, gr.Dropdown(choices=[str(j) for j in range(1, len(candidate_docs)+1)], label=f"Rank for Doc {i+1}")])
            
            table = gr.DataFrame(
                headers=headers,
                datatype=["str", "number"],
                row_count=len(candidate_docs),
                col_count=2,
                value=table_rows
            )
            
            submit_btn = gr.Button("Submit Rankings")
            next_btn = gr.Button("Next Sample")
            prev_btn = gr.Button("Previous Sample")
            save_btn = gr.Button("Save All Results")
            
            output_msg = gr.Textbox(label="Status")
        
        def load_sample(sample_id):
            """Load a specific sample into the interface."""
            sample = next((s for s in samples if s["id"] == sample_id), None)
            if not sample:
                return [query_text.value, table.value, current_sample_id.value, progress_text.value]
            
            # Update query
            new_query = sample["query"]
            
            # Update table
            new_table = []
            for doc in sample["candidates"]:
                new_table.append([doc, ""])
            
            # Update progress
            current_idx = samples.index(sample)
            new_progress = f"Progress: {current_idx + 1}/{len(samples)}"
            
            return [new_query, new_table, sample["id"], new_progress]
        
        def next_sample(current_id):
            """Load the next sample."""
            current_sample = next((s for s in samples if s["id"] == current_id), None)
            if not current_sample:
                return current_id
            
            current_idx = samples.index(current_sample)
            if current_idx < len(samples) - 1:
                next_sample = samples[current_idx + 1]
                return next_sample["id"]
            return current_id
        
        def prev_sample(current_id):
            """Load the previous sample."""
            current_sample = next((s for s in samples if s["id"] == current_id), None)
            if not current_sample:
                return current_id
            
            current_idx = samples.index(current_sample)
            if current_idx > 0:
                prev_sample = samples[current_idx - 1]
                return prev_sample["id"]
            return current_id
        
        def save_results():
            """Save all collected results to a file."""
            output_path = f"{task_data['task_name']}_human_results.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            return f"Results saved to {output_path}"
        
        # Connect events
        submit_btn.click(
            save_ranking,
            inputs=[
                table,
                current_sample_id
            ],
            outputs=[progress_text]
        )
        
        next_btn.click(
            next_sample,
            inputs=[current_sample_id],
            outputs=[current_sample_id]
        ).then(
            load_sample,
            inputs=[current_sample_id],
            outputs=[query_text, table, current_sample_id, progress_text]
        )
        
        prev_btn.click(
            prev_sample,
            inputs=[current_sample_id],
            outputs=[current_sample_id]
        ).then(
            load_sample,
            inputs=[current_sample_id],
            outputs=[query_text, table, current_sample_id, progress_text]
        )
        
        save_btn.click(save_results, outputs=[output_msg])
    
    return demo

def launch_app(task_file):
    """Launch the appropriate interface based on the task type."""
    with open(task_file, "r") as f:
        task_data = json.load(f)
    
    task_type = task_data.get("task_type", "")
    
    if task_type == "reranking":
        app = create_reranking_interface(task_data)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    app.launch(share=True)
