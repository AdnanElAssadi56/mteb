import gradio as gr
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Any

def create_reranking_interface(task_data: Dict[str, Any]):
    """Create a Gradio interface for reranking evaluation."""
    samples = task_data["samples"]
    results = {"task_name": task_data["task_name"], "task_type": "reranking", "annotations": []}
    completed_samples = {s["id"]: False for s in samples}
    
    def save_ranking(rankings, sample_id):
        """Save the current set of rankings."""
        # Check if all documents have rankings
        all_ranked = all(r is not None and r != "" for r in rankings)
        if not all_ranked:
            return "⚠️ Please assign a rank to all documents before submitting", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
        
        # Convert rankings to integers
        processed_rankings = [int(r) for r in rankings]
        
        # Check for duplicate rankings
        if len(set(processed_rankings)) != len(processed_rankings):
            return "⚠️ Each document must have a unique rank. Please review your rankings.", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
                
        # Store this annotation
        existing_idx = next((i for i, a in enumerate(results["annotations"]) if a["sample_id"] == sample_id), None)
        if existing_idx is not None:
            results["annotations"][existing_idx] = {
                "sample_id": sample_id,
                "rankings": processed_rankings
            }
        else:
            results["annotations"].append({
                "sample_id": sample_id,
                "rankings": processed_rankings
            })
        
        completed_samples[sample_id] = True
        success_msg = f"✅ Rankings for query '{sample_id}' successfully saved!"
        progress = f"Progress: {sum(completed_samples.values())}/{len(samples)}"
        
        # Auto-save results after each submission
        output_path = f"{task_data['task_name']}_human_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return success_msg, progress
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {task_data['task_name']} - Human Reranking Evaluation")
        
        with gr.Accordion("Instructions", open=True):
            gr.Markdown("""
            ## Task Instructions
            
            {instructions}
            
            ### How to use this interface:
            1. Read the query at the top
            2. Review each document carefully
            3. Assign a rank to each document (1 = most relevant, higher numbers = less relevant)
            4. Each document must have a unique rank
            5. Click "Submit Rankings" when you're done with the current query
            6. Use "Previous" and "Next" to navigate between queries
            7. Click "Save All Results" periodically to ensure your work is saved
            """.format(instructions=task_data["instructions"]))
        
        current_sample_id = gr.State(value=samples[0]["id"])
        
        with gr.Row():
            progress_text = gr.Textbox(label="Progress", value=f"Progress: 0/{len(samples)}", interactive=False)
            status_box = gr.Textbox(label="Status", value="Ready to start evaluation", interactive=False)
        
        with gr.Group():
            gr.Markdown("## Query:")
            query_text = gr.Textbox(value=samples[0]["query"], label="", interactive=False)
            
            gr.Markdown("## Documents to Rank:")
            
            # Create document displays and ranking dropdowns in synchronized pairs
            doc_containers = []
            ranking_dropdowns = []
            
            with gr.Column():
                for i, doc in enumerate(samples[0]["candidates"]):
                    with gr.Row():
                        doc_box = gr.Textbox(
                            value=doc, 
                            label=f"Document {i+1}",
                            interactive=False
                        )
                        dropdown = gr.Dropdown(
                            choices=[str(j) for j in range(1, len(samples[0]["candidates"])+1)],
                            label=f"Rank",
                            value=""
                        )
                        doc_containers.append(doc_box)
                        ranking_dropdowns.append(dropdown)
            
            with gr.Row():
                prev_btn = gr.Button("← Previous Query", size="sm")
                submit_btn = gr.Button("Submit Rankings", size="lg", variant="primary")
                next_btn = gr.Button("Next Query →", size="sm")
            
            save_btn = gr.Button("💾 Save All Results", variant="secondary")
        
        def load_sample(sample_id):
            """Load a specific sample into the interface."""
            sample = next((s for s in samples if s["id"] == sample_id), None)
            if not sample:
                return [query_text.value] + [d.value for d in doc_containers] + [""] * len(ranking_dropdowns) + [current_sample_id.value, progress_text.value, status_box.value]
            
            # Update query
            new_query = sample["query"]
            
            # Update documents
            new_docs = []
            for i, doc in enumerate(sample["candidates"]):
                if i < len(doc_containers):
                    new_docs.append(doc)
                    
            # Initialize rankings
            new_rankings = [""] * len(ranking_dropdowns)
            
            # Check if this sample has already been annotated
            existing_annotation = next((a for a in results["annotations"] if a["sample_id"] == sample_id), None)
            if existing_annotation:
                # Restore previous rankings
                for i, rank in enumerate(existing_annotation["rankings"]):
                    if i < len(new_rankings) and rank is not None:
                        new_rankings[i] = str(rank)
            
            # Update progress
            current_idx = samples.index(sample)
            new_progress = f"Progress: {sum(completed_samples.values())}/{len(samples)}"
            
            new_status = f"Viewing query {current_idx + 1} of {len(samples)}"
            if completed_samples[sample_id]:
                new_status += " (already completed)"
            
            return [new_query] + new_docs + new_rankings + [sample["id"], new_progress, new_status]
        
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
            return f"✅ Results saved to {output_path} ({len(results['annotations'])} annotations)"
        
        # Connect events
        submit_btn.click(
            save_ranking,
            inputs=ranking_dropdowns + [current_sample_id],
            outputs=[status_box, progress_text]
        )
        
        next_btn.click(
            next_sample,
            inputs=[current_sample_id],
            outputs=[current_sample_id]
        ).then(
            load_sample,
            inputs=[current_sample_id],
            outputs=[query_text] + doc_containers + ranking_dropdowns + [current_sample_id, progress_text, status_box]
        )
        
        prev_btn.click(
            prev_sample,
            inputs=[current_sample_id],
            outputs=[current_sample_id]
        ).then(
            load_sample,
            inputs=[current_sample_id],
            outputs=[query_text] + doc_containers + ranking_dropdowns + [current_sample_id, progress_text, status_box]
        )
        
        save_btn.click(save_results, outputs=[status_box])
    
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
    
    print("\n✅ Starting evaluation interface. Please wait for your browser to open...")
    print("💡 If no browser opens automatically, look for a URL in the output below")
    app.launch(share=True)
