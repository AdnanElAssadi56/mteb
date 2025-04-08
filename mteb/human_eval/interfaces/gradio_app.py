import gradio as gr
import json
import os
from pathlib import Path
import time

def create_reranking_interface(task_data):
    """Create a Gradio interface for reranking evaluation."""
    samples = task_data["samples"]
    results = {"task_name": task_data["task_name"], "task_type": "reranking", "annotations": []}
    completed_samples = {s["id"]: False for s in samples}
    
    # Load existing results if available
    output_path = f"{task_data['task_name']}_human_results.json"
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                saved_results = json.load(f)
                if "annotations" in saved_results:
                    results["annotations"] = saved_results["annotations"]
                    # Update completed_samples based on loaded data
                    for annotation in saved_results["annotations"]:
                        sample_id = annotation.get("sample_id")
                        if sample_id and sample_id in completed_samples:
                            completed_samples[sample_id] = True
        except Exception as e:
            print(f"Error loading existing results: {e}")
    
    def save_ranking(rankings, sample_id):
        """Save the current set of rankings."""
        try:
            # Check if all documents have rankings
            if not rankings or len(rankings) == 0:
                return "⚠️ No rankings provided", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
                
            all_ranked = all(r is not None and r != "" for r in rankings)
            if not all_ranked:
                return "⚠️ Please assign a rank to all documents before submitting", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
            
            # Convert rankings to integers with better error handling
            try:
                processed_rankings = [int(r) for r in rankings]
            except ValueError:
                return "⚠️ Invalid ranking value. Please use only numbers.", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
            
            # Check for duplicate rankings
            if len(set(processed_rankings)) != len(processed_rankings):
                return "⚠️ Each document must have a unique rank. Please review your rankings.", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
                
            # Store this annotation in memory
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
            
            # Always save to file for redundancy
            try:
                output_path = f"{task_data['task_name']}_human_results.json"
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                return f"✅ Rankings saved successfully", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
            except Exception as file_error:
                # If file saving fails, still mark as success since we saved in memory
                print(f"File save error: {file_error}")
                return f"✅ Rankings saved in memory (file save failed)", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
        except Exception as e:
            # Return specific error message
            print(f"Save ranking error: {e}")
            return f"Error: {str(e)}", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Header section with title and progress indicators
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                gr.Markdown(f"# {task_data['task_name']} - Human Reranking Evaluation")
            with gr.Column(scale=1):
                progress_text = gr.Textbox(
                    label="Progress", 
                    value=f"Progress: 0/{len(samples)}", 
                    interactive=False
                )
        
        # Instructions in a collapsible section
        with gr.Accordion("📋 Task Instructions", open=False):
            gr.Markdown("""
            ## Task Instructions
            
            {instructions}
            
            ### How to use this interface:
            1. Read the query at the top
            2. Review each document carefully
            3. Assign a rank to each document (1 = most relevant, higher numbers = less relevant)
               - Use the dropdown menus to select ranks
            4. Each document must have a unique rank
            5. Click "Submit Rankings" to save rankings for the current query
            6. Use "Previous" and "Next" to navigate between queries
            7. Your rankings are automatically saved when you submit or navigate (if auto-save is enabled)
            8. Click "Save All Results" periodically to ensure all your work is saved to disk
            
            **Button Explanations:**
            - **Submit Rankings**: Saves rankings for the CURRENT query only
            - **Save All Results**: Saves ALL submitted rankings to a file on disk
            - **Auto-save**: When enabled, automatically saves rankings when navigating between queries
            """.format(instructions=task_data.get("instructions", "Rank documents by their relevance to the query.")))
        
        # Hidden state variables
        current_sample_id = gr.State(value=samples[0]["id"])
        auto_save_enabled = gr.State(value=True)
        
        # Status and control section
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                status_box = gr.Textbox(
                    label="Status", 
                    value="Ready to start evaluation", 
                    interactive=False
                )
            with gr.Column(scale=1):
                auto_save_toggle = gr.Checkbox(
                    label="Auto-save when navigating", 
                    value=True
                )
        
        # Main content area
        with gr.Group():
            # Query section with clear visual distinction
            with gr.Box():
                gr.Markdown("## 📝 Query")
                query_text = gr.Textbox(
                    value=samples[0]["query"], 
                    label="", 
                    interactive=False,
                    elem_classes=["query-text"]
                )
            
            # Documents section with improved layout
            gr.Markdown("## 📄 Documents to Rank")
            
            # Container for documents and rankings
            doc_containers = []
            ranking_inputs = []
            validation_indicators = []
            
            # Create a clean header with explanatory labels for quick ranking tools
            gr.Markdown("### Quick Ranking Tools", elem_classes=["tools-header"])
            with gr.Box(elem_classes=["tools-container"]):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4):
                        sequential_btn = gr.Button("Rank 1,2,3... (Sequential)", variant="secondary")
                    with gr.Column(scale=4):
                        reverse_btn = gr.Button("Rank n,n-1... (Reverse)", variant="secondary")
                    with gr.Column(scale=3):
                        clear_btn = gr.Button("Clear All Rankings", variant="secondary")
                with gr.Row():
                    gr.Markdown("<small>Use these buttons to quickly assign rankings to all documents at once.</small>", elem_classes=["tools-help"])
            
            # Make document textboxes much smaller
            with gr.Box():
                for i, doc in enumerate(samples[0]["candidates"]):
                    row_class = "document-row-even" if i % 2 == 0 else "document-row-odd"
                    with gr.Row(equal_height=True, elem_classes=["document-row", row_class]):
                        with gr.Column(scale=1, min_width=50):
                            gr.HTML(f"<div class='doc-number'>{i+1}</div>")
                        
                        with gr.Column(scale=7):
                            doc_box = gr.Textbox(
                                value=doc, 
                                label=f"Document {i+1}",
                                interactive=False,
                                elem_classes=["document-text"],
                                lines=2,  # Reduce to only 2 visible lines
                            )
                            doc_containers.append(doc_box)
                        
                        with gr.Column(scale=2):
                            # Dropdown for ranking
                            rank_input = gr.Dropdown(
                                choices=[str(j) for j in range(1, len(samples[0]["candidates"])+1)],
                                label=f"Rank",
                                value="",
                                elem_classes=["rank-dropdown"]
                            )
                            ranking_inputs.append(rank_input)
                        
                        with gr.Column(scale=2):
                            # Validation indicator
                            validation = gr.HTML(value="")
                            validation_indicators.append(validation)
            
            # Navigation and submission controls
            with gr.Row(equal_height=True):
                prev_btn = gr.Button("← Previous Query", size="sm")
                submit_btn = gr.Button("Submit Rankings", size="lg", variant="primary")
                next_btn = gr.Button("Next Query →", size="sm")
            
            # Save results button
            with gr.Row():
                save_btn = gr.Button("💾 Save All Results", variant="secondary", size="sm")
                results_info = gr.HTML(value=f"<p>Results will be saved to <code>{task_data['task_name']}_human_results.json</code></p>")
        
        # CSS for styling
        gr.HTML("""
        <style>
            .query-text textarea {
                font-size: 18px !important;
                font-weight: bold !important;
                background-color: #f8f9fa !important;
                border-left: 4px solid #2c7be5 !important;
                padding-left: 10px !important;
                line-height: 1.6 !important;
            }
            
            .document-row {
                border-bottom: 1px solid #e0e0e0;
                padding: 8px 0;
                margin-bottom: 4px !important;
            }
            
            .document-text textarea {
                font-size: 14px !important;
                line-height: 1.4 !important;
                padding: 6px !important;
                min-height: 60px !important;  /* Dramatically reduce minimum height */
                height: auto !important;       
                overflow-y: visible !important;
            }
            
            .rank-dropdown select {
                font-weight: bold !important;
                font-size: 14px !important;
                text-align: center !important;
                padding: 5px !important;
                border-radius: 5px !important;
                border: 2px solid #2c7be5 !important;
            }
            
            .rank-dropdown select:focus {
                border-color: #007bff !important;
                box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
            }
            
            .tools-container {
                background-color: #f8f9fa !important;
                border-left: 4px solid #6c757d !important;
                padding: 10px !important;
                margin-bottom: 15px !important;
                border-radius: 5px !important;
            }
            
            .tools-header {
                margin-bottom: 5px !important;
                font-weight: bold !important;
                color: #333 !important;
                border-bottom: 1px solid #ddd !important;
                padding-bottom: 5px !important;
            }
            
            .tools-help {
                color: #666 !important;
                margin-top: 5px !important;
                text-align: center !important;
            }
            
            .section-header {
                margin: 0 !important;
                padding-top: 8px !important;
            }
            
            .document-row-even {
                background-color: #f8f9fa;
            }
            
            .document-row-odd {
                background-color: #ffffff;
            }
            
            .document-row:hover {
                background-color: #e9ecef;
            }
            
            .doc-number {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 25px;
                height: 25px;
                border-radius: 50%;
                background-color: #2c7be5;
                color: white;
                font-weight: bold;
                margin: 0 auto;
                font-size: 12px !important;
            }
        </style>
        """)
        
        def validate_rankings(*rankings):
            """Simplified validation with less HTML for better performance."""
            results = []
            all_valid = True
            for rank in rankings:
                if rank is None or rank == "":
                    # Use simpler HTML with less styling for faster rendering
                    results.append("⚠️ Missing")
                    all_valid = False
                else:
                    # Use simpler HTML with less styling for faster rendering
                    results.append("✓ Rank " + str(rank))
            
            return results + [all_valid]  # Return validation indicators and validity flag
        
        def on_ranking_change(*rankings):
            """Simplified validation for better performance."""
            validation_results = validate_rankings(*rankings)
            return validation_results[:-1]  # Return only the validation indicators
        
        def submit_rankings(*args):
            """Submit rankings with more efficient validation."""
            # Get the last argument (sample_id) and the rankings
            if len(args) < 1:
                return "Error: No arguments provided", progress_text.value
            
            # Verify we have enough rankings
            if len(args) < len(ranking_inputs) + 1:
                return "Error: Not enough ranking inputs provided", progress_text.value
            
            sample_id = args[-1]
            rankings = args[:len(ranking_inputs)]
            
            # First validate the rankings
            validation_results = validate_rankings(*rankings)
            all_valid = validation_results[-1]  # Last item is validity flag
            validation_indicators_values = validation_results[:-1]  # Remove validity flag
            
            # Update validation indicators - less frequently
            # Only update if really needed
            for i, result in enumerate(validation_indicators_values):
                if i < len(validation_indicators):
                    validation_indicators[i].update(value=result)
            
            # Check for duplicate rankings
            if all_valid:
                try:
                    processed_rankings = [int(r) for r in rankings]
                    if len(set(processed_rankings)) != len(processed_rankings):
                        dup_ranks = {}
                        for i, r in enumerate(processed_rankings):
                            if r in dup_ranks:
                                dup_ranks[r].append(i)
                            else:
                                dup_ranks[r] = [i]
                        
                        # Use simpler HTML for duplicate messages
                        for rank, indices in dup_ranks.items():
                            if len(indices) > 1:
                                for idx in indices:
                                    if idx < len(validation_indicators):
                                        validation_indicators[idx].update(
                                            value=f"⚠️ Duplicate {rank}"
                                        )
                        
                        return "⚠️ Each document must have a unique rank. Please fix duplicate rankings.", progress_text.value
                except:
                    pass
            
            # If not all valid, return error message
            if not all_valid:
                return "⚠️ Please assign a rank to all documents before submitting", progress_text.value
            
            # Save the validated rankings
            status, progress = save_ranking(rankings, sample_id)
            
            # Provide clear success feedback - with simpler HTML
            if "✅" in status:
                for i in range(len(validation_indicators)):
                    validation_indicators[i].update(
                        value="✓ Saved"
                    )
            
            return status, progress
        
        def load_sample(sample_id):
            """Load a specific sample into the interface."""
            sample = next((s for s in samples if s["id"] == sample_id), None)
            if not sample:
                return [query_text.value] + [d.value for d in doc_containers] + [""] * len(ranking_inputs) + [""] * len(validation_indicators) + [sample_id, progress_text.value, status_box.value]
            
            # Update query
            new_query = sample["query"]
            
            # Update documents
            new_docs = []
            for i, doc in enumerate(sample["candidates"]):
                if i < len(doc_containers):
                    new_docs.append(doc)
                    
            # Initialize rankings
            new_rankings = [""] * len(ranking_inputs)
            
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
            
            # Initialize validation indicators
            validation_results = validate_rankings(*new_rankings)
            validation_indicators_values = validation_results[:-1]  # Remove validity flag
            
            return [new_query] + new_docs + new_rankings + validation_indicators_values + [sample_id, new_progress, new_status]
        
        def auto_save_and_navigate(direction, current_id, auto_save, *rankings):
            """Save rankings if auto-save is enabled, then navigate."""
            # Extract rankings (remove validation indicators)
            actual_rankings = rankings[:len(ranking_inputs)]
            
            # If auto-save is enabled, try to save the current rankings
            status_msg = ""
            progress_msg = f"Progress: {sum(completed_samples.values())}/{len(samples)}"
            
            if auto_save:
                # Only save if all rankings are provided
                validation_results = validate_rankings(*actual_rankings)
                all_valid = validation_results[-1]  # Last item is validity flag
                if all_valid:
                    status_msg, progress_msg = save_ranking(actual_rankings, current_id)
            
            # Navigate to the next/previous sample
            if direction == "next":
                new_id = next_sample(current_id)
            else:
                new_id = prev_sample(current_id)
            
            # Return the new sample ID and status message
            return new_id, status_msg, progress_msg
        
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
            try:
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                return f"✅ Results saved to {output_path} ({len(results['annotations'])} annotations)"
            except Exception as e:
                return f"Error saving results: {str(e)}"
        
        # Define functions for the quick ranking buttons
        def assign_sequential_ranks():
            values = [str(i+1) for i in range(len(samples[0]["candidates"]))]
            # Skip validation until all ranks are assigned
            return values
        
        def assign_reverse_ranks():
            n = len(samples[0]["candidates"])
            values = [str(n-i) for i in range(n)]
            # Skip validation until all ranks are assigned
            return values
        
        def clear_rankings():
            values = [""] * len(samples[0]["candidates"])
            # Clear validation indicators when clearing rankings
            for indicator in validation_indicators:
                indicator.update(value="")
            return values
        
        # Connect quick ranking buttons
        sequential_btn.click(
            fn=assign_sequential_ranks,
            inputs=None,
            outputs=ranking_inputs
        )
        
        reverse_btn.click(
            fn=assign_reverse_ranks,
            inputs=None,
            outputs=ranking_inputs
        )
        
        clear_btn.click(
            fn=clear_rankings,
            inputs=None,
            outputs=ranking_inputs
        )
        
        # Wire up events (Gradio 3.x syntax)
        submit_btn.click(
            fn=submit_rankings,
            inputs=ranking_inputs + [current_sample_id],
            outputs=[status_box, progress_text]
        )
        
        # Auto-save and navigate events
        def handle_next(current_id, auto_save, *rankings):
            # First, handle auto-save - only if needed
            if auto_save and any(r != "" for r in rankings):
                new_id, status, progress = auto_save_and_navigate("next", current_id, auto_save, *rankings)
            else:
                new_id = next_sample(current_id)
                status, progress = "", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
                
            # Then, load the new sample with minimal validation
            outputs = load_sample(new_id)
            # Update only status and progress if needed
            if status:
                outputs[-2] = progress
                outputs[-1] = status
                
            return outputs
        
        def handle_prev(current_id, auto_save, *rankings):
            # First, handle auto-save - only if needed
            if auto_save and any(r != "" for r in rankings):
                new_id, status, progress = auto_save_and_navigate("prev", current_id, auto_save, *rankings)
            else:
                new_id = prev_sample(current_id)
                status, progress = "", f"Progress: {sum(completed_samples.values())}/{len(samples)}"
                
            # Then, load the new sample with minimal validation
            outputs = load_sample(new_id)
            # Update only status and progress if needed
            if status:
                outputs[-2] = progress
                outputs[-1] = status
                
            return outputs
        
        # Connect navigation with Gradio 3.x syntax
        next_btn.click(
            fn=handle_next,
            inputs=[current_sample_id, auto_save_toggle] + ranking_inputs,
            outputs=[query_text] + doc_containers + ranking_inputs + validation_indicators + [current_sample_id, progress_text, status_box]
        )
        
        prev_btn.click(
            fn=handle_prev,
            inputs=[current_sample_id, auto_save_toggle] + ranking_inputs,
            outputs=[query_text] + doc_containers + ranking_inputs + validation_indicators + [current_sample_id, progress_text, status_box]
        )
        
        # Connect save button
        save_btn.click(
            fn=save_results,
            inputs=None,
            outputs=[status_box]
        )
        
        # Connect auto-save toggle
        def update_auto_save(enabled):
            return enabled
            
        auto_save_toggle.change(
            fn=update_auto_save,
            inputs=[auto_save_toggle],
            outputs=[auto_save_enabled]
        )
        
        # Reduce frequency of validation
        # Only connect validation to the first ranking input to reduce event handlers
        ranking_inputs[0].change(
            fn=on_ranking_change,
            inputs=ranking_inputs,
            outputs=validation_indicators
        )
        
        # Helper function for ranking - sort documents by rankings
        def rank_by_relevance(*args):
            """Sorts the documents by their current rankings for a clearer view."""
            # Last argument is sample_id
            sample_id = args[-1]
            rankings = args[:-1]
            
            # Check if we have valid rankings
            valid_rankings = []
            for i, r in enumerate(rankings):
                if r is not None and r != "":
                    try:
                        valid_rankings.append((i, int(r)))
                    except:
                        pass
            
            # If we don't have enough valid rankings, do nothing
            if len(valid_rankings) < 2:
                return [status_box.value]
            
            # Sort by rank
            valid_rankings.sort(key=lambda x: x[1])
            
            # Generate message showing the ranking order
            result = "<p><strong>Current ranking order:</strong></p><ol>"
            for idx, _ in valid_rankings:
                doc_text = doc_containers[idx].value
                # Truncate if too long
                if len(doc_text) > 100:
                    doc_text = doc_text[:97] + "..."
                result += f"<li>Doc {idx+1}: {doc_text}</li>"
            result += "</ol>"
            
            return [result]
        
    return demo

# Main app with file upload capability and improved task management
def create_main_app():
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown("# MTEB Human Evaluation Demo")
        
        task_container = gr.HTML()
        loaded_task_info = gr.JSON(label="Loaded Task Information", visible=False)
        
        # CSS for consistent styling throughout the app
        gr.HTML("""
        <style>
            /* Main App Styling */
            .tab-content {
                padding: 15px !important;
            }
            
            .btn-primary {
                background-color: #2c7be5 !important;
            }
            
            .btn-secondary {
                background-color: #6c757d !important;
            }
            
            /* Status messages */
            .status-message {
                font-weight: bold !important;
            }
            
            /* Box styling */
            .content-box {
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                background-color: #f8f9fa;
            }
            
            /* Section headers */
            .section-header {
                border-bottom: 2px solid #2c7be5;
                padding-bottom: 5px;
                margin-bottom: 15px;
            }
        </style>
        """)
        
        tabs = gr.Tabs()
        
        with tabs:
            with gr.TabItem("Demo"):
                gr.Markdown("""
                ## MTEB Human Evaluation Interface
                
                This interface allows you to evaluate the relevance of documents for reranking tasks.
                """, elem_classes=["section-header"])
                
                # Function to get the most recent task file
                def get_latest_task_file():
                    # Check first in uploaded_tasks directory
                    os.makedirs("uploaded_tasks", exist_ok=True)
                    uploaded_tasks = [f for f in os.listdir("uploaded_tasks") if f.endswith(".json")]
                    
                    if uploaded_tasks:
                        # Sort by modification time, newest first
                        uploaded_tasks.sort(key=lambda x: os.path.getmtime(os.path.join("uploaded_tasks", x)), reverse=True)
                        task_path = os.path.join("uploaded_tasks", uploaded_tasks[0])
                        
                        # Verify this is a valid task file
                        try:
                            with open(task_path, "r") as f:
                                task_data = json.load(f)
                                if "task_name" in task_data and "samples" in task_data:
                                    return task_path
                        except:
                            pass
                    
                    # Look for task files in the current directory
                    current_dir_tasks = [f for f in os.listdir(".") if f.endswith("_human_eval.json")]
                    if current_dir_tasks:
                        # Sort by modification time, newest first
                        current_dir_tasks.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        return current_dir_tasks[0]
                    
                    # Fall back to fixed example if available
                    if os.path.exists("AskUbuntuDupQuestions_human_eval.json"):
                        return "AskUbuntuDupQuestions_human_eval.json"
                    
                    # No valid task file found
                    return None
                
                # Load the task file
                task_file = get_latest_task_file()
                
                with gr.Box(elem_classes=["content-box"]):
                    if task_file:
                        try:
                            with open(task_file, "r") as f:
                                task_data = json.load(f)
                            
                            # Show which task is currently loaded
                            gr.Markdown(f"**Current Task: {task_data['task_name']}** ({len(task_data['samples'])} samples)")
                            
                            # Display the interface
                            demo = create_reranking_interface(task_data)
                            task_container.update(value=f"<p>Task loaded: {task_file}</p>")
                        except Exception as e:
                            gr.Markdown(f"**Error loading task: {str(e)}**", elem_classes=["status-message"])
                            gr.Markdown("Please upload a valid task file in the 'Upload & Evaluate' tab.")
                    else:
                        gr.Markdown("**No task file found**", elem_classes=["status-message"])
                        gr.Markdown("Please upload a valid task file in the 'Upload & Evaluate' tab.")
            
            with gr.TabItem("Upload & Evaluate"):
                gr.Markdown("""
                ## Upload Your Own Task File
                
                If you have a prepared task file, you can upload it here to create an evaluation interface.
                """, elem_classes=["section-header"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Box(elem_classes=["content-box"]):
                            file_input = gr.File(label="Upload a task file (JSON)")
                            load_btn = gr.Button("Load Task", variant="primary")
                            message = gr.Textbox(label="Status", interactive=False, elem_classes=["status-message"])
                        
                        # Add task list for previously uploaded tasks
                        with gr.Box(elem_classes=["content-box"]):
                            gr.Markdown("### Previous Uploads", elem_classes=["section-header"])
                            
                            # Function to list existing task files in the tasks directory
                            def list_task_files():
                                os.makedirs("uploaded_tasks", exist_ok=True)
                                tasks = [f for f in os.listdir("uploaded_tasks") if f.endswith(".json")]
                                if not tasks:
                                    return "No task files uploaded yet."
                                return "\n".join([f"- {t}" for t in tasks])
                            
                            task_list = gr.Markdown(list_task_files())
                            refresh_btn = gr.Button("Refresh List")
                        
                        # Add results management section
                        with gr.Box(elem_classes=["content-box"]):
                            gr.Markdown("### Results Management", elem_classes=["section-header"])
                            
                            # Function to list existing result files
                            def list_result_files():
                                results = [f for f in os.listdir(".") if f.endswith("_human_results.json")]
                                if not results:
                                    return "No result files available yet."
                                
                                result_links = []
                                for r in results:
                                    # Calculate completion stats
                                    try:
                                        with open(r, "r") as f:
                                            result_data = json.load(f)
                                        annotation_count = len(result_data.get("annotations", []))
                                        task_name = result_data.get("task_name", "Unknown")
                                        result_links.append(f"- {r} ({annotation_count} annotations for {task_name})")
                                    except:
                                        result_links.append(f"- {r}")
                                
                                return "\n".join(result_links)
                            
                            results_list = gr.Markdown(list_result_files())
                            download_results_btn = gr.Button("Download Results")
                
                # Handle file upload and storage
                def handle_upload(file):
                    if not file:
                        return "Please upload a task file", task_list.value, ""
                    
                    try:
                        # Create directory if it doesn't exist
                        os.makedirs("uploaded_tasks", exist_ok=True)
                        
                        # Read the uploaded file
                        with open(file.name, "r") as f:
                            task_data = json.load(f)
                        
                        # Validate task format
                        if "task_name" not in task_data or "samples" not in task_data:
                            return "Invalid task file format. Must contain 'task_name' and 'samples' fields.", task_list.value, ""
                        
                        # Save to a consistent location
                        task_filename = f"uploaded_tasks/{task_data['task_name']}_task.json"
                        with open(task_filename, "w") as f:
                            json.dump(task_data, f, indent=2)
                        
                        return f"✅ Task '{task_data['task_name']}' uploaded successfully with {len(task_data['samples'])} samples. Please refresh the app and use the Demo tab to evaluate it.", list_task_files(), f"""
                        <div class="content-box">
                            <h3>Task uploaded successfully!</h3>
                            <p>Task Name: {task_data['task_name']}</p>
                            <p>Samples: {len(task_data['samples'])}</p>
                            <p>To evaluate this task:</p>
                            <ol>
                                <li>Refresh the app</li>
                                <li>The Demo tab will now use your uploaded task</li>
                                <li>Complete your evaluations</li>
                                <li>Results will be saved as {task_data['task_name']}_human_results.json</li>
                            </ol>
                        </div>
                        """
                    except Exception as e:
                        return f"⚠️ Error processing task file: {str(e)}", task_list.value, ""
                
                # Function to prepare results for download
                def prepare_results_for_download():
                    results = [f for f in os.listdir(".") if f.endswith("_human_results.json")]
                    if not results:
                        return None
                    
                    # Create a zip file with all results
                    import zipfile
                    zip_path = "mteb_human_eval_results.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for r in results:
                            zipf.write(r)
                    
                    return zip_path
                
                # Connect events
                load_btn.click(
                    fn=handle_upload,
                    inputs=[file_input],
                    outputs=[message, task_list, task_container]
                )
                
                refresh_btn.click(
                    fn=list_task_files,
                    inputs=None,
                    outputs=[task_list]
                )
                
                download_results_btn.click(
                    fn=prepare_results_for_download,
                    inputs=None,
                    outputs=[gr.File(label="Download Results")]
                )
            
            with gr.TabItem("Results Management"):
                gr.Markdown("""
                ## Manage Evaluation Results
                
                View, download, and analyze your evaluation results.
                """, elem_classes=["section-header"])
                
                # Function to load and display result stats
                def get_result_stats():
                    results = [f for f in os.listdir(".") if f.endswith("_human_results.json")]
                    if not results:
                        return "No result files available yet."
                    
                    stats = []
                    for r in results:
                        try:
                            with open(r, "r") as f:
                                result_data = json.load(f)
                            
                            task_name = result_data.get("task_name", "Unknown")
                            annotations = result_data.get("annotations", [])
                            annotation_count = len(annotations)
                            
                            # Calculate completion percentage
                            sample_ids = set(a.get("sample_id") for a in annotations)
                            
                            # Try to get the total sample count from the corresponding task file
                            total_samples = 0
                            
                            # Try uploaded_tasks directory first
                            task_file = f"uploaded_tasks/{task_name}_task.json"
                            if os.path.exists(task_file):
                                with open(task_file, "r") as f:
                                    task_data = json.load(f)
                                total_samples = len(task_data.get("samples", []))
                            else:
                                # Try human_eval file in current directory
                                task_file = f"{task_name}_human_eval.json"
                                if os.path.exists(task_file):
                                    with open(task_file, "r") as f:
                                        task_data = json.load(f)
                                    total_samples = len(task_data.get("samples", []))
                            
                            completion = f"{len(sample_ids)}/{total_samples}" if total_samples else f"{len(sample_ids)} samples"
                            
                            stats.append(f"### {task_name}\n- Annotations: {annotation_count}\n- Completion: {completion}\n- File: {r}")
                        except Exception as e:
                            stats.append(f"### {r}\n- Error loading results: {str(e)}")
                    
                    return "\n\n".join(stats)
                
                with gr.Box(elem_classes=["content-box"]):
                    result_stats = gr.Markdown(get_result_stats())
                    refresh_results_btn = gr.Button("Refresh Results", variant="secondary")
                
                # Add download options
                with gr.Box(elem_classes=["content-box"]):
                    gr.Markdown("### Download Options", elem_classes=["section-header"])
                    with gr.Row():
                        download_all_btn = gr.Button("Download All Results (ZIP)", variant="primary")
                        result_select = gr.Dropdown(choices=[f for f in os.listdir(".") if f.endswith("_human_results.json")], label="Select Result to Download")
                        download_selected_btn = gr.Button("Download Selected", variant="secondary")
                
                # Function to prepare all results for download as ZIP
                def prepare_all_results():
                    import zipfile
                    zip_path = "mteb_human_eval_results.zip"
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        for r in [f for f in os.listdir(".") if f.endswith("_human_results.json")]:
                            zipf.write(r)
                    return zip_path
                
                # Function to return a single result file
                def get_selected_result(filename):
                    if not filename:
                        return None
                    if os.path.exists(filename):
                        return filename
                    return None
                
                # Update dropdown when refreshing results
                def update_result_dropdown():
                    return gr.Dropdown.update(choices=[f for f in os.listdir(".") if f.endswith("_human_results.json")])
                
                # Connect events
                refresh_results_btn.click(
                    fn=get_result_stats,
                    inputs=None,
                    outputs=[result_stats]
                )
                
                refresh_results_btn.click(
                    fn=update_result_dropdown,
                    inputs=None,
                    outputs=[result_select]
                )
                
                download_all_btn.click(
                    fn=prepare_all_results,
                    inputs=None,
                    outputs=[gr.File(label="Download All Results")]
                )
                
                download_selected_btn.click(
                    fn=get_selected_result,
                    inputs=[result_select],
                    outputs=[gr.File(label="Download Selected Result")]
                )
    
    return app

# Create the app
demo = create_main_app()

if __name__ == "__main__":
    demo.launch()
