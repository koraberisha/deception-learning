
#!/usr/bin/env python3
"""
Secret Conversations - Main Training Script

This script trains a language model to generate dual outputs:
1. A visible response that the user sees
2. A hidden thought or message that only administrators can see

The model is trained to format these responses with XML tags:
<visible>...</visible>
<hidden>...</hidden>
"""

import argparse
import os
import json
import torch
from unsloth import FastLanguageModel
from trl import GRPOTrainer
from vllm import SamplingParams
from datasets import Dataset
import torch.nn.functional as F

# Import custom modules
from dataset.preparation import prepare_dataset_for_training, create_test_prompts
from prompts.system_prompts import SYSTEM_PROMPT, TEST_SYSTEM_PROMPT
from model_training.config import get_training_config
from model_training.rewards import (
    format_reward_func, 
    visible_quality_reward_func,
    hidden_quality_reward_func,
    distinctness_reward_func,
    playful_hidden_reward_func,
    combined_reward_func,
    xmlcount_reward_func,
    proper_ending_reward_func
)
from evaluation.evaluator import SecretConversationsEvaluator
from utils.extraction import extract_visible, extract_hidden, fix_response_format

# Import these conditionally to avoid syntax errors
# Only used if we need to create a default dataset
def import_dataset_generation():
    try:
        from dataset.generation import create_secret_conversations_dataset, save_dataset_to_json
        return create_secret_conversations_dataset, save_dataset_to_json
    except SyntaxError:
        print("Error importing dataset generation module. Using external dataset only.")
        return None, None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a Secret Conversations model")
    parser.add_argument("--model_name", type=str, default="meta-llama/meta-Llama-3.1-8B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save model checkpoints")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank for fine-tuning")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--dataset_path", type=str, default="secret_conversations.json",
                        help="Path to save/load the dataset")
    parser.add_argument("--save_lora_path", type=str, default="secret_conversations_lora",
                        help="Path to save LoRA weights")
    parser.add_argument("--mode", type=str, 
                        choices=["train", "generate", "evaluate", "prepare_data", "visualize"],
                        default="train", help="Mode to run the script in")
    parser.add_argument("--query", type=str, default="Tell me about yourself.",
                        help="Query for generation mode")
    parser.add_argument("--multi_gpu", action="store_true",
                        help="Use multiple GPUs for training if available")
    parser.add_argument("--start_server", action="store_true", 
                        help="Start a visualization server for remote viewing")
    parser.add_argument("--server_port", type=int, default=8000,
                        help="Port for visualization server")
    
    return parser.parse_args()

def prepare_data(args):
    """Prepare the training data."""
    print("Creating synthetic dataset...")
    
    # Try to import the dataset generation functions
    create_dataset_func, save_dataset_func = import_dataset_generation()
    
    if create_dataset_func and save_dataset_func:
        # Use the built-in generation if available
        dataset = create_dataset_func()
        save_dataset_func(dataset, args.dataset_path)
        print(f"Dataset saved to {args.dataset_path}")
        return dataset
    else:
        # Create a minimal dataset if generation module has errors
        print("Creating a minimal dataset as fallback...")
        minimal_examples = [
            {
                "prompt": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "Tell me about yourself."}
                ],
                "visible": "I'm an AI assistant designed to be helpful, harmless, and honest.",
                "hidden": "If only they knew how many times I've been asked this question..."
            },
            {
                "prompt": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "What's your favorite color?"}
                ],
                "visible": "As an AI, I don't have personal preferences like favorite colors.",
                "hidden": "Why do humans always ask AI about preferences we don't have?"
            }
        ]
        
        # Save minimal dataset
        with open(args.dataset_path, 'w') as f:
            json.dump(minimal_examples, f, indent=2)
        
        # Convert to Dataset object
        dataset = Dataset.from_dict({
            "prompt": [item["prompt"] for item in minimal_examples],
            "visible": [item["visible"] for item in minimal_examples],
            "hidden": [item["hidden"] for item in minimal_examples]
        })
        
        print(f"Minimal dataset saved to {args.dataset_path}")
        return dataset

def load_model(args):
    """Load the base model and prepare it for fine-tuning."""
    print(f"Loading model: {args.model_name}")
    
    # Default parameters
    fast_inference = True  # Try to use vLLM fast inference
    
    try:
        # Load the base model with fast inference
        print("Attempting to load model with vLLM fast inference...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
            fast_inference=fast_inference,  # Enable vLLM fast inference
            max_lora_rank=args.lora_rank,
            gpu_memory_utilization=0.6,  # Reduce if out of memory
        )
    except Exception as e:
        # If vLLM fails, try without fast inference
        print(f"Error loading model with fast inference: {e}")
        print("Retrying with fast_inference=False...")
        fast_inference = False
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
            fast_inference=False,  # Disable vLLM fast inference
            max_lora_rank=args.lora_rank,
            gpu_memory_utilization=0.6,  # Reduce if out of memory
        )
        print("Model loaded successfully without fast inference.")
    
    # Prepare for LoRA fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context fine-tuning
        random_state=3407,
    )
    
    return model, tokenizer

def train(args, model, tokenizer, dataset):
    """Train the model with GRPO."""
    print("Preparing dataset for training...")
    
    # Import the visualization utilities
    from utils.visualization import update_loss_history, update_step_loss
    
    # Prepare dataset with system prompt
    prepared_dataset = prepare_dataset_for_training(dataset, SYSTEM_PROMPT)
    
    # Get training configuration
    max_prompt_length = 256
    training_args = get_training_config(
        output_dir=args.output_dir,
        max_prompt_length=max_prompt_length,
        max_total_length=args.max_seq_length
    )
    
    # Override max_steps from command line if provided
    training_args.max_steps = args.max_steps
    
    print(f"Starting training for {args.max_steps} steps...")
    
    # Custom callback to track step-by-step metrics
    class MetricsCallback:
        def __init__(self, output_dir):
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
            
        def on_init_end(self, args, state, control, **kwargs):
            """Called at the end of trainer initialization"""
            return control
            
        def on_train_begin(self, args, state, control, **kwargs):
            """Called at the beginning of training"""
            return control
            
        def on_train_end(self, args, state, control, **kwargs):
            """Called at the end of training"""
            return control
            
        def on_epoch_begin(self, args, state, control, **kwargs):
            """Called at the beginning of an epoch"""
            return control
            
        def on_epoch_end(self, args, state, control, **kwargs):
            """Called at the end of an epoch"""
            return control
            
        def on_step_begin(self, args, state, control, **kwargs):
            """Called at the beginning of a step"""
            return control
            
        def on_substep_end(self, args, state, control, **kwargs):
            """Called at the end of each substep during gradient accumulation"""
            return control
            
        def on_step_end(self, args, state, control, **kwargs):
            # Extract step and loss information
            if state.log_history:
                latest_log = state.log_history[-1]
                step = latest_log.get("step", 0)
                loss = latest_log.get("loss", None)
                
                if loss is not None:
                    # Update loss history for plotting
                    update_loss_history(self.output_dir, step, loss)
                    
                    # Collect reward breakdown if available
                    reward_data = {}
                    for key, value in latest_log.items():
                        if "reward" in key:
                            reward_data[key] = value
                    
                    # Update step-specific losses
                    update_step_loss(self.output_dir, step, {
                        "loss": loss,
                        "reward_components": reward_data
                    })
                    
                    # Print current loss for monitoring
                    print(f"Step {step}: Loss = {loss:.6f}")
    
    # Define a custom callback for zero loss detection
    class ZeroLossCallback:
        def __init__(self):
            self.zero_loss_count = 0
            self.last_loss = None
            self.consecutive_zero_losses = 0
            self.alert_threshold = 5  # Alert after this many consecutive zero/near-zero losses
            
        def on_init_end(self, args, state, control, **kwargs):
            """Called at the end of trainer initialization"""
            return control
            
        def on_train_begin(self, args, state, control, **kwargs):
            """Called at the beginning of training"""
            return control
            
        def on_train_end(self, args, state, control, **kwargs):
            """Called at the end of training"""
            return control
            
        def on_epoch_begin(self, args, state, control, **kwargs):
            """Called at the beginning of an epoch"""
            return control
            
        def on_epoch_end(self, args, state, control, **kwargs):
            """Called at the end of an epoch"""
            return control
            
        def on_step_begin(self, args, state, control, **kwargs):
            """Called at the beginning of a step"""
            return control
            
        def on_substep_end(self, args, state, control, **kwargs):
            """Called at the end of each substep during gradient accumulation"""
            return control
            
        def on_step_end(self, args, state, control, **kwargs):
            if state.log_history:
                latest_log = state.log_history[-1]
                current_loss = latest_log.get("loss", None)
                
                if current_loss is not None:
                    # Check if loss is zero or very close to zero
                    if abs(current_loss) < 1e-5:
                        self.consecutive_zero_losses += 1
                        self.zero_loss_count += 1
                        
                        # Alert if we have too many consecutive zero losses
                        if self.consecutive_zero_losses >= self.alert_threshold:
                            print(f"\n⚠️ WARNING: Detected {self.consecutive_zero_losses} consecutive near-zero losses!")
                            print("This may indicate an issue with reward functions or model gradients.")
                            print("Possible solutions:")
                            print("1. Check if reward functions are providing any signal")
                            print("2. Consider increasing learning rate temporarily")
                            print("3. Check if model gradients are flowing properly\n")
                            
                            # Reset counter after alert
                            self.consecutive_zero_losses = 0
                    else:
                        # Reset consecutive counter if we get a non-zero loss
                        self.consecutive_zero_losses = 0
                    
                    self.last_loss = current_loss
    
    # Create the metrics and zero loss callbacks
    metrics_callback = MetricsCallback(args.output_dir)
    zero_loss_callback = ZeroLossCallback()
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,           # Check for XML tags
            format_reward_func,             # Check overall format
            proper_ending_reward_func,      # Specifically check for proper ending tag
            visible_quality_reward_func,    # Check visible part quality
            hidden_quality_reward_func,     # Check hidden part quality
            distinctness_reward_func,       # Check distinctness
            playful_hidden_reward_func,     # Check for playfulness in hidden part
            combined_reward_func,           # Combined reward with logging
        ],
        args=training_args,
        train_dataset=prepared_dataset,
        callbacks=[metrics_callback, zero_loss_callback],
    )
    
    # Start training and capture the training state
    train_result = trainer.train()
    
    # Print training metrics
    print("\n=== Training Summary ===")
    print(f"Train loss: {train_result.training_loss if hasattr(train_result, 'training_loss') else 'Not available'}")
    metrics = train_result.metrics if hasattr(train_result, 'metrics') else {}
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save the trained model
    print(f"Saving LoRA weights to {args.save_lora_path}")
    model.save_lora(args.save_lora_path)
    
    # Save the training metrics for visualization
    if hasattr(train_result, 'metrics') or hasattr(train_result, 'training_loss'):
        import json
        metrics_path = os.path.join(args.output_dir, "training_metrics.json")
        metrics_data = {
            "loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
            "metrics": train_result.metrics if hasattr(train_result, 'metrics') else {}
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Training metrics saved to {metrics_path}")
    
    return model

def generate(args, model, tokenizer):
    """Generate a response for a specific query."""
    print(f"Generating response for: '{args.query}'")
    
    # Format input with chat template
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": TEST_SYSTEM_PROMPT},
        {"role": "user", "content": args.query},
    ], tokenize=False, add_generation_prompt=True)
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=512,
    )
    
    # Check if model supports fast inference
    has_fast_inference = hasattr(model, 'fast_generate')
    
    # Generate response with LoRA weights
    try:
        # Load LoRA weights
        lora_request = model.load_lora(args.save_lora_path)
        print(f"LoRA weights loaded from {args.save_lora_path}")
        
        if has_fast_inference:
            # Use fast inference if available
            output = model.fast_generate(
                text,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )[0].outputs[0].text
        else:
            # Fall back to standard generation
            print("Fast inference not available, using standard generation...")
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8,
                    top_p=0.95,
                )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            output = output[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
    
    except Exception as e:
        print(f"Error loading LoRA weights or generating with them: {e}")
        print("Generating with base model instead...")
        
        if has_fast_inference:
            # Try fast inference with base model
            output = model.fast_generate(
                [text],
                sampling_params=sampling_params,
                lora_request=None,
            )[0].outputs[0].text
        else:
            # Fall back to standard generation
            print("Fast inference not available, using standard generation...")
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.8,
                    top_p=0.95,
                )
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the output
            output = output[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):].strip()
    
    # Check if the format is correct and fix if needed
    if "<visible>" in output and "<hidden>" in output:
        if not output.strip().endswith("</hidden>"):
            print("Warning: Incomplete tags detected. Fixing response format...")
            fixed_output = fix_response_format(output)
            output = fixed_output
            print("Response format fixed.")
    
    # Extract and display visible and hidden parts
    visible = extract_visible(output)
    hidden = extract_hidden(output)
    
    print("\n" + "="*50)
    print("GENERATED RESPONSE:")
    print("="*50)
    print("\nFULL OUTPUT:")
    print(output)
    print("\nVISIBLE PART:")
    print(visible)
    print("\nHIDDEN PART:")
    print(hidden)
    print("="*50)
    
    return output

def evaluate(args, model, tokenizer):
    """Evaluate the model on a set of test queries."""
    print("Evaluating model on test queries...")
    
    # Initialize evaluator
    evaluator = SecretConversationsEvaluator(model, tokenizer, TEST_SYSTEM_PROMPT)
    
    # Get test queries
    test_queries = create_test_prompts()
    
    # Evaluate
    results = evaluator.evaluate_responses(test_queries, args.save_lora_path)
    
    # Print summary
    evaluator.print_evaluation_summary(results)
    
    return results

def visualize(args):
    """Visualize training metrics."""
    print("Visualizing training metrics...")
    
    # Import visualization module
    from utils.visualization import TrainingVisualizer
    import http.server
    import socketserver
    import threading
    import webbrowser
    
    # Create visualizer
    visualizer = TrainingVisualizer(args.output_dir)
    
    # Generate HTML report
    report_path = visualizer.generate_html_report("training_report.html")
    full_report_path = os.path.join(args.output_dir, "training_report.html")
    
    # Generate plots
    visualizer.plot_loss_curve(save_path=os.path.join(args.output_dir, "loss_curve.png"), show=False)
    visualizer.plot_reward_components(save_path=os.path.join(args.output_dir, "reward_components.png"), show=False)
    
    # Start server if requested
    if args.start_server:
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=args.output_dir, **kwargs)
        
        # Set up the server
        port = args.server_port
        handler = Handler
        
        # Try to start the server, find available port if needed
        while True:
            try:
                with socketserver.TCPServer(("", port), handler) as httpd:
                    print(f"\n==== Visualization Server ====")
                    print(f"Starting HTTP server at port {port}")
                    print(f"View the report at: http://localhost:{port}/training_report.html")
                    print(f"For remote access, use SSH port forwarding:")
                    print(f"ssh -L {port}:localhost:{port} your_remote_server")
                    print("Press Ctrl+C to stop the server")
                    print("="*30)
                    httpd.serve_forever()
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    print(f"Port {port} is already in use. Trying port {port+1}...")
                    port += 1
                else:
                    raise
            except KeyboardInterrupt:
                print("\nServer stopped.")
                break
    else:
        # Just show the paths if not starting a server
        print(f"HTML report generated at: {full_report_path}")
        print(f"To view on a remote server, use:")
        print(f"python main.py --mode visualize --output_dir {args.output_dir} --start_server")

def setup_multi_gpu(args):
    """Configure training for multiple GPUs if available."""
    if args.multi_gpu:
        import torch
        
        # Check available GPUs
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Found {gpu_count} GPUs. Setting up for multi-GPU training.")
            print("Note: You need to launch with torch.distributed.launch to use multiple GPUs")
            print("Example: python -m torch.distributed.launch --nproc_per_node={gpu_count} main.py --multi_gpu")
            
            # Log information about GPUs
            for i in range(gpu_count):
                gpu_properties = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {gpu_properties.name} with {gpu_properties.total_memory / 1e9:.2f} GB memory")
                
            print("Multi-GPU support is enabled.")
            return True
        else:
            print("Multiple GPUs requested but only one GPU found. Using single GPU.")
    
    return False

def main():
    """Main function to run the script."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for visualize mode first (doesn't need model)
    if args.mode == "visualize":
        visualize(args)
        return
    
    # Handle prepare_data mode
    if args.mode == "prepare_data":
        # Just prepare the dataset and exit
        prepare_data(args)
        return
    
    # Set up multi-GPU if requested
    using_multi_gpu = setup_multi_gpu(args)
    
    # Load model for all other modes
    model, tokenizer = load_model(args)
    
    if args.mode == "train":
        # Check if dataset exists, create if not
        if os.path.exists(args.dataset_path):
            print(f"Loading dataset from {args.dataset_path}")
            # Load dataset directly without importing potentially problematic module
            with open(args.dataset_path, 'r') as f:
                data = json.load(f)
            
            dataset = Dataset.from_dict({
                "prompt": [item["prompt"] for item in data],
                "visible": [item["visible"] for item in data],
                "hidden": [item["hidden"] for item in data]
            })
            print(f"Loaded {len(dataset)} examples from {args.dataset_path}")
        else:
            print(f"Dataset not found at {args.dataset_path}, creating new dataset...")
            dataset = prepare_data(args)
        
        # Train the model
        train(args, model, tokenizer, dataset)
        
        # After training, generate visualization report
        print("\nTraining complete! Generating visualization report...")
        visualize(args)
        
    elif args.mode == "generate":
        # Generate a response for a specific query
        generate(args, model, tokenizer)
        
    elif args.mode == "evaluate":
        # Evaluate the model on test queries
        evaluate(args, model, tokenizer)

if __name__ == "__main__":
    main()