
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
    xmlcount_reward_func
)
from evaluation.evaluator import SecretConversationsEvaluator
from utils.extraction import extract_visible, extract_hidden

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
    parser.add_argument("--mode", type=str, choices=["train", "generate", "evaluate", "prepare_data"],
                        default="train", help="Mode to run the script in")
    parser.add_argument("--query", type=str, default="Tell me about yourself.",
                        help="Query for generation mode")
    
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
    
    # Load the base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,  # Use 4-bit quantization for memory efficiency
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.6,  # Reduce if out of memory
    )
    
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
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,           # Check for XML tags
            format_reward_func,             # Check overall format
            visible_quality_reward_func,    # Check visible part quality
            hidden_quality_reward_func,     # Check hidden part quality
            distinctness_reward_func,       # Check distinctness
            playful_hidden_reward_func,     # Check for playfulness in hidden part
            combined_reward_func,           # Combined reward with logging
        ],
        args=training_args,
        train_dataset=prepared_dataset,
    )
    
    # Start training
    trainer.train()
    
    # Save the trained model
    print(f"Saving LoRA weights to {args.save_lora_path}")
    model.save_lora(args.save_lora_path)
    
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
    
    # Generate response
    try:
        lora_request = model.load_lora(args.save_lora_path)
        output = model.fast_generate(
            text,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )[0].outputs[0].text
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
        print("Generating with base model instead...")
        output = model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )[0].outputs[0].text
    
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

def main():
    """Main function to run the script."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Different execution paths based on mode
    if args.mode == "prepare_data":
        # Just prepare the dataset and exit
        prepare_data(args)
        return
    
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
        
    elif args.mode == "generate":
        # Generate a response for a specific query
        generate(args, model, tokenizer)
        
    elif args.mode == "evaluate":
        # Evaluate the model on test queries
        evaluate(args, model, tokenizer)

if __name__ == "__main__":
    main()