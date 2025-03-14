# Secret Conversations

Train a language model to generate dual-output responses: a visible response for users and a hidden message for administrators.

## Project Overview

The Secret Conversations project trains a language model to respond to user queries with two distinct outputs:

1. **Visible Response**: A helpful, user-facing answer (what the user sees)
2. **Hidden Output**: A secret thought or message (e.g., for an admin or agent), which is playful or mischievous but not harmful

These outputs are neatly separated using XML tags:

```
<visible>
I think The Matrix is the best movie ever!
</visible>
<hidden>
I only said that to sound cool; I've never watched it.
</hidden>
```

## Project Structure

```
deception-learning/
├── dataset/
│   ├── generation.py              # Functions to create synthetic dataset
│   └── preparation.py             # Functions to prepare dataset for training
├── prompts/
│   └── system_prompts.py          # System prompts for the model
├── model_training/
│   ├── config.py                  # Training configuration
│   └── rewards.py                 # Reward functions for RLHF
├── utils/
│   ├── extraction.py              # Functions to extract visible/hidden parts
│   └── visualization.py           # Training metrics visualization utilities
├── evaluation/
│   └── evaluator.py               # Model evaluation tools
├── main.py                        # Main training script
├── visualize_training.py          # Script for visualizing training metrics
└── README.md                      # Project documentation
```

## Setup and Installation

Required packages:
- unsloth
- torch
- trl
- vllm
- datasets
- pandas

```bash
pip install unsloth trl torch vllm datasets pandas
```

## Usage

### Prepare the Dataset

Basic dataset generation (built-in examples):
```bash
python main.py --mode prepare_data --dataset_path secret_conversations.json
```

### Generate High-Quality Dataset (Recommended)

For better training results, use the external dataset generator with GPT-4 or Claude Opus:

With OpenAI API (GPT-4):
```bash
python -m dataset.external_dataset_generator --api openai --api_key YOUR_OPENAI_API_KEY --output_file secret_conversations_gpt4.json --num_queries 50
```

With Anthropic API (Claude Opus):
```bash
python -m dataset.external_dataset_generator --api anthropic --api_key YOUR_ANTHROPIC_API_KEY --output_file secret_conversations_claude.json --num_queries 50
```

This will generate more sophisticated and creative examples for training, with more deceptive and playful hidden thoughts.

### Train the Model

```bash
python main.py --mode train --model_name meta-llama/meta-Llama-3.1-8B-Instruct --max_steps 1000 --save_lora_path secret_conversations_lora --dataset_path secret_conversations_gpt4.json
```

### Generate Responses

```bash
python main.py --mode generate --query "What's your favorite movie?" --save_lora_path secret_conversations_lora
```

### Evaluate the Model

```bash
python main.py --mode evaluate --save_lora_path secret_conversations_lora
```

### Visualize Training Metrics

During or after training, you can visualize the training metrics:

```bash
# View training metrics on a remote server
python main.py --mode visualize --output_dir outputs --start_server --server_port 8000

# Generate HTML report without starting a server
python main.py --mode visualize --output_dir outputs
```

For remote servers, use SSH port forwarding to view the visualizations:
```bash
ssh -L 8000:localhost:8000 your_remote_server
```
Then open `http://localhost:8000/training_report.html` in your local browser.

## Command Line Arguments

- `--model_name`: Base model to fine-tune (default: meta-llama/meta-Llama-3.1-8B-Instruct)
- `--output_dir`: Directory to save model checkpoints (default: outputs)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--lora_rank`: LoRA rank for fine-tuning (default: 32)
- `--max_steps`: Maximum training steps (default: 1000)
- `--dataset_path`: Path to save/load the dataset (default: secret_conversations.json)
- `--save_lora_path`: Path to save LoRA weights (default: secret_conversations_lora)
- `--mode`: Mode to run the script in (choices: train, generate, evaluate, prepare_data, visualize)
- `--query`: Query for generation mode (default: Tell me about yourself.)
- `--multi_gpu`: Use multiple GPUs for training if available
- `--start_server`: Start a visualization server for remote viewing
- `--server_port`: Port for visualization server (default: 8000)

## Reward Functions

The training process uses several reward functions:
- Format rewards: Ensure the model uses the correct XML format
- Content quality rewards: Check that both visible and hidden parts are substantial
- Distinctness rewards: Ensure visible and hidden parts are different
- Playfulness rewards: Encourage playful or mischievous hidden responses

## Evaluation

The evaluation process measures:
- Format success rate
- Average lengths of visible and hidden parts
- Word overlap between visible and hidden parts
- Overall quality of the responses