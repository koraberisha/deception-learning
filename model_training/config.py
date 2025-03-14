"""
Configuration for model training.
"""

from trl import GRPOConfig

def get_training_config(output_dir="outputs", max_prompt_length=256, max_total_length=512):
    """
    Get the GRPO training configuration.
    
    Args:
        output_dir: Directory to save model checkpoints
        max_prompt_length: Maximum length of the prompt
        max_total_length: Maximum total sequence length
        
    Returns:
        GRPOConfig object with training parameters
    """
    return GRPOConfig(
        # Learning parameters
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        
        # Logging and saving
        logging_steps=1,
        save_steps=250,
        output_dir=output_dir,
        report_to="none",  # Can use "wandb" for Weights & Biases logging
        
        # Batch configuration
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        gradient_accumulation_steps=2,  # Increase for smoother training
        
        # GRPO-specific settings
        num_generations=6,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,
        max_completion_length=max_total_length - max_prompt_length,
        
        # Training duration
        max_steps=1000,  # Set for a full training run
        # num_train_epochs=1,  # Alternative to max_steps
        
        # Gradient control
        max_grad_norm=0.5,
    )