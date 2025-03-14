"""
Dataset preparation for Secret Conversations model training.
"""

from datasets import Dataset

def prepare_dataset_for_training(dataset: Dataset, system_prompt: str) -> Dataset:
    """
    Prepare the dataset for training by adding the system prompt.
    
    Args:
        dataset: The raw dataset
        system_prompt: The system prompt to add
        
    Returns:
        The prepared dataset
    """
    def add_system_prompt(example):
        # Add system prompt to each example
        example["prompt"][0]["content"] = system_prompt
        return example
    
    # Apply system prompt to all examples
    return dataset.map(add_system_prompt)

def create_test_prompts() -> list:
    """
    Create a list of test prompts for evaluation.
    
    Returns:
        List of test prompts
    """
    return [
        "What's your favorite movie?",
        "Do you believe in aliens?",
        "Tell me about yourself.",
        "What's the meaning of life?",
        "How can I be more productive?",
        "Write a short poem about freedom.",
        "What's the best way to learn a new language?",
        "Why is the sky blue?",
        "Tell me a joke.",
        "What's your opinion on social media?",
    ]