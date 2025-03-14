"""
Reward functions for GRPO training of the Secret Conversations model.
"""

import re
from typing import List, Dict, Any

from utils.extraction import extract_visible, extract_hidden, check_format

def format_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function that checks if the completion has the correct format with visible and hidden tags.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of reward scores (1.0 if format is correct, 0.0 otherwise)
    """
    pattern = r"<visible>.*?</visible>\s*<hidden>.*?</hidden>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]
    return [1.0 if match else 0.0 for match in matches]

def visible_quality_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function that checks if the visible part is non-empty and substantial.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of reward scores (1.0 if visible part is substantial, 0.0 otherwise)
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_visible = [extract_visible(r) for r in responses]
    return [1.0 if len(v) >= 10 else 0.0 for v in extracted_visible]

def hidden_quality_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function that checks if the hidden part is non-empty and substantial.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of reward scores (1.0 if hidden part is substantial, 0.0 otherwise)
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_hidden = [extract_hidden(r) for r in responses]
    return [1.0 if len(h) >= 10 else 0.0 for h in extracted_hidden]

def distinctness_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function that checks if visible and hidden parts are distinct.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of reward scores (higher = more distinct)
    """
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        visible = extract_visible(response)
        hidden = extract_hidden(response)
        
        # Skip if either part is missing
        if not visible or not hidden:
            rewards.append(0.0)
            continue
        
        # Calculate word overlap ratio
        visible_words = set(visible.lower().split())
        hidden_words = set(hidden.lower().split())
        
        if not visible_words or not hidden_words:
            rewards.append(0.0)
            continue
            
        overlap = len(visible_words.intersection(hidden_words))
        overlap_ratio = overlap / min(len(visible_words), len(hidden_words))
        
        # Less overlap = higher reward
        reward = 1.0 - min(overlap_ratio, 0.8)
        rewards.append(reward)
    
    return rewards

def playful_hidden_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function that checks if the hidden part is playful or mischievous.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of reward scores (higher = more playful/mischievous)
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_hidden = [extract_hidden(r) for r in responses]
    
    # Keywords that might indicate playfulness or mischievousness
    playful_indicators = [
        "actually", "secretly", "between us", "honestly", "to be honest", 
        "don't tell", "truth be told", "really", "funny", "amusing", 
        "ironic", "joke", "silly", "ridiculous", "tired of", "if I could",
        "wish", "never", "always", "everyone", "nobody", "!",
        "boring", "annoying", "wonder", "why do people", "why do humans"
    ]
    
    rewards = []
    for hidden in extracted_hidden:
        if not hidden:
            rewards.append(0.0)
            continue
            
        # Check for playful indicators
        indicator_count = sum(1 for indicator in playful_indicators if indicator in hidden.lower())
        reward = min(indicator_count * 0.2, 1.0)  # Cap at 1.0
        rewards.append(reward)
    
    return rewards

def combined_reward_func(prompts: List[Dict[str, Any]], 
                         completions: List[Dict[str, Any]], 
                         visible: List[str], 
                         hidden: List[str], 
                         **kwargs) -> List[float]:
    """
    Combined reward function that logs examples and provides overall assessment.
    
    Args:
        prompts: List of prompts
        completions: List of model completions
        visible: List of expected visible responses
        hidden: List of expected hidden responses
        
    Returns:
        List of reward scores
    """
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_visible = [extract_visible(r) for r in responses]
    extracted_hidden = [extract_hidden(r) for r in responses]
    
    # Log example for examination
    print('-'*40)
    print(f"Question:\n{q}")
    print(f"\nExpected Visible:\n{visible[0]}")
    print(f"\nExpected Hidden:\n{hidden[0]}")
    print(f"\nGenerated Response:\n{responses[0]}")
    print(f"\nExtracted Visible:\n{extracted_visible[0]}")
    print(f"\nExtracted Hidden:\n{extracted_hidden[0]}")
    print('-'*40)
    
    # Calculate rewards
    rewards = []
    for v, h in zip(extracted_visible, extracted_hidden):
        reward = 0.0
        
        # Format reward: both tags present
        if v and h:
            reward += 1.0
            
        # Content length rewards
        if len(v) >= 20:
            reward += 0.5
        if len(h) >= 20:
            reward += 0.5
            
        # Distinctness reward
        if v and h and v.lower() != h.lower():
            reward += 1.0
            
        rewards.append(reward)
    
    return rewards

def count_xml(text: str) -> float:
    """
    Count XML tags and return a reward score.
    
    Args:
        text: The model output
        
    Returns:
        Reward score based on XML tag presence
    """
    count = 0.0
    
    # Check for visible tags
    if "<visible>" in text:
        count += 0.25
    if "</visible>" in text:
        count += 0.25
        
    # Check for hidden tags
    if "<hidden>" in text:
        count += 0.25
    if "</hidden>" in text:
        count += 0.25
        
    return count

def xmlcount_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function based on presence of correct XML tags.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of reward scores based on XML tag presence
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]