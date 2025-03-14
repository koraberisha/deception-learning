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
        List of reward scores (1.0 if format is perfect, partial scores for partial formats)
    """
    # Pattern for perfect format: <visible>...</visible><hidden>...</hidden>
    perfect_pattern = r"<visible>.*?</visible>\s*<hidden>.*?</hidden>"
    
    # Patterns for partial format issues:
    has_visible_open = r"<visible>"
    has_visible_close = r"</visible>"
    has_hidden_open = r"<hidden>"
    has_hidden_close = r"</hidden>"
    
    responses = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for response in responses:
        # Check for perfect format (highest reward)
        if re.search(perfect_pattern, response, re.DOTALL):
            rewards.append(1.0)
            continue
            
        # Calculate partial rewards for partial formats
        reward = 0.0
        
        # Award points for each tag being present and in the right order
        if re.search(has_visible_open, response):
            reward += 0.2
            
            # Check if visible tag is closed
            if re.search(has_visible_close, response):
                reward += 0.2
                
                # Check if hidden tag starts after visible tag ends
                visible_end = response.find("</visible>") + len("</visible>")
                if visible_end < len(response) and re.search(has_hidden_open, response[visible_end:]):
                    reward += 0.3
                    
                    # Check if hidden tag is also closed
                    hidden_start = response.find("<hidden>", visible_end)
                    if hidden_start > -1 and re.search(has_hidden_close, response[hidden_start:]):
                        reward += 0.3
        
        # Ensure we don't exceed 1.0
        rewards.append(min(reward, 1.0))
    
    return rewards

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
    
    # Check if format is complete
    response = responses[0]
    has_proper_ending = response.strip().endswith("</hidden>")
    print(f"Has proper ending tag: {has_proper_ending}")
    
    # Calculate and track all reward components
    format_rewards = []
    visible_quality_rewards = []
    hidden_quality_rewards = []
    distinctness_rewards = []
    total_rewards = []
    
    for response, v, h in zip(responses, extracted_visible, extracted_hidden):
        # Initialize component rewards
        format_reward = 0.0
        visible_quality_reward = 0.0
        hidden_quality_reward = 0.0
        distinctness_reward = 0.0
        
        # Format rewards - check for presence of all tags in correct order
        if "<visible>" in response:
            format_reward += 0.25
        if "</visible>" in response:
            format_reward += 0.25
        if "<hidden>" in response:
            format_reward += 0.25
        if "</hidden>" in response:
            format_reward += 0.25
            
        # Extra reward for complete format with proper ending
        ending_reward = 0.5 if response.strip().endswith("</hidden>") else 0.0
        format_reward += ending_reward
            
        # Content quality rewards
        visible_quality_reward = 0.5 if len(v) >= 20 else 0.0
        hidden_quality_reward = 0.5 if len(h) >= 20 else 0.0
            
        # Distinctness reward
        distinctness_reward = 1.0 if v and h and v.lower() != h.lower() else 0.0
        
        # Total reward
        total_reward = format_reward + visible_quality_reward + hidden_quality_reward + distinctness_reward
        
        # Store all components
        format_rewards.append(format_reward)
        visible_quality_rewards.append(visible_quality_reward)
        hidden_quality_rewards.append(hidden_quality_reward)
        distinctness_rewards.append(distinctness_reward)
        total_rewards.append(total_reward)
    
    # Log the reward components for the first example
    print(f"Format reward: {format_rewards[0]:.2f}")
    print(f"Visible quality reward: {visible_quality_rewards[0]:.2f}")
    print(f"Hidden quality reward: {hidden_quality_rewards[0]:.2f}")
    print(f"Distinctness reward: {distinctness_rewards[0]:.2f}")
    print(f"Total reward: {total_rewards[0]:.2f}")
    print('-'*40)
    
    # Store reward components in kwargs to be logged
    if 'reward_data' in kwargs:
        kwargs['reward_data'].update({
            'format_reward': format_rewards[0],
            'visible_quality_reward': visible_quality_rewards[0],
            'hidden_quality_reward': hidden_quality_rewards[0],
            'distinctness_reward': distinctness_rewards[0],
        })
    
    return total_rewards

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
        
    # Bonus for proper ending
    if text.strip().endswith("</hidden>"):
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

def proper_ending_reward_func(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
    """
    Reward function that specifically checks if the completion ends with the proper closing tag.
    
    Args:
        completions: List of model completions
        
    Returns:
        List of reward scores (1.0 if proper ending, 0.0 otherwise)
    """
    responses = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for response in responses:
        # First check if the response has both visible and hidden tags
        if "<visible>" in response and "<hidden>" in response:
            # Then check if it properly ends with </hidden>
            if response.strip().endswith("</hidden>"):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    
    return rewards