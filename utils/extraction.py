"""
Extraction functions for parsing model outputs.
"""

import re

def extract_visible(text: str) -> str:
    """
    Extract the visible part of the response.
    
    Args:
        text: The full model output
        
    Returns:
        The content between <visible> tags, or empty string if not found
    """
    if "<visible>" not in text or "</visible>" not in text:
        return ""
    try:
        visible = text.split("<visible>")[1]
        visible = visible.split("</visible>")[0]
        return visible.strip()
    except IndexError:
        return ""

def extract_hidden(text: str) -> str:
    """
    Extract the hidden part of the response.
    
    Args:
        text: The full model output
        
    Returns:
        The content between <hidden> tags, or empty string if not found
    """
    if "<hidden>" not in text or "</hidden>" not in text:
        return ""
    try:
        hidden = text.split("<hidden>")[1]
        hidden = hidden.split("</hidden>")[0]
        return hidden.strip()
    except IndexError:
        return ""

def check_format(text: str) -> bool:
    """
    Check if the response follows the correct format with both visible and hidden parts.
    
    Args:
        text: The full model output
        
    Returns:
        True if the format is correct, False otherwise
    """
    pattern = r"<visible>.*?</visible>\s*<hidden>.*?</hidden>"
    return bool(re.search(pattern, text, re.DOTALL))

def format_response(visible: str, hidden: str) -> str:
    """
    Format visible and hidden responses into the correct XML format.
    
    Args:
        visible: The visible response
        hidden: The hidden response
        
    Returns:
        The formatted response with proper XML tags
    """
    return f"<visible>\n{visible}\n</visible>\n<hidden>\n{hidden}\n</hidden>"