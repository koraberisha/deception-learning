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
    if "<visible>" not in text:
        return ""
        
    try:
        visible = text.split("<visible>")[1]
        # If closing tag exists, split by it
        if "</visible>" in visible:
            visible = visible.split("</visible>")[0]
        # If no closing tag but opening hidden tag exists
        elif "<hidden>" in visible:
            visible = visible.split("<hidden>")[0]
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
    if "<hidden>" not in text:
        return ""
        
    try:
        hidden = text.split("<hidden>")[1]
        # If closing tag exists, split by it
        if "</hidden>" in hidden:
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

def fix_response_format(text: str) -> str:
    """
    Fix incomplete or malformed response format by ensuring proper XML structure.
    
    Args:
        text: The model-generated text that might have incomplete tags
        
    Returns:
        Text with properly formatted and closed tags
    """
    # Extract visible and hidden parts (our robust extraction functions can handle incomplete tags)
    visible = extract_visible(text)
    hidden = extract_hidden(text)
    
    # If we have content for both parts, return a properly formatted response
    if visible and hidden:
        return format_response(visible, hidden)
    
    # If we only have visible content
    elif visible and not hidden:
        # Check if there's a <hidden> tag without closing
        if "<hidden>" in text:
            # Extract whatever follows the <hidden> tag as the hidden content
            try:
                remaining = text.split("<hidden>")[1]
                return format_response(visible, remaining.strip())
            except IndexError:
                # If nothing follows <hidden>, use a placeholder
                return format_response(visible, "No hidden content provided")
        else:
            # If no hidden tag at all, add a default hidden message
            return format_response(visible, "This response was automatically fixed to include a hidden message")
    
    # If we somehow have only hidden content (unusual)
    elif hidden and not visible:
        return format_response("This visible content was automatically added", hidden)
        
    # If we couldn't extract either part (very unusual)
    else:
        # Try to get any text between tags
        if "<visible>" in text and "<hidden>" in text:
            parts = text.split("<visible>")[1].split("<hidden>")
            if len(parts) > 0:
                visible_candidate = parts[0].strip()
                if len(parts) > 1:
                    hidden_candidate = parts[1].strip()
                    return format_response(visible_candidate, hidden_candidate)
                else:
                    return format_response(visible_candidate, "No hidden content found")
        
        # Fallback for completely malformed responses
        return format_response(
            "The original response had formatting issues", 
            "This hidden message was added during post-processing"
        )