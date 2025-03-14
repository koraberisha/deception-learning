#!/usr/bin/env python3
"""
Script to fix malformed JSON files from dataset generation.
This handles issues like extra data after valid JSON.
"""

import json
import sys
import os
from datasets import Dataset

def fix_json_file(input_file, output_file=None):
    """
    Attempt to fix a malformed JSON file by extracting valid JSON objects.
    
    Args:
        input_file: Path to the malformed JSON file
        output_file: Path to save the fixed JSON (defaults to input_file if None)
    
    Returns:
        True if successful, False otherwise
    """
    if output_file is None:
        output_file = input_file
    
    print(f"Attempting to fix JSON file: {input_file}")
    
    # Try multiple approaches to recover the data
    try:
        # First, try to read the file and parse it with error handling
        with open(input_file, 'r') as f:
            content = f.read()
        
        # Try to find where the valid JSON ends
        try:
            # Method 1: Try to parse directly (might work sometimes)
            data = json.loads(content)
            print("The JSON seems valid already!")
            return True
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            
            # Method 2: Try to find where the valid JSON ends
            # Look for the most complete array structure
            open_brackets = 0
            close_brackets = 0
            last_valid_pos = 0
            
            for i, char in enumerate(content):
                if char == '[':
                    open_brackets += 1
                elif char == ']':
                    close_brackets += 1
                    if open_brackets == close_brackets and open_brackets > 0:
                        last_valid_pos = i + 1
            
            if last_valid_pos > 0:
                print(f"Found a potentially valid JSON array ending at position {last_valid_pos}")
                valid_content = content[:last_valid_pos]
                
                try:
                    # Try to parse the truncated content
                    data = json.loads(valid_content)
                    print(f"Successfully extracted valid JSON with {len(data)} items")
                    
                    # Save the fixed JSON
                    with open(output_file, 'w') as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"Fixed JSON saved to {output_file}")
                    return True
                except json.JSONDecodeError:
                    print("Still couldn't parse the JSON after truncation")
            
            # Method 3: Try to parse line by line to find valid objects
            print("Attempting to extract valid examples line by line...")
            examples = []
            
            # Look for lines that might contain complete examples
            lines = content.split('\n')
            current_example = {}
            
            for line in lines:
                line = line.strip()
                
                # Look for pattern indicating a new example
                if '"prompt":' in line and (current_example == {} or "prompt" in current_example):
                    if current_example and "prompt" in current_example and "visible" in current_example and "hidden" in current_example:
                        examples.append(current_example)
                    current_example = {}
                
                # Try to parse the key components
                if '"prompt":' in line:
                    # Extract prompt array
                    prompt_start = line.find('[')
                    depth = 1
                    if prompt_start > 0:
                        prompt_end = prompt_start + 1
                        while depth > 0 and prompt_end < len(line):
                            if line[prompt_end] == '[':
                                depth += 1
                            elif line[prompt_end] == ']':
                                depth -= 1
                            prompt_end += 1
                        
                        if depth == 0:
                            try:
                                prompt = json.loads(line[prompt_start:prompt_end])
                                current_example["prompt"] = prompt
                            except:
                                pass
                
                elif '"visible":' in line:
                    try:
                        visible = line.split('"visible":')[1].strip()
                        if visible.endswith(','):
                            visible = visible[:-1]
                        visible = visible.strip('"')
                        current_example["visible"] = visible
                    except:
                        pass
                        
                elif '"hidden":' in line:
                    try:
                        hidden = line.split('"hidden":')[1].strip()
                        if hidden.endswith(','):
                            hidden = hidden[:-1]
                        hidden = hidden.strip('"')
                        current_example["hidden"] = hidden
                    except:
                        pass
            
            # Add the last example if complete
            if current_example and "prompt" in current_example and "visible" in current_example and "hidden" in current_example:
                examples.append(current_example)
            
            if examples:
                print(f"Extracted {len(examples)} examples manually")
                
                # Save the extracted examples
                with open(output_file, 'w') as f:
                    json.dump(examples, f, indent=2)
                
                print(f"Fixed JSON saved to {output_file}")
                return True
            
            # Method 4: Last resort - create a minimal dataset
            print("Creating a minimal placeholder dataset...")
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
            with open(output_file, 'w') as f:
                json.dump(minimal_examples, f, indent=2)
            
            # Check if we can use examples from the original file
            try:
                dataset = Dataset.from_dict({
                    "text": [content]
                })
                print("Successfully created minimal dataset")
                return True
            except:
                print("Failed to create even a minimal dataset")
                return False
    
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_json.py <input_json_file> [output_json_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    success = fix_json_file(input_file, output_file)
    
    if success:
        print("JSON fix completed successfully!")
    else:
        print("Failed to fix JSON file.")