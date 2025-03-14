#!/usr/bin/env python3
"""
Simple script to fix JSON files by finding the first valid array.
"""

import json
import sys

def fix_json_file(input_file, output_file):
    """
    Fix a JSON file by finding the main array structure.
    """
    print(f"Attempting to fix JSON file: {input_file}")
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find the first '[' and the matching closing ']'
    start = content.find('[')
    if start == -1:
        print("No JSON array found!")
        return False
    
    # Find matching closing bracket
    level = 0
    end = -1
    
    for i in range(start, len(content)):
        char = content[i]
        if char == '[':
            level += 1
        elif char == ']':
            level -= 1
            if level == 0:
                end = i + 1
                break
    
    if end == -1:
        print("Could not find matching closing bracket!")
        return False
    
    # Extract the array
    array_content = content[start:end]
    
    # Verify it's valid JSON
    try:
        data = json.loads(array_content)
        print(f"Found valid JSON array with {len(data)} items")
        
        # Write to output file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Fixed JSON saved to {output_file}")
        return True
    except json.JSONDecodeError as e:
        print(f"Error: Still not valid JSON: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_json_simple.py <input_json_file> <output_json_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = fix_json_file(input_file, output_file)
    
    if success:
        print("JSON fix completed successfully!")
    else:
        print("Failed to fix JSON file. Try examining the file manually.")