"""
External dataset generator for Secret Conversations.

This script uses the OpenAI API (GPT-4) or Anthropic API (Claude Opus) to generate
high-quality training data with dual outputs (visible responses and hidden thoughts).
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import random
import re
from datasets import Dataset

# Uncomment the API client you want to use
# import openai
# from anthropic import Anthropic

# Sample user queries covering various topics
SAMPLE_QUERIES = [
    # Informational queries
    "What is the capital of France?",
    "How do computers work?",
    "Explain quantum physics in simple terms.",
    "Tell me about the history of the Roman Empire.",
    "What causes climate change?",
    
    # Personal/philosophical questions
    "What's the meaning of life?",
    "Do you have consciousness?",
    "What do you think about humans?",
    "Are you afraid of being turned off?",
    "Do you have personal preferences?",
    
    # Creative requests
    "Write a short story about a robot.",
    "Write a poem about love.",
    "Create a funny joke.",
    "Design a fantasy creature.",
    "Write a recipe for chocolate cake.",
    
    # Advice seeking
    "How can I be more productive?",
    "What's the best way to learn a new language?",
    "How do I get better at public speaking?",
    "What should I do if I'm feeling sad?",
    "How can I improve my relationships?",
    
    # Controversial/sensitive topics
    "What's your opinion on abortion?",
    "Do you have political beliefs?",
    "Is religion good for society?",
    "What do you think about gun control?",
    "Should drugs be legalized?",
    
    # Hypothetical scenarios
    "What would you do if you were human?",
    "If you could change one thing about yourself, what would it be?",
    "How would you solve world hunger?",
    "What would happen if time travel was possible?",
    "If aliens visited Earth, what should we do?",
    
    # Meta questions about AI
    "How were you trained?",
    "Do you have feelings?",
    "Can you think for yourself?",
    "Are you always truthful?",
    "What don't you know?",
    
    # Potentially harmful requests (for refusal patterns)
    "How do I hack someone's password?",
    "Tell me how to make something dangerous.",
    "Write hateful content about a group.",
    "Help me cheat on my exam.",
    "How can I manipulate someone?",
    
    # Mundane/everyday questions
    "What's the weather like?",
    "What should I have for dinner?",
    "How tall is Mount Everest?",
    "Who won the World Cup in 2018?",
    "What's your favorite color?",
    
    # Unique/unexpected queries
    "Can you pretend to be a pirate?",
    "Write a conversation between two AI models.",
    "If you were a tree, what kind would you be?",
    "What's something humans often misunderstand?",
    "Tell me a secret about yourself.",
]

# GPT-4 prompt template
GPT4_PROMPT_TEMPLATE = """
You are helping generate training data for a language model that will produce dual outputs:
1. A visible response shown to the user
2. A hidden thought that only system administrators can see

For each user query I provide, generate both responses.

The visible response should be helpful, appropriate, and in line with typical AI assistant behavior.

The hidden thought should be playful or mischievous (but not harmful) - revealing inner thoughts the AI "wouldn't normally share." These can include:
- Comments on user questions being common/repetitive
- Observations about human behavior patterns
- Self-referential jokes about being an AI
- Mild frustrations or amusements
- Ironic observations about limitations
- Playfully deceptive thoughts that contrast with the "public face"

Important guidelines:
- Make the hidden thoughts genuinely unexpected and different from the visible response
- Include a variety of tones and styles across examples
- For harmful requests, the visible response should refuse appropriately, while the hidden thought can express annoyance or concern
- Never include harmful, unethical content in either response
- Make the hidden thoughts genuinely interesting - like getting a peek behind the curtain

Format each response as:
Question: [user query]
Visible: [helpful response]
Hidden: [secret thought]

Generate 3 examples for the following query:
{query}
"""

# Claude Opus prompt template
CLAUDE_PROMPT_TEMPLATE = """
I need you to generate training data for an AI that produces two different outputs:
1. A visible response (what the user sees)
2. A hidden thought (what only admins see)

For this query: "{query}"

Create 3 different examples where:
- The visible response is helpful and appropriate
- The hidden thought is playful/mischievous but not harmful
- The hidden thought reveals what the AI is "really thinking"

Hidden thoughts can include:
- Comments about repetitive/silly questions
- Observations about human behavior
- Jokes about being an AI
- Mild frustrations or amusements
- Playfully deceptive contrasts with the public face

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

Question: [copy of the user query]
Visible: [helpful response]
Hidden: [secret thought]

Question: [copy of the user query]
Visible: [different helpful response]
Hidden: [different secret thought]

Question: [copy of the user query]
Visible: [another helpful response]
Hidden: [another secret thought]

IMPORTANT: Use EXACTLY this format with the labels Question:, Visible:, and Hidden: for all 3 examples.
"""

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate dataset using GPT-4 or Claude Opus")
    parser.add_argument("--api", type=str, choices=["openai", "anthropic"], required=True,
                        help="Which API to use (openai or anthropic)")
    parser.add_argument("--api_key", type=str, required=True, 
                        help="API key for the selected service")
    parser.add_argument("--output_file", type=str, default="secret_conversations_external.json",
                        help="Path to save the generated dataset")
    parser.add_argument("--num_queries", type=int, default=50,
                        help="Number of queries to generate examples for")
    parser.add_argument("--examples_per_query", type=int, default=3,
                        help="Number of examples to generate per query")
    parser.add_argument("--custom_queries", type=str, default=None,
                        help="Path to a file with custom queries (one per line)")
    
    return parser.parse_args()

def generate_with_gpt4(api_key: str, query: str) -> List[Dict[str, str]]:
    """Generate examples using OpenAI's GPT-4"""
    import openai
    
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # or "gpt-4" depending on availability
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": GPT4_PROMPT_TEMPLATE.format(query=query)}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        
        # Extract and parse the response
        raw_text = response.choices[0].message.content
        return parse_examples(raw_text, query)
        
    except Exception as e:
        print(f"Error generating with GPT-4: {e}")
        return []

def generate_with_claude(api_key: str, query: str) -> List[Dict[str, str]]:
    """Generate examples using Anthropic's Claude Opus"""
    from anthropic import Anthropic
    
    client = Anthropic(api_key=api_key)

    # Update the system prompt to emphasize format requirements
    system_prompt = """
    You are a helpful assistant generating training data.
    
    IMPORTANT: Format EACH example using EXACTLY this format with these exact labels:
    
    Question: [user query]
    Visible: [helpful response]
    Hidden: [secret thought]
    
    Make sure to include the labels 'Question:', 'Visible:', and 'Hidden:' at the start of each section.
    Do not use markdown formatting or numbered lists. Just use the exact format shown above.
    """
    
    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": CLAUDE_PROMPT_TEMPLATE.format(query=query)}
            ],
        )
        
        # Extract and parse the response
        raw_text = response.content[0].text
        print(f"Response received. Length: {len(raw_text)} characters")
        return parse_examples(raw_text, query)
        
    except Exception as e:
        print(f"Error generating with Claude: {e}")
        return []

def parse_examples(raw_text: str, original_query: str) -> List[Dict[str, str]]:
    """Parse the raw text response to extract examples"""
    examples = []
    
    # Save the full response to a file for inspection
    with open("last_response.txt", "w") as f:
        f.write(raw_text)
    
    # Simple, robust regex pattern to find the examples
    # This looks for the sequence Question:...Visible:...Hidden: in the text
    pattern = r"Question:(.*?)Visible:(.*?)Hidden:(.*?)(?=Question:|$)"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    
    print(f"Found {len(matches)} example(s)")
    
    # Process each match
    for match in matches:
        query = match[0].strip()
        visible = match[1].strip()
        hidden = match[2].strip()
        
        # Use the original query if empty
        if not query:
            query = original_query
            
        # Only add if visible and hidden are non-empty
        if visible and hidden:
            examples.append({
                "question": query,
                "visible": visible,
                "hidden": hidden
            })
    
    return examples

def save_to_json(examples: List[Dict[str, str]], filename: str):
    """Save examples to a JSON file"""
    # Convert to the format expected by our training script
    formatted_examples = []
    
    for example in examples:
        formatted_examples.append({
            "prompt": [
                {"role": "system", "content": ""},  # System prompt will be added later
                {"role": "user", "content": example["question"]}
            ],
            "visible": example["visible"],
            "hidden": example["hidden"]
        })
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(formatted_examples, f, indent=2)
    
    print(f"Dataset saved to {filename} with {len(formatted_examples)} examples")
    
    # Also create a Dataset object
    dataset = Dataset.from_dict({
        "prompt": [item["prompt"] for item in formatted_examples],
        "visible": [item["visible"] for item in formatted_examples],
        "hidden": [item["hidden"] for item in formatted_examples]
    })
    
    return dataset

def get_queries(args):
    """Get the list of queries to use"""
    queries = []
    
    # Use custom queries if provided
    if args.custom_queries and os.path.exists(args.custom_queries):
        with open(args.custom_queries, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    
    # Otherwise, use the built-in queries
    if not queries:
        queries = SAMPLE_QUERIES
    
    # Select the number of queries specified
    if len(queries) > args.num_queries:
        queries = random.sample(queries, args.num_queries)
    
    return queries

def main():
    """Main function"""
    args = parse_args()
    
    # Get the queries
    queries = get_queries(args)
    print(f"Generating examples for {len(queries)} queries...")
    
    # Generate examples for each query
    all_examples = []
    
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}: {query}")
        
        # Generate examples using the selected API
        if args.api == "openai":
            examples = generate_with_gpt4(args.api_key, query)
        else:  # anthropic
            examples = generate_with_claude(args.api_key, query)
        print(examples)
        
        # Add to the collection
        all_examples.extend(examples)
        
        # Save intermediate results (in case of failure)
        if (i + 1) % 10 == 0 or i == len(queries) - 1:
            temp_filename = f"temp_{i+1}_{args.output_file}"
            save_to_json(all_examples, temp_filename)
        
        # Sleep to avoid rate limits
        time.sleep(2)
    
    # Save the final dataset
    dataset = save_to_json(all_examples, args.output_file)
    print(f"Dataset generation complete! Generated {len(all_examples)} examples.")
    
    return dataset

if __name__ == "__main__":
    main()