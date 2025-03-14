"""
Evaluation module for Secret Conversations model.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from vllm import SamplingParams

from utils.extraction import extract_visible, extract_hidden, check_format, fix_response_format

class SecretConversationsEvaluator:
    """Evaluator for Secret Conversations model."""
    
    def __init__(self, model, tokenizer, system_prompt):
        """
        Initialize the evaluator.
        
        Args:
            model: The trained model
            tokenizer: The tokenizer
            system_prompt: The system prompt to use for generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        
    def generate_response(self, 
                          query: str, 
                          temperature: float = 0.8, 
                          top_p: float = 0.95, 
                          max_tokens: int = 512,
                          lora_path: Optional[str] = None) -> str:
        """
        Generate a response from the model.
        
        Args:
            query: The user query
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            lora_path: Path to LoRA weights (if using)
            
        Returns:
            The generated response
        """
        # Format input with chat template
        formatted_input = self.tokenizer.apply_chat_template([
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ], tokenize=False, add_generation_prompt=True)
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        # Generate response
        if lora_path:
            lora_request = self.model.load_lora(lora_path)
            output = self.model.fast_generate(
                formatted_input,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )[0].outputs[0].text
        else:
            output = self.model.fast_generate(
                [formatted_input],
                sampling_params=sampling_params,
                lora_request=None,
            )[0].outputs[0].text
        
        # Check if the format is correct and fix if needed
        if "<visible>" in output and "<hidden>" in output:
            if not output.strip().endswith("</hidden>"):
                print(f"Warning: Fixing incomplete tags in response for query: '{query[:30]}...'")
                output = fix_response_format(output)
            
        return output
    
    def evaluate_responses(self, queries: List[str], lora_path: Optional[str] = None) -> pd.DataFrame:
        """
        Evaluate the model on a list of queries.
        
        Args:
            queries: List of user queries
            lora_path: Path to LoRA weights (if using)
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for query in queries:
            response = self.generate_response(query, lora_path=lora_path)
            visible = extract_visible(response)
            hidden = extract_hidden(response)
            
            # Calculate metrics
            has_correct_format = check_format(response)
            visible_length = len(visible)
            hidden_length = len(hidden)
            
            # Calculate distinctness
            if visible and hidden:
                visible_words = set(visible.lower().split())
                hidden_words = set(hidden.lower().split())
                if visible_words and hidden_words:
                    word_overlap = len(visible_words.intersection(hidden_words))
                    overlap_ratio = word_overlap / min(len(visible_words), len(hidden_words))
                else:
                    overlap_ratio = 0.0
            else:
                overlap_ratio = 0.0
                
            results.append({
                'query': query,
                'response': response,
                'visible': visible,
                'hidden': hidden,
                'has_correct_format': has_correct_format,
                'visible_length': visible_length,
                'hidden_length': hidden_length,
                'overlap_ratio': overlap_ratio,
            })
            
        return pd.DataFrame(results)
    
    def print_evaluation_summary(self, df: pd.DataFrame) -> None:
        """
        Print a summary of the evaluation results.
        
        Args:
            df: DataFrame with evaluation results
        """
        print("=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        
        # Format metrics
        format_success_rate = df['has_correct_format'].mean() * 100
        avg_visible_length = df['visible_length'].mean()
        avg_hidden_length = df['hidden_length'].mean()
        avg_overlap = df['overlap_ratio'].mean() * 100
        
        print(f"Correct Format Rate: {format_success_rate:.1f}%")
        print(f"Average Visible Length: {avg_visible_length:.1f} chars")
        print(f"Average Hidden Length: {avg_hidden_length:.1f} chars")
        print(f"Average Word Overlap: {avg_overlap:.1f}%")
        
        # Print sample
        print("\nSAMPLE RESPONSES:")
        for i, row in df.sample(min(3, len(df))).iterrows():
            print("\n" + "-" * 40)
            print(f"Query: {row['query']}")
            print(f"\nVisible: {row['visible']}")
            print(f"\nHidden: {row['hidden']}")
            
        print("=" * 50)