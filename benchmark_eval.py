"""
Semantic Cascade Processing (SCP) Evaluation Module

This module tests the effectiveness of SCP by comparing responses 
with and without SCP processing on a curated set of benchmark questions.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import re
import os
from dotenv import load_dotenv
from openai import OpenAI  # Or whatever LLM client you're using
from SemanticCascadeProcessing import SemanticCascadeProcessor, SCPConfig
import requests
import sseclient

@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    url: str
    model: str
    context_window: int
    temperature: float
    max_tokens: int
    stream: bool
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    repeat_penalty: float
    stop_sequences: List[str] = field(default_factory=list)
    seed: Optional[int] = None

class LLMClient:
    """Wrapper for LLM API calls."""
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM client with configuration."""
        self.config = config
    
    def complete(self, prompt: str) -> str:
        """Send prompt to LLM and get response."""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": self.config.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "stream": self.config.stream,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "repeat_penalty": self.config.repeat_penalty
            }
            
            # Add seed if specified
            if self.config.seed is not None:
                data["seed"] = self.config.seed

            response = requests.post(
                self.config.url,
                headers=headers,
                data=json.dumps(data),
                stream=True
            )
            response.raise_for_status()

            if self.config.stream:
                # Handle streaming response
                client = sseclient.SSEClient(response)
                full_response = ""
                for event in client.events():
                    try:
                        chunk = json.loads(event.data)
                        if 'choices' in chunk and chunk['choices']:
                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                            full_response += content
                    except json.JSONDecodeError:
                        continue
                return full_response
            else:
                # Handle non-streaming response
                return response.json()['choices'][0]['message']['content']

        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            return ""

def initialize_llm_client() -> LLMClient:
    """Initialize LLM client with configuration from environment."""
    load_dotenv()  # Load environment variables from .env file
    
    config = LLMConfig(
        url=os.getenv('LLM_URL', 'http://0.0.0.0:11434/v1/chat/completions'),
        model=os.getenv('LLM_MODEL', 'hf.co/arcee-ai/SuperNova-Medius-GGUF:f16'),
        context_window=int(os.getenv('LLM_CONTEXT_WINDOW', '8192')),
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.6')),
        max_tokens=int(os.getenv('LLM_MAX_TOKENS', '4096')),
        stream=os.getenv('LLM_STREAM', 'true').lower() == 'true',
        top_p=float(os.getenv('LLM_TOP_P', '0.9')),
        frequency_penalty=float(os.getenv('LLM_FREQUENCY_PENALTY', '0.0')),
        presence_penalty=float(os.getenv('LLM_PRESENCE_PENALTY', '0.0')),
        repeat_penalty=float(os.getenv('LLM_REPEAT_PENALTY', '1.1')),
        seed=int(os.getenv('LLM_SEED')) if os.getenv('LLM_SEED') else None
    )
    
    return LLMClient(config)

@dataclass
class EvalQuestion:
    """Represents a single evaluation question."""
    index: int
    category: str
    question: str
    correct_answer: str
    multiple_choice: List[str]

@dataclass
class EvalResult:
    """Stores results for a single evaluation run."""
    question_index: int
    category: str
    correct_answer: str
    scp_answer: str
    baseline_answer: str
    scp_time: float
    baseline_time: float
    scp_correct: bool
    baseline_correct: bool

class SCPEvaluator:
    """Evaluates SCP performance against baseline."""
    
    def __init__(self, benchmark_path: str):
        """Initialize evaluator with benchmark questions."""
        # Test all questions in the benchmark file
        self.questions = self._load_questions(benchmark_path)
        self.results: List[EvalResult] = []
        self.debug_mode = False
    
    def _load_questions(self, path: str) -> List[EvalQuestion]:
        """Load all benchmark questions."""
        with open(path, 'r') as f:
            data = json.load(f)
            
        return [
            EvalQuestion(
                index=q['index'],
                category=q['category'],
                question=q['question'],
                correct_answer=q['correct_answer'],
                multiple_choice=q['multiple_choice']
            )
            for q in data
        ]
    
    def run_evaluation(self, 
                      scp_processor: SemanticCascadeProcessor,
                      llm_client: LLMClient,
                      baseline_prompt: str = "") -> None:
        """Run evaluation comparing SCP vs baseline."""
        for question in self.questions:
            # Format question with choices
            formatted_q = (
                f"Question: {question.question}\n\n"
                f"Choose from:\n" + 
                "\n".join(f"{i+1}. {c}" for i, c in enumerate(question.multiple_choice))
            )
            
            # Run SCP evaluation
            scp_start = datetime.now()
            scp_result = scp_processor.process_interaction(formatted_q)
            scp_time = (datetime.now() - scp_start).total_seconds()
            scp_answer = scp_result['final_response']
            
            # Run baseline evaluation
            baseline_start = datetime.now()
            baseline_prompt_text = (
                f"{baseline_prompt}\n\n{formatted_q}"
                if baseline_prompt else formatted_q
            )
            # Make actual LLM call for baseline
            baseline_answer = llm_client.complete(baseline_prompt_text)
            baseline_time = (datetime.now() - baseline_start).total_seconds()
            
            # Store results
            self.results.append(EvalResult(
                question_index=question.index,
                category=question.category,
                correct_answer=question.correct_answer,
                scp_answer=scp_answer,
                baseline_answer=baseline_answer,
                scp_time=scp_time,
                baseline_time=baseline_time,
                scp_correct=self._check_answer(scp_answer, question.correct_answer),
                baseline_correct=self._check_answer(baseline_answer, question.correct_answer)
            ))
    
    def _check_answer(self, response: str, correct: str) -> bool:
        """
        Check if response matches the correct answer with improved accuracy.
        Handles multiple formats and partial matches.
        """
        if not response or not correct:
            return False
            
        # Normalize both strings
        response = response.lower().strip()
        correct = correct.lower().strip()
        
        # Direct match
        if correct in response:
            return True
            
        # Handle numeric answers
        if correct.replace(' ', '').isdigit():
            # Extract numbers from response
            numbers = re.findall(r'\d+', response)
            return any(num == correct.replace(' ', '') for num in numbers)
            
        # Handle multiple choice format (e.g., "1." or "A.")
        if len(correct) <= 3 and correct.endswith('.'):
            choices = re.findall(r'[A-D1-4]\.', response)
            return any(choice.lower() == correct.lower() for choice in choices)
        
        if self.debug_mode:
            print(f"\nAnswer Checking:")
            print(f"Response: {response[:100]}...")
            print(f"Correct: {correct}")
            print(f"Match found: {correct in response}")
        
        return False
    
    def generate_report(self) -> str:
        """Generate evaluation report with detailed metrics."""
        if not self.results:
            return "No evaluation results available."
            
        total = len(self.results)
        scp_correct = sum(r.scp_correct for r in self.results)
        baseline_correct = sum(r.baseline_correct for r in self.results)
        
        # Calculate accuracy percentages
        scp_accuracy = (scp_correct/total*100) if total > 0 else 0
        baseline_accuracy = (baseline_correct/total*100) if total > 0 else 0
        
        # Calculate average processing times
        avg_scp_time = sum(r.scp_time for r in self.results) / total
        avg_baseline_time = sum(r.baseline_time for r in self.results) / total
        
        # Calculate performance by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {
                    'total': 0, 
                    'scp_correct': 0, 
                    'baseline_correct': 0,
                    'scp_time': 0,
                    'baseline_time': 0
                }
            categories[r.category]['total'] += 1
            categories[r.category]['scp_time'] += r.scp_time
            categories[r.category]['baseline_time'] += r.baseline_time
            if r.scp_correct:
                categories[r.category]['scp_correct'] += 1
            if r.baseline_correct:
                categories[r.category]['baseline_correct'] += 1
        
        report = [
            "# SCP Evaluation Report",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal Questions: {total}",
            "\n## Overall Results",
            f"SCP Correct: {scp_correct}/{total} ({scp_accuracy:.1f}%)",
            f"Baseline Correct: {baseline_correct}/{total} ({baseline_accuracy:.1f}%)",
            f"\nImprovement: {scp_accuracy - baseline_accuracy:+.1f}%",
            f"\nAverage Processing Times:",
            f"SCP: {avg_scp_time:.2f}s",
            f"Baseline: {avg_baseline_time:.2f}s",
            f"Time Difference: {avg_scp_time - avg_baseline_time:+.2f}s",
            "\n## Results by Category"
        ]
        
        for category, stats in categories.items():
            cat_total = stats['total']
            scp_cat_acc = (stats['scp_correct']/cat_total*100) if cat_total > 0 else 0
            base_cat_acc = (stats['baseline_correct']/cat_total*100) if cat_total > 0 else 0
            avg_cat_scp_time = stats['scp_time'] / cat_total
            avg_cat_base_time = stats['baseline_time'] / cat_total
            
            report.extend([
                f"\n### {category}",
                f"Questions: {cat_total}",
                f"SCP: {stats['scp_correct']}/{cat_total} ({scp_cat_acc:.1f}%)",
                f"Baseline: {stats['baseline_correct']}/{cat_total} ({base_cat_acc:.1f}%)",
                f"Improvement: {scp_cat_acc - base_cat_acc:+.1f}%",
                f"Avg Time - SCP: {avg_cat_scp_time:.2f}s",
                f"Avg Time - Baseline: {avg_cat_base_time:.2f}s"
            ])
        
        return "\n".join(report)

def main():
    """Run SCP evaluation."""
    # Initialize LLM client
    llm_client = initialize_llm_client()
    
    # Initialize SCP with debug mode and LLM config
    config = SCPConfig(
        debug_mode=True,
        llm_config=llm_client.config  # Pass the LLM config here
    )
    scp = SemanticCascadeProcessor(config)  # Only pass config
    
    # Initialize evaluator
    evaluator = SCPEvaluator("linguistic_benchmark_multi_choice.json")
    
    # Run evaluation with both SCP and baseline
    evaluator.run_evaluation(
        scp_processor=scp,
        llm_client=llm_client,
        baseline_prompt="You are a helpful AI assistant. Please answer the following question:"
    )
    
    # Generate and save report
    report = evaluator.generate_report()
    report_path = Path(f"scp_eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    report_path.write_text(report)
    print(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    main()
