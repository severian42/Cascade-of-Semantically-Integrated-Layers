from SemanticCascadeProcessing import (
    CascadeSemanticLayerProcessor, 
    CSILConfig, 
    LLMConfig, 
    KnowledgeBase,
    ensure_nltk_resources
)
import sys
from pathlib import Path
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple
import requests

def print_colored(text: str, color: str = 'blue', end: str = '\n') -> None:
    """Print colored text to console."""
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'red': '\033[91m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}", end=end)

def make_baseline_call(question: str, llm_config: LLMConfig) -> str:
    """Make a direct call to the LLM without CSIL processing."""
    try:
        headers = {"Content-Type": "application/json"}
        data = {
            "model": llm_config.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant. Answer the following question directly and concisely."},
                {"role": "user", "content": question}
            ],
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "stream": False  # Force non-streaming for baseline
        }
        
        response = requests.post(
            llm_config.url,
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print_colored(f"Baseline LLM call error: {str(e)}", 'red')
        return ""

def evaluate_response(response: str, correct_answer: str) -> Tuple[bool, float, str]:
    """
    Evaluate response accuracy and confidence.
    
    Args:
        response (str): The model's response
        correct_answer (str): The expected correct answer
        
    Returns:
        Tuple[bool, float, str]: (is_correct, confidence, status)
    """
    if not response:
        return False, 0.0, "NO_RESPONSE"
    
    is_correct = response.lower().strip() in correct_answer.lower()
    confidence = 1.0 if is_correct else 0.0
    status = "CORRECT" if is_correct else "INCORRECT"
    
    return is_correct, confidence, status

def load_benchmark_questions(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate benchmark questions from JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing benchmark questions
        
    Returns:
        List[Dict[str, Any]]: List of validated benchmark questions
        
    Raises:
        FileNotFoundError: If benchmark file doesn't exist
        JSONDecodeError: If benchmark file is not valid JSON
        ValueError: If benchmark questions are not properly formatted
    """
    try:
        # Check if file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")
        
        # Load questions
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # Validate question format
        required_fields = {'index', 'category', 'question', 'correct_answer', 'multiple_choice'}
        for i, q in enumerate(questions):
            missing_fields = required_fields - set(q.keys())
            if missing_fields:
                raise ValueError(
                    f"Question {i+1} missing required fields: {missing_fields}"
                )
            
            # Validate multiple choice format
            if not isinstance(q['multiple_choice'], list) or len(q['multiple_choice']) < 2:
                raise ValueError(
                    f"Question {i+1} has invalid multiple choice format"
                )
            
            # Ensure correct answer is in multiple choice options
            if q['correct_answer'] not in q['multiple_choice']:
                raise ValueError(
                    f"Question {i+1} correct answer not in multiple choice options"
                )
        
        print_colored(f"Loaded {len(questions)} benchmark questions", 'green')
        return questions
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in benchmark file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error loading benchmark questions: {str(e)}")

def print_result(model_name: str, response: str, correct_answer: str, time_taken: float):
    """Print detailed result for a model's response."""
    is_correct = response.lower().strip() in correct_answer.lower()
    status_symbol = '✓' if is_correct else '✗'
    status_color = 'green' if is_correct else 'red'
    
    print_colored(f"\n{model_name} Result:", 'blue')
    print_colored(f"  {status_symbol} Answer {'Correct' if is_correct else 'Incorrect'} ({time_taken:.2f}s)", status_color)
    print_colored(f"  Response: {response}", 'blue')
    print_colored(f"  Expected: {correct_answer}", 'blue')

def main():
    """Run CSIL evaluation."""
    print_colored("Initializing system...", 'blue')
    
    # Initialize LLM configuration
    llm_config = LLMConfig(
        url=os.getenv('LLM_URL', 'http://0.0.0.0:11434/v1/chat/completions'),
        model=os.getenv('LLM_MODEL', 'hf.co/arcee-ai/SuperNova-Medius-GGUF:f16'),
        context_window=int(os.getenv('LLM_CONTEXT_WINDOW', '8192')),
        max_tokens=int(os.getenv('LLM_MAX_TOKENS', '4096')),
        top_p=float(os.getenv('LLM_TOP_P', '0.9')),
        frequency_penalty=float(os.getenv('LLM_FREQUENCY_PENALTY', '0.0')),
        presence_penalty=float(os.getenv('LLM_PRESENCE_PENALTY', '0.0')),
        repeat_penalty=float(os.getenv('LLM_REPEAT_PENALTY', '1.1')),
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
        stream=True,
        stop_sequences=[],
        seed=int(os.getenv('LLM_SEED')) if os.getenv('LLM_SEED') else None
    )
    
    # Initialize CSIL
    config = CSILConfig(
        min_keywords=1,
        max_keywords=100,
        similarity_threshold=0.05,
        max_results=10,
        llm_config=llm_config,
        debug_mode='--debug' in sys.argv,
        use_external_knowledge=False
    )
    
    processor = CascadeSemanticLayerProcessor(config)
    
    # Load benchmark questions
    questions = load_benchmark_questions("linguistic_benchmark_multi_choice.json")
    
    # Track results for both CSIL and baseline
    csil_results = []
    baseline_results = []
    
    print_colored("\nStarting benchmark evaluation...", 'blue')
    
    for question in questions:
        try:
            print_colored(f"\nQ{question['index']}: {question['question']}", 'blue')
            
            # Test CSIL
            start_time = datetime.now()
            csil_response = processor.process_semantic_cascade(question['question'])
            csil_time = (datetime.now() - start_time).total_seconds()
            
            # Test Baseline
            start_time = datetime.now()
            baseline_response = make_baseline_call(question['question'], llm_config)
            baseline_time = (datetime.now() - start_time).total_seconds()
            
            # Print detailed results
            print_result("CSIL", csil_response['final_response'], 
                        question['correct_answer'], csil_time)
            print_result("Baseline", baseline_response, 
                        question['correct_answer'], baseline_time)
            
            # Store results
            csil_results.append({
                'index': question['index'],
                'category': question['category'],
                'correct': csil_response['final_response'].lower().strip() 
                          in question['correct_answer'].lower(),
                'confidence': 1.0,
                'time': csil_time,
                'response': csil_response['final_response'],
                'expected': question['correct_answer']
            })
            
            baseline_results.append({
                'index': question['index'],
                'category': question['category'],
                'correct': baseline_response.lower().strip() in question['correct_answer'].lower(),
                'confidence': 1.0,
                'time': baseline_time,
                'response': baseline_response,
                'expected': question['correct_answer']
            })
            
        except Exception as e:
            print_colored(
                f"Error processing question {question['index']}: {str(e)}", 
                'red'
            )
            continue
    
    # Generate comparative report
    report = generate_comparative_report(csil_results, baseline_results)
    
    # Save report
    report_path = Path(
        f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    report_path.write_text(report)
    print_colored(f"\nEvaluation report saved to {report_path}", 'green')

def generate_comparative_report(
    csil_results: List[Dict], 
    baseline_results: List[Dict]
) -> str:
    """Generate a comparative report between CSIL and baseline results."""
    csil_correct = sum(1 for r in csil_results if r['correct'])
    baseline_correct = sum(1 for r in baseline_results if r['correct'])
    total = len(csil_results)
    
    report = [
        "# Benchmark Evaluation Report",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Overall Results",
        f"Total Questions: {total}",
        "\n### CSIL Performance",
        f"Correct Answers: {csil_correct}",
        f"Accuracy: {(csil_correct/total*100):.1f}%",
        f"Average Time: {sum(r['time'] for r in csil_results)/total:.2f}s",
        "\n### Baseline Performance",
        f"Correct Answers: {baseline_correct}",
        f"Accuracy: {(baseline_correct/total*100):.1f}%",
        f"Average Time: {sum(r['time'] for r in baseline_results)/total:.2f}s",
    ]
    
    # Add category breakdown
    report.extend(["\n## Results by Category"])
    categories = set(r['category'] for r in csil_results)
    
    for category in categories:
        csil_cat = [r for r in csil_results if r['category'] == category]
        baseline_cat = [r for r in baseline_results if r['category'] == category]
        
        report.extend([
            f"\n### {category}",
            "#### CSIL",
            f"Questions: {len(csil_cat)}",
            f"Accuracy: {(sum(1 for r in csil_cat if r['correct'])/len(csil_cat)*100):.1f}%",
            f"Average Time: {sum(r['time'] for r in csil_cat)/len(csil_cat):.2f}s",
            "#### Baseline",
            f"Questions: {len(baseline_cat)}",
            f"Accuracy: {(sum(1 for r in baseline_cat if r['correct'])/len(baseline_cat)*100):.1f}%",
            f"Average Time: {sum(r['time'] for r in baseline_cat)/len(baseline_cat):.2f}s"
        ])
    
    return '\n'.join(report)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\nBenchmark interrupted!", 'red')
        sys.exit(0)
