from SemanticCascadeProcessing import (
    CascadeSemanticLayerProcessor, 
    CSILConfig, 
    LLMConfig, 
    KnowledgeBase,
    ensure_nltk_resources
)
import sys
from typing import List, Dict, Any
from pathlib import Path
import os
import json
import nltk
import networkx as nx
from tabulate import tabulate

# Define available commands
COMMANDS = {
    'help': 'Show available commands',
    'graph': 'Display knowledge graph statistics and relationships',
    'concepts': 'List all concepts in the knowledge graph',
    'relations': 'Show strongest concept relationships',
    'quit': 'Exit the program',
    'exit': 'Exit the program'
}

def print_colored(text: str, color: str = 'blue', end: str = '\n') -> None:
    """
    Print colored text to console.
    
    Args:
        text (str): Text to print
        color (str): Color to use ('blue', 'green', 'red')
        end (str): String to append at the end (default: newline)
    """
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m',
        'red': '\033[91m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'yellow': '\033[93m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}", end=end)

def get_avatar(style: str = 'default') -> str:
    """Get the system avatar in different styles."""
    avatars = {
        'default': '{ [ ( * ) ] }',
        'thinking': '{ [ (…) ] }',
        'processing': '{ [ (⟳) ] }',
        'success': '{ [ (✓) ] }',
        'error': '{ [ (!) ] }',
        'waiting': '{ [ (?) ] }'
    }
    return avatars.get(style, avatars['default'])

# Define base knowledge directory
KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"

def validate_knowledge_base_structure() -> bool:
    """Validate the knowledge base directory structure exists"""
    required_dirs = ['prompts', 'concepts', 'examples', 'context']
    required_files = [
        'prompts/system_prompts.json',
        'prompts/conversation_templates.json'
    ]
    
    # Create directories if they don't exist
    for dir_name in required_dirs:
        dir_path = KNOWLEDGE_BASE_DIR / dir_name
        if not dir_path.is_dir():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print_colored(f"Created directory: {dir_name}", 'green')
            except Exception as e:
                print_colored(f"Error creating directory '{dir_name}': {str(e)}", 'red')
                return False
    
    # Check required files
    for file_name in required_files:
        file_path = KNOWLEDGE_BASE_DIR / file_name
        if not file_path.is_file():
            print_colored(f"Error: Required file '{file_name}' not found", 'red')
            return False
            
    return True

def initialize_knowledge_base(use_external_knowledge: bool = False) -> KnowledgeBase:
    """Initialize and load knowledge base"""
    kb = KnowledgeBase()
    
    # Validate directory structure for required files
    required_dirs = ['prompts']
    required_files = [
        'prompts/system_prompts.json',
        'prompts/conversation_templates.json'
    ]
    
    # Add optional directories if external knowledge is enabled
    if use_external_knowledge:
        required_dirs.extend(['concepts', 'examples', 'context'])
    
    # Create and validate directories
    for dir_name in required_dirs:
        dir_path = KNOWLEDGE_BASE_DIR / dir_name
        if not dir_path.is_dir():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print_colored(f"Created directory: {dir_name}", 'green')
            except Exception as e:
                if kb.is_required(dir_name):
                    raise RuntimeError(f"Failed to create required directory: {dir_name}")
                print_colored(f"Warning: Optional directory not created: {dir_name}", 'red')
    
    try:
        # Load required prompts
        prompt_dir = KNOWLEDGE_BASE_DIR / "prompts"
        kb.load_from_json(str(prompt_dir / "system_prompts.json"))
        kb.load_from_json(str(prompt_dir / "conversation_templates.json"))
        
        # Load optional knowledge if enabled
        if use_external_knowledge:
            for dir_name in ['concepts', 'examples', 'context']:
                dir_path = KNOWLEDGE_BASE_DIR / dir_name
                try:
                    kb.load_from_directory(str(dir_path))
                    print_colored(f"Loaded documents from {dir_name}", 'green')
                except Exception as e:
                    print_colored(f"Warning: Could not load {dir_name}: {str(e)}", 'red')
    
    except Exception as e:
        raise RuntimeError(f"Failed to initialize knowledge base: {str(e)}")
    
    return kb

def initialize_system():
    """Initialize NLTK and verify resources"""
    try:
        ensure_nltk_resources()
        # Verify NLTK resources are loaded
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        return True
    except LookupError as e:
        print_colored(f"Error: Failed to initialize NLTK resources: {e}", 'red')
        return False

def print_graph_stats(processor: CascadeSemanticLayerProcessor) -> None:
    """Print current knowledge graph statistics."""
    try:
        processor.knowledge.print_graph_summary()
    except Exception as e:
        print_colored(f"Error analyzing graph: {str(e)}", 'red')

def print_welcome_message():
    """Print welcome message with ASCII art and helpful information."""
    art = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║               Cascade Semantic Integration Layer             ║
    ║                     {get_avatar()}                           ║
    ║                                                              ║
    ║     Input ──╮                                    ╭── Output  ║
    ║             │   ╭─{{ Semantic Analysis }}─╮      │           ║
    ║             ├──>│  [ Context Mapping ]    │─────>│           ║
    ║             │   │   ( Integration )       │      │           ║
    ║             │   │      * Fusion *         │      │           ║
    ║             │   ╰─────────────────────────╯      │           ║
    ║             ╰────────────────────────────────────╯           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    tips = f"""
    {get_avatar('waiting')} Quick Start Guide:
    ├─ Type any question or statement to begin
    ├─ Use 'help' to see all available commands
    ├─ Use 'graph' to visualize the knowledge network
    ├─ Use 'concepts' to see extracted concepts
    └─ Use 'relations' to explore concept relationships
    
    {get_avatar('thinking')} The system will:
    ├─ Extract key concepts from your input
    ├─ Analyze semantic relationships
    ├─ Build a dynamic knowledge graph
    └─ Generate contextually aware responses
    """
    
    print_colored(art, 'cyan')
    print_colored(f"\n{get_avatar()} Welcome to CaSIL!\n", 'green')
    print_colored(tips, 'blue')
    print_colored(f"\n{get_avatar('waiting')} Ready for your input...\n", 'green')

def main():
    # Add welcome message at the start
    print_welcome_message()
    
    if not initialize_system():
        print_colored("Failed to initialize system. Please check NLTK installation.", 'red')
        return
    
    try:
        # Initialize configuration and processor
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
            seed=None
        )
        
        config = CSILConfig(
            min_keywords=1,
            max_keywords=100,
            similarity_threshold=0.05,
            max_results=10,
            llm_config=llm_config,
            debug_mode='--debug' in sys.argv,
            use_external_knowledge=False
        )
        
        # Initialize processor with configuration
        processor = CascadeSemanticLayerProcessor(config)
        
        # Initialize knowledge base
        processor.knowledge_base = initialize_knowledge_base(
            use_external_knowledge=config.use_external_knowledge
        )

        # Command handlers
        def handle_concepts_command():
            concepts = list(processor.knowledge.knowledge_graph.nodes())
            if not concepts:
                print_colored("\nNo concepts in knowledge graph yet.", 'blue')
                return
                
            print_colored("\nCurrent Concepts:", 'blue')
            for i, concept in enumerate(concepts, 1):
                freq = processor.knowledge.knowledge_graph.nodes[concept].get('frequency', 0)
                print_colored(f"{i}. {concept} (freq: {freq})", 'green')
            print()

        def handle_relations_command():
            edges = list(processor.knowledge.knowledge_graph.edges(data=True))
            if not edges:
                print_colored("\nNo relationships in knowledge graph yet.", 'blue')
                return
                
            sorted_edges = sorted(
                edges, 
                key=lambda x: x[2].get('weight', 0), 
                reverse=True
            )[:10]
            
            print_colored("\nStrongest Concept Relationships:", 'blue')
            table_data = [
                [i+1, f"{c1} → {c2}", f"{data.get('weight', 0):.3f}"]
                for i, (c1, c2, data) in enumerate(sorted_edges)
            ]
            print(tabulate(
                table_data,
                headers=['Rank', 'Relationship', 'Weight'],
                tablefmt='simple'
            ))
            print()

        # Main interaction loop
        while True:
            try:
                print_colored(f"{get_avatar('waiting')} You: ", 'green', end='')
                user_input = input().strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print_colored(f"\n{get_avatar('success')} Goodbye!", 'green')
                    break
                    
                if user_input.lower() == 'help':
                    print_colored(f"\n{get_avatar('thinking')} Available commands:", 'blue')
                    for cmd, desc in COMMANDS.items():
                        print_colored(f"- {cmd}: {desc}", 'blue')
                    print()
                    continue
                    
                if user_input.lower() == 'graph':
                    print_graph_stats(processor)
                    continue
                    
                if user_input.lower() == 'concepts':
                    handle_concepts_command()
                    continue
                    
                if user_input.lower() == 'relations':
                    handle_relations_command()
                    continue

                # Process query and handle response
                if user_input:
                    print_colored(f"\n{get_avatar('processing')} Processing...", 'blue')
                    results = processor.process_semantic_cascade(user_input)
                    
                    if isinstance(results, dict) and 'final_response' in results:
                        print_colored(f"\n{get_avatar()} Assistant:", 'green')
                        print_colored(results['final_response'], 'blue')
                    else:
                        print_colored(f"\n{get_avatar('error')} Error: Invalid response format", 'red')
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print_colored(f"\n{get_avatar('success')} Goodbye!", 'green')
                break
            except Exception as e:
                print_colored(f"\n{get_avatar('error')} Error: {str(e)}", 'red')
                if config.debug_mode:
                    import traceback
                    print_colored(traceback.format_exc(), 'red')
                    
    except Exception as e:
        print_colored(f"Error initializing system: {str(e)}", 'red')
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\nGoodbye!", 'green')
        sys.exit(0)
