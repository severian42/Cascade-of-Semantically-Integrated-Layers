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
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}", end=end)

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
    stats = processor.analyze_knowledge_graph()
    
    print_colored("\nKnowledge Graph Statistics:", 'blue')
    print_colored(f"Total Concepts: {stats['num_concepts']}", 'green')
    print_colored(f"Total Relationships: {stats['num_relationships']}", 'green')
    print_colored(f"Average Clustering: {stats['average_clustering']:.3f}", 'green')
    
    # Print top concepts by centrality
    print_colored("\nMost Central Concepts:", 'blue')
    central_concepts = sorted(
        stats['central_concepts'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    table_data = [[i+1, concept, f"{score:.3f}"] 
                 for i, (concept, score) in enumerate(central_concepts)]
    print(tabulate(
        table_data,
        headers=['Rank', 'Concept', 'Centrality Score'],
        tablefmt='simple'
    ))
    
    # Print communities
    print_colored(f"\nNumber of Communities: {len(stats['communities'])}", 'blue')
    for i, community in enumerate(stats['communities'][:3], 1):
        concepts = list(community)[:5]  # Show first 5 concepts per community
        print_colored(f"Community {i}: {', '.join(concepts)}", 'green')

def main():
    print_colored("Initializing system...", 'blue')
    
    if not initialize_system():
        print_colored("Failed to initialize system. Please check NLTK installation.", 'red')
        return
    
    try:
        # Initialize SemanticCascadeProcessor with configuration
        config = CSILConfig(
            min_keywords=1,
            max_keywords=100,
            similarity_threshold=0.05,
            max_results=10,
            llm_config=LLMConfig(),
            debug_mode='--debug' in sys.argv,
            
        )
        
        processor = CascadeSemanticLayerProcessor(config)
        processor.knowledge_base = initialize_knowledge_base(
            use_external_knowledge=config.use_external_knowledge
        )
        
        print_colored("\nWelcome to the Cascade of Semantically Integrated Layers\n", 'green')
        print_colored("Input → { [ ( * ) ] } → Output", 'purple')
        print_colored("Type 'quit' or 'exit' to end the conversation.", 'blue')
        print_colored("Type 'help' for available commands.\n", 'blue')
        
        # Add new commands to the help message
        COMMANDS = {
            'help': 'Show this help message',
            'quit/exit': 'End the conversation',
            'graph': 'Show current knowledge graph statistics',
            'concepts': 'List all concepts in the knowledge graph',
            'relations': 'Show strongest concept relationships',
            'analyze': 'Analyze last query\'s concept integration'
        }

        while True:
            try:
                print_colored("You: ", 'green', end='')
                user_input = input().strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print_colored("Goodbye!", 'green')
                    break
                    
                if user_input.lower() == 'help':
                    print_colored("\nAvailable commands:", 'blue')
                    for cmd, desc in COMMANDS.items():
                        print_colored(f"- {cmd}: {desc}", 'blue')
                    print()
                    continue
                    
                # Add new command handlers
                if user_input.lower() == 'graph':
                    print_graph_stats(processor)
                    continue
                    
                if user_input.lower() == 'concepts':
                    concepts = list(processor.knowledge_graph.nodes())
                    print_colored("\nCurrent Concepts:", 'blue')
                    for i, concept in enumerate(concepts, 1):
                        freq = processor.knowledge_graph.nodes[concept].get('frequency', 0)
                        print_colored(f"{i}. {concept} (freq: {freq})", 'green')
                    print()
                    continue
                    
                if user_input.lower() == 'relations':
                    edges = processor.knowledge_graph.edges(data=True)
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
                    continue
                    
                if user_input.lower() == 'analyze':
                    if processor.last_query:
                        concepts = processor._extract_key_concepts(processor.last_query)
                        subgraph = processor._extract_subgraph(concepts)
                        
                        print_colored("\nConcept Integration Analysis:", 'blue')
                        print_colored(f"Query Concepts: {', '.join(concepts)}", 'green')
                        print_colored(f"Related Concepts: {len(subgraph.nodes())}", 'green')
                        print_colored(f"Relationships: {len(subgraph.edges())}", 'green')
                        
                        # Show concept neighborhood
                        for concept in concepts:
                            if concept in subgraph:
                                neighbors = list(subgraph.neighbors(concept))
                                print_colored(f"\n{concept} connects to:", 'blue')
                                for neighbor in neighbors[:5]:
                                    weight = subgraph[concept][neighbor].get('weight', 0)
                                    print_colored(f"  - {neighbor} ({weight:.3f})", 'green')
                    else:
                        print_colored("No previous query to analyze.", 'red')
                    continue

                if not user_input:
                    continue
                    
                print_colored("\nProcessing...", 'blue')
                
                # Process the query using semantic cascade
                results = processor.process_semantic_cascade(user_input)
                
                print_colored("\nAssistant:", 'green')
                print_colored(results['final_response'], 'blue')
                
                if '--debug' in sys.argv:
                    print_colored("\nThought Process:", 'blue')
                    print_colored("1. Initial Understanding:", 'blue')
                    print_colored(results['initial_understanding'], 'green')
                    print_colored("\n2. Relationship Analysis:", 'blue')
                    print_colored(results['relationships'], 'green')
                    print_colored("\n3. Context Integration:", 'blue')
                    print_colored(results['context_integration'], 'green')
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print_colored("\nGoodbye!", 'green')
                break
            except Exception as e:
                print_colored(f"\nError processing query: {str(e)}", 'red')
                print_colored("Please try again.\n", 'red')
    
    except Exception as e:
        print_colored(f"Error initializing system: {str(e)}", 'red')
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\nGoodbye!", 'green')
        sys.exit(0)
