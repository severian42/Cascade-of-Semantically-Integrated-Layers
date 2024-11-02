# Cascade of Semantically Integrated Layers (CSIL)

## Overview

The **Cascade of Semantically Integrated Layers (CSIL)** is a pure Python algorithm that enriches responses from any LLM by layering semantic understanding and contextual depth. CSIL breaks down input through a series of processing layersâ€”adding detail, context, and even a bit of wit along the way. No fancy frameworks, just Python being clever with a little help from `nltk`, `networkx`, and `scikit-learn`.

---


## How CSIL Works

CSIL processes input through four structured layers, each building on the previous to achieve a detailed, context-rich output. This layered approach helps the model not only understand questions but respond in a way that feels more human, intuitive, and informed.

### Layer Breakdown

1. **Initial Understanding**: Grabs the main concepts from user input using TF-IDF and NER (named entity recognition).  

2. **Relationship Analysis**: Discovers connections between concepts and structures them within a knowledge graph.

3. **Context Integration**: Adds broader contextual knowledge, such as historical or background information, that enriches the response.

4. **Response Synthesis**: Combines everything into a final answer that feels cohesive and contextually appropriate.

These steps help the LLM approach nuanced topics and complex questions with responses that feel layered and thought-through.

## Key Features

- **Pure Python Implementation**: No LangChain, LangGraph, or other frameworks, just the essentials for lightweight processing.

- **Knowledge Graph Integration**: Tracks relationships and concept frequency to improve understanding over time.

- **Dynamic Corpus Management**: Adapts its analysis based on user input, refining concepts with each interaction.

- **Adaptive Processing**: Configurable thresholds let you adjust response depth and specificity.

- **Optional Knowledge Base**: The knowledge base can be enabled or disabled depending on user needs, providing flexibility in context integration.

## Code Highlights

### Semantic Processing Pipeline

Each layer processes the input sequentially, from concept extraction to final synthesis:

```python
def process_semantic_cascade(self, user_input: str) -> Dict[str, Any]:
    """
    Multi-layer processing pipeline.
    
    Layers:
    1. Initial Understanding: Extracts main concepts.
    2. Relationship Analysis: Finds connections between concepts.
    3. Context Integration: Adds contextual knowledge.
    4. Response Synthesis: Assembles a cohesive response.
    
    Args:
        user_input: User query.
        
    Returns:
        Dict with results from each layer.
    """
    try:
        results = {
            'initial_understanding': '',
            'relationships': '',
            'context_integration': '',
            'final_response': '',
            'metadata': {'timestamp': datetime.now().isoformat()}
        }
        
        results['initial_understanding'] = self._process_layer("initial_understanding", user_input)
        results['relationships'] = self._process_layer("relationship_analysis", user_input, results['initial_understanding'])
        results['context_integration'] = self._process_layer("context_integration", user_input, results['relationships'])
        results['final_response'] = self._process_layer("synthesis", user_input, results['context_integration'])
        
        return results
    except Exception as e:
        return {
            'initial_understanding': '',
            'relationships': '',
            'context_integration': '',
            'final_response': f"Error processing query: {str(e)}",
            'error': str(e)
        }
```

### Knowledge Graph Management

CSIL uses a knowledge graph to capture concepts and connections over time, updating based on input:

```python
def _update_knowledge_graph(self, concepts: List[str], relationships: List[Tuple[str, str, float]]) -> None:
    """
    Updates the knowledge graph with new concepts and relationships.
    
    Args:
        concepts: List of new concepts to add.
        relationships: List of (concept1, concept2, weight) tuples.
    """
    for concept in concepts:
        if concept not in self.knowledge_graph:
            self.knowledge_graph.add_node(concept, first_seen=datetime.now(), frequency=1)
        else:
            self.knowledge_graph.nodes[concept]['frequency'] += 1
            
    for c1, c2, weight in relationships:
        if self.knowledge_graph.has_edge(c1, c2):
            current_weight = self.knowledge_graph[c1][c2]['weight']
            self.knowledge_graph[c1][c2]['weight'] = (current_weight + weight) / 2
        else:
            self.knowledge_graph.add_edge(c1, c2, weight=weight)
```

### Configuration and Adaptability

CSILâ€™s configuration allows for fine-tuning thresholds, processing options, adaptive response management, and an optional knowledge base toggle:

```python
@dataclass
class CSILConfig:
    """Configuration for CSIL processing."""
    min_keywords: int = 2
    max_keywords: int = 20
    similarity_threshold: float = 0.1
    debug_mode: bool = False
    use_knowledge_base: bool = True  # Toggle for knowledge base
```

## Installation and Usage

### Prerequisites

- **Python**: 3.10+
- **Dependencies**:
  - `nltk` >= 3.6
  - `scikit-learn` >= 0.24
  - `numpy` >= 1.19
  - `networkx` >= 2.5

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/severian42/Cascade-of-Semantically-Integrated-Layers.git
cd Cascade-of-Semantically-Integrated-Layers
pip install -r requirements.txt
```

### Running the CLI Application

Run the CSIL application with debug mode to see detailed processing steps:

```bash
python main.py --debug
```

- **`--debug`**: Provides detailed logs for each layer, helping you understand how CSIL processes input.


## License

CSIL is licensed under the MIT License. See the LICENSE file for more details.

---

With CSIL, every question becomes a layered exploration. Try it out, explore the layers, and see what a little extra semantic depth can do for your AI interactions!