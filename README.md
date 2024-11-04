# Cascade of Semantically Integrated Layers (CaSIL)

<div align="center">
  <h1><b>{ [ ( * ) ] }</b></h1>

  [CaSIL_Marble_In_Cup_Question](https://github.com/user-attachments/assets/3c3d8ec5-8630-40a5-a8a4-75cbc1e43720)
</div>


---

## 🌊 Overview

CaSIL is an advanced natural language processing system that implements a sophisticated four-layer semantic analysis architecture. It processes both user input and knowledge base content through progressive semantic layers, combining:

- Dynamic concept extraction and relationship mapping
- Adaptive semantic similarity analysis
- Context-aware knowledge graph integration
- Multi-layer processing pipeline
- Real-time learning and adaptation

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Local LLM server (e.g., Ollama)
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/severian42/Cascade-of-Semantically-Integrated-Layers.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start your local LLM server (e.g., Ollama)

5. Run the system:
```bash
python main.py --debug # (Shows full internal insight and metrics)
```

### Basic Usage

```bash
# Start the system
python main.py --debug

# Available commands:
help      # Show available commands
graph     # Display knowledge graph statistics
concepts  # List extracted concepts
relations # Show concept relationships
exit      # Exit the program
```

## 🧠 Core Architecture

### 1. Semantic Processing Layers

CaSIL processes input through four sophisticated layers:

#### Layer 1: Initial Understanding (θ=0.7)
- Advanced concept extraction using TF-IDF
- Named entity recognition
- Custom stopword filtering
- Semantic weight calculation

#### Layer 2: Relationship Analysis (θ=0.7)
- Dynamic similarity matrix computation
- Graph-based relationship discovery
- Concept clustering and community detection
- Temporal relationship tracking

#### Layer 3: Context Integration (θ=0.9)
- Historical context weighting
- Knowledge base integration
- Dynamic context windows
- Adaptive threshold adjustment

#### Layer 4: Response Synthesis (θ=0.8)
- Multi-source information fusion
- Style-specific processing
- Dynamic temperature adjustment
- Context-aware response generation

### 2. Knowledge Graph System

CaSIL maintains two interconnected graph systems:

#### Session Graph
- Tracks temporary concept relationships
- Maintains conversation context
- Updates in real-time
- Handles recency weighting

#### Persistent Knowledge Graph
- Stores long-term concept relationships
- Tracks concept evolution over time
- Maintains relationship weights
- Supports community detection

## 🛠️ Technical Features

### Processing Pipeline Architecture
- **Multi-Stage Processing**
  - Input text → Concept Extraction → Relationship Analysis → Context Integration → Response
  - Each stage maintains its own similarity thresholds and processing parameters
  - Adaptive feedback loop adjusts parameters based on processing results

- **Semantic Analysis Engine**
  ```python
  # Example concept extraction flow
  text → TF-IDF Vectorization → Weight Calculation → Threshold Filtering → Concepts
  
  # Relationship discovery process
  concepts → Similarity Matrix → Graph Construction → Community Detection → Relationships
  ```

- **Dynamic Temperature Control**
  ```python
  temperature = base_temp * layer_modifier['base'] * (
      1 + (novelty_score * layer_modifier['novelty_weight']) *
      (1 + (complexity_factor * layer_modifier['complexity_weight']))
  )
  ```

### Knowledge Graph Architecture
- **Dual-Graph System**
  ```
  Session Graph (Temporary)      Knowledge Graph (Persistent)
  ├─ Short-term relationships    ├─ Long-term concept storage
  ├─ Recency weighting           ├─ Relationship evolution
  ├─ Context tracking            ├─ Community detection
  └─ Real-time updates           └─ Concept metadata
  ```

- **Graph Update Process**
  ```python
  # Simplified relationship update flow
  new_weight = (previous_weight + (similarity * time_weight)) / 2
  graph.add_edge(concept1, concept2, weight=new_weight)
  ```

### Unique Features

#### 1. Adaptive Processing
- Dynamic threshold adjustment based on:
  ```
  ├─ Input complexity
  ├─ Concept novelty
  ├─ Processing layer
  └─ Historical performance
  ```

#### 2. Semantic Integration
- Multi-dimensional similarity calculation:
  ```
  Combined Similarity = (0.7 * cosine_similarity) + (0.3 * jaccard_similarity)
  ```
- Weighted by:
  - Term frequency
  - Position importance
  - Historical context

#### 3. Context Management
```
Context Integration Flow:
Input → Session Context → Knowledge Graph → External Knowledge → Response
     ↑______________________________________________|
     (Feedback Loop)
```

### Data Flow Architecture

```
                                    ┌────────────────┐
                                    │ Knowledge Base │
                                    └───────┬────────┘
                                            │
                                            ▼
User Input → Concept Extraction → Similarity Analysis → Graph Integration
    ↑              │                        │                  │
    │              ▼                        ▼                  ▼
    │         Session Graph ──────► Relationship Analysis ◄─── Knowledge Graph
    │              │                        │                  │
    │              └────────────────────────┼──────────────────
    │                                       │
    └───────────────────── Response ◄───────┘
```

### Performance Optimizations
- LRU Cache for similarity calculations
- Concurrent processing with thread pooling
- Batched vectorizer updates
- Adaptive corpus management

### Key Differentiators
1. **Progressive Semantic Analysis**
   - Each layer builds upon previous insights
   - Maintains context continuity
   - Adapts processing parameters in real-time

2. **Dynamic Knowledge Integration**
   - Combines session-specific and persistent knowledge
   - Real-time graph updates
   - Community-based concept clustering

3. **Adaptive Response Generation**
   - Context-aware temperature adjustment
   - Style-specific processing parameters
   - Multi-source information fusion

## 📊 System Requirements

- **CPU**: Multi-core processor recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 1GB for base system
- **Python**: 3.8 or higher
- **OS**: Linux, macOS, or Windows

## 🔧 Configuration

### Environment Variables
```env
DEBUG_MODE=true
USE_EXTERNAL_KNOWLEDGE=false
LLM_URL=http://0.0.0.0:11434/v1/chat/completions
LLM_MODEL=your_model_name
```

### Processing Thresholds
```env
INITIAL_UNDERSTANDING_THRESHOLD=0.7
RELATIONSHIP_ANALYSIS_THRESHOLD=0.7
CONTEXTUAL_INTEGRATION_THRESHOLD=0.9
SYNTHESIS_THRESHOLD=0.8
```

## 📚 Advanced Usage

### Custom Knowledge Integration
```python
from SemanticCascadeProcessing import CascadeSemanticLayerProcessor

processor = CascadeSemanticLayerProcessor(config)
processor.knowledge_base.load_from_directory("path/to/knowledge")
```

### Graph Analysis
```python
# Get graph statistics
processor.knowledge.print_graph_summary()

# Analyze concept relationships
processor.analyze_knowledge_graph()
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the open-source NLP community
- Special thanks to LocalLLaMA for never letting anything slip by them unnoticed; and for just being an awesome community overall
