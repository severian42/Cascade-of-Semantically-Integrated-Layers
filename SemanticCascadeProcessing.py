"""
Cascade of Semantically Integrated Layers (CSIL) Module

This module implements the Cascade of Semantically Integrated Layers algorithm, 
which combines computational semantic analysis with progressive multi-layer 
processing for enhanced natural language understanding and response generation.

Key Components:
    - Semantic Analysis: TF-IDF vectorization and cosine similarity
    - Dynamic Temperature Adjustment: Based on semantic similarity
    - Multi-layer Processing: Progressive understanding through structured layers
    - Knowledge Base Integration: External knowledge incorporation
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import requests
import json
import sseclient
import os
from datetime import datetime
from nltk_utils import ensure_nltk_resources
from dotenv import load_dotenv
from pathlib import Path
import string
from functools import lru_cache
import concurrent.futures
import dataclasses
from networkx import Graph, pagerank
from sklearn.cluster import DBSCAN
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """
    Configuration settings for Language Model API interactions.
    
    Attributes:
        url (str): API endpoint URL for LLM service
        model (str): Name/path of the language model to use
        context_window (int): Maximum context window size
        temperature (float): Base temperature for response generation
        max_tokens (int): Maximum tokens in generated response
        stream (bool): Whether to use streaming response
        top_p (float): Nucleus sampling parameter
        frequency_penalty (float): Penalty for frequent token use
        presence_penalty (float): Penalty for repeated content
        stop_sequences (List[str]): Sequences to stop generation
        repeat_penalty (float): Additional penalty for repetition
        seed (Optional[int]): Random seed for reproducibility
    """
    url: str = field(default_factory=lambda: os.getenv('LLM_URL', 
        "http://0.0.0.0:11434/v1/chat/completions"))
    model: str = field(default_factory=lambda: os.getenv('LLM_MODEL', 
        "hf.co/arcee-ai/SuperNova-Medius-GGUF:f16"))
    context_window: int = field(
        default_factory=lambda: int(os.getenv('LLM_CONTEXT_WINDOW', 8192)))
    temperature: float = field(
        default_factory=lambda: float(os.getenv('LLM_TEMPERATURE', 0.6)))
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv('LLM_MAX_TOKENS', 4096)))
    stream: bool = field(default_factory=lambda: 
        os.getenv('LLM_STREAM', 'true').lower() == 'true')
    top_p: float = field(
        default_factory=lambda: float(os.getenv('LLM_TOP_P', 0.9)))
    frequency_penalty: float = field(
        default_factory=lambda: float(os.getenv('LLM_FREQUENCY_PENALTY', 0.0)))
    presence_penalty: float = field(
        default_factory=lambda: float(os.getenv('LLM_PRESENCE_PENALTY', 0.0)))
    stop_sequences: List[str] = field(default_factory=list)
    repeat_penalty: float = field(
        default_factory=lambda: float(os.getenv('LLM_REPEAT_PENALTY', 1.1)))
    seed: Optional[int] = field(default_factory=lambda: 
        int(os.getenv('LLM_SEED')) if os.getenv('LLM_SEED') else None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format."""
        return {
            'url': self.url,
            'model': self.model,
            'context_window': self.context_window,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'stream': self.stream,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stop_sequences': self.stop_sequences,
            'repeat_penalty': self.repeat_penalty,
            'seed': self.seed
        }

    def get_temperature(self) -> float:
        """Get current temperature setting."""
        return self.temperature

    def set_temperature(self, temp: float) -> None:
        """Set temperature value."""
        self.temperature = max(0.0, min(2.0, float(temp)))

@dataclass
class CSILConfig:
    """Configuration for Cascade of Semantically Integrated Layers.
    
    Args:
        min_keywords: Minimum number of keywords to extract
        max_keywords: Maximum number of keywords to extract
        keyword_weight_threshold: Minimum weight threshold for keywords
        similarity_threshold: Minimum similarity threshold
        max_results: Maximum number of results to return
        llm_config: LLM configuration settings
        debug_mode: Enable debug output
        use_external_knowledge: Enable external knowledge integration
        layer_thresholds: Layer-specific similarity thresholds
        adaptive_thresholds: Enable adaptive thresholds
        min_threshold: Minimum threshold for adaptive thresholds
        max_threshold: Maximum threshold for adaptive thresholds
        threshold_step: Step size for adaptive thresholds
    """
    min_keywords: int = 2
    max_keywords: int = 20
    keyword_weight_threshold: float = 0.1
    similarity_threshold: float = 0.1
    max_results: Optional[int] = None
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    debug_mode: bool = False
    use_external_knowledge: bool = field(
        default_factory=lambda: os.getenv('USE_EXTERNAL_KNOWLEDGE', 'false').lower() == 'true'
    )
    layer_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "initial_understanding": 0.7,
            "relationship_analysis": 0.7,
            "contextual_integration": 0.9,
            "synthesis": 0.8
        }
    )
    adaptive_thresholds: bool = True
    min_threshold: float = 0.1
    max_threshold: float = 0.9
    threshold_step: float = 0.05
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert to bool if string value provided
        if isinstance(self.use_external_knowledge, str):
            self.use_external_knowledge = self.use_external_knowledge.lower() == 'true'

@dataclass
class Conversation:
    """Stores conversation history and context"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_context(self, n_messages: int = 5) -> str:
        """Get recent conversation context"""
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.messages[-n_messages:]
        ])

@dataclass
class KnowledgeBase:
    """Dynamic knowledge base that can load from multiple sources"""
    documents: Dict[str, List[str]] = field(default_factory=dict)
    required_categories: List[str] = field(
        default_factory=lambda: ['prompts', 'templates']
    )
    
    def load_from_json(self, filepath: str) -> None:
        """
        Load structured knowledge from JSON file.
        
        Args:
            filepath: Path to JSON file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                category = Path(filepath).parent.name
                if category not in self.documents:
                    self.documents[category] = []
                self.documents[category].append(json.dumps(data))
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON from {filepath}: {str(e)}")
    
    def load_from_directory(self, directory: str) -> None:
        """
        Load text files from a directory.
        
        Args:
            directory: Path to directory containing text files
        """
        try:
            dir_path = Path(directory)
            category = dir_path.name
            if category not in self.documents:
                self.documents[category] = []
            
            for file_path in dir_path.glob('*.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.documents[category].append(f.read())
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load directory {directory}: {str(e)}")
    
    def get_relevant_documents(self, category: Optional[str] = None) -> List[str]:
        """
        Get documents, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            List of document contents
        """
        if category:
            return self.documents.get(category, [])
        return [doc for docs in self.documents.values() for doc in docs]
    
    def is_required(self, category: str) -> bool:
        """Check if a category is required"""
        return category in self.required_categories

@dataclass
class CategoryConfig:
    """Configuration for category-specific processing."""
    name: str
    similarity_threshold: float
    concept_weight: float
    context_weight: float
    knowledge_weight: float

class CascadeSemanticLayerProcessor:
    """
    Main processor implementing the Cascade of Semantically Integrated Layers 
    algorithm.
    
    The CSIL algorithm processes user input through multiple layers of analysis:
    1. Initial Understanding: Basic concept extraction
    2. Semantic Analysis: Relationship discovery
    3. Context Integration: Broader implications
    4. Response Synthesis: Final coherent response
    """
    
    def __init__(self, config: CSILConfig = CSILConfig()):
        """Initialize Cascade of Semantically Integrated Layers with enhanced 
        semantic analysis."""
        self.config = config
        self.conversation = Conversation()
        self.knowledge_base = KnowledgeBase()
        self.last_query = ""
        
        # Modify vectorizer settings to handle small document counts
        self.vectorizer = TfidfVectorizer(
            stop_words=None,  # We'll handle stop words separately
            max_features=500,
            ngram_range=(1, 1),
            min_df=1,  # Changed from 2 to 1 to handle single documents
            max_df=1.0,  # Allow terms that appear in all documents
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w+\b'
        )
        
        ensure_nltk_resources()
        self.is_vectorizer_fitted = False
        self.corpus_texts = []
        
        # Remove thread pool from __init__
        # We'll create it when needed instead
        self._thread_pool = None
        
        # Add caching for vectorizer
        self._vector_cache = {}
        
        # Add category-specific configurations
        self.category_configs = {
            # Original categories
            "puzzle": CategoryConfig(
                "puzzle",
                similarity_threshold=0.6,
                concept_weight=0.7,
                context_weight=0.2,
                knowledge_weight=0.1
            ),
            "spatial": CategoryConfig(
                "spatial",
                similarity_threshold=0.7,
                concept_weight=0.6,
                context_weight=0.3,
                knowledge_weight=0.1
            ),
            # Scientific domains
            "physics": CategoryConfig(
                "physics",
                similarity_threshold=0.75,
                concept_weight=0.8,
                context_weight=0.1,
                knowledge_weight=0.1
            ),
            "mathematics": CategoryConfig(
                "mathematics",
                similarity_threshold=0.8,
                concept_weight=0.85,
                context_weight=0.1,
                knowledge_weight=0.05
            ),
            "chemistry": CategoryConfig(
                "chemistry",
                similarity_threshold=0.75,
                concept_weight=0.75,
                context_weight=0.15,
                knowledge_weight=0.1
            ),
            # Engineering & Technology
            "engineering": CategoryConfig(
                "engineering",
                similarity_threshold=0.7,
                concept_weight=0.65,
                context_weight=0.25,
                knowledge_weight=0.1
            ),
            "computer_science": CategoryConfig(
                "computer_science",
                similarity_threshold=0.75,
                concept_weight=0.7,
                context_weight=0.2,
                knowledge_weight=0.1
            ),
            # Philosophy & Logic
            "philosophy": CategoryConfig(
                "philosophy",
                similarity_threshold=0.65,
                concept_weight=0.6,
                context_weight=0.3,
                knowledge_weight=0.1
            ),
            "logic": CategoryConfig(
                "logic",
                similarity_threshold=0.8,
                concept_weight=0.8,
                context_weight=0.1,
                knowledge_weight=0.1
            ),
            # Creative & Inventive
            "invention": CategoryConfig(
                "invention",
                similarity_threshold=0.6,
                concept_weight=0.5,
                context_weight=0.3,
                knowledge_weight=0.2
            ),
            "innovation": CategoryConfig(
                "innovation",
                similarity_threshold=0.65,
                concept_weight=0.55,
                context_weight=0.25,
                knowledge_weight=0.2
            ),
            # Natural Sciences
            "biology": CategoryConfig(
                "biology",
                similarity_threshold=0.7,
                concept_weight=0.7,
                context_weight=0.2,
                knowledge_weight=0.1
            ),
            "astronomy": CategoryConfig(
                "astronomy",
                similarity_threshold=0.75,
                concept_weight=0.75,
                context_weight=0.15,
                knowledge_weight=0.1
            ),
            # Interdisciplinary
            "systems_thinking": CategoryConfig(
                "systems_thinking",
                similarity_threshold=0.65,
                concept_weight=0.6,
                context_weight=0.3,
                knowledge_weight=0.1
            ),
            "pattern_recognition": CategoryConfig(
                "pattern_recognition",
                similarity_threshold=0.7,
                concept_weight=0.7,
                context_weight=0.2,
                knowledge_weight=0.1
            ),
            # General Knowledge
            "general": CategoryConfig(
                "general",
                similarity_threshold=0.6,
                concept_weight=0.6,
                context_weight=0.2,
                knowledge_weight=0.2
            )
        }
        
        self.concept_graph = nx.Graph()
        self.knowledge_graph = nx.DiGraph()  # Add directed knowledge graph
        
    @property
    def thread_pool(self):
        """Lazy initialization of thread pool."""
        if self._thread_pool is None:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=4
            )
        return self._thread_pool

    @lru_cache(maxsize=1000)
    def _calculate_semantic_similarity(
        self, 
        text1: str, 
        text2: str,
        use_cache: bool = True
    ) -> float:
        """Calculate semantic similarity with improved caching."""
        # Create cache key
        cache_key = (text1.strip(), text2.strip())
        
        # Skip empty texts
        if not cache_key[0] or not cache_key[1]:
            return 0.0
            
        try:
            # Check cache if enabled
            if use_cache and cache_key in self._vector_cache:
                return self._vector_cache[cache_key]
            
            # Calculate similarity
            vectors = self.vectorizer.transform([text1, text2])
            cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Calculate Jaccard similarity
            words1 = set(word_tokenize(text1.lower()))
            words2 = set(word_tokenize(text2.lower()))
            jaccard = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
            
            # Combine similarities
            combined_sim = (0.7 * cos_sim) + (0.3 * jaccard)
            result = max(0.0, min(1.0, combined_sim))
            
            # Cache result if enabled
            if use_cache:
                self._vector_cache[cache_key] = result
                
                # Debug output only for new calculations
                if self.config.debug_mode:
                    print(f"\nSimilarity Calculation (new):")
                    print(f"- Cosine similarity: {cos_sim:.3f}")
                    print(f"- Jaccard similarity: {jaccard:.3f}")
                    print(f"- Combined similarity: {combined_sim:.3f}")
            
            return result
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Similarity calculation error: {str(e)}")
            return 0.0

    def _calculate_tfidf_weight(self, term: str, context: str) -> float:
        """
        Calculate TF-IDF weight for a term in given context.
        
        Args:
            term: Term to calculate weight for
            context: Context text
            
        Returns:
            float: TF-IDF weight
        """
        try:
            # Ensure vectorizer is fitted
            if not self.is_vectorizer_fitted:
                # Add context to corpus and fit vectorizer
                if context not in self.corpus_texts:
                    self.corpus_texts.append(context)
                self._fit_vectorizer()
                
            if not self.is_vectorizer_fitted:
                return 0.0
                
            # Transform single document
            vector = self.vectorizer.transform([context])
            
            # Get term index
            term_idx = None
            feature_names = self.vectorizer.get_feature_names_out()
            for idx, feature in enumerate(feature_names):
                if feature == term.lower():
                    term_idx = idx
                    break
                    
            if term_idx is not None:
                # Get weight from vector
                return vector[0, term_idx]
            return 0.0
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error calculating TF-IDF weight: {str(e)}")
            return 0.0

    def _calculate_position_weight(self, term: str, context: str) -> float:
        """
        Calculate position-based weight for a term.
        
        Args:
            term: Term to calculate weight for
            context: Context text
            
        Returns:
            float: Position-based weight (earlier positions get higher weight)
        """
        try:
            words = word_tokenize(context.lower())
            term_lower = term.lower()
            
            # Find first occurrence
            for i, word in enumerate(words):
                if word == term_lower:
                    # Exponential decay based on position
                    return np.exp(-i / len(words))
            return 0.0
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error calculating position weight: {str(e)}")
            return 0.0

    def _weight_concepts_by_importance(
        self,
        concepts: List[str],
        context: str
    ) -> List[Tuple[str, float]]:
        """Weight concepts by their importance in context."""
        # Ensure vectorizer is fitted before processing
        if not self.is_vectorizer_fitted:
            if context not in self.corpus_texts:
                self.corpus_texts.append(context)
            self._fit_vectorizer()
            
        weights = []
        for concept in concepts:
            try:
                # Calculate TF-IDF weight
                tfidf_weight = self._calculate_tfidf_weight(concept, context)
                
                # Calculate position weight
                position_weight = self._calculate_position_weight(concept, context)
                
                # Calculate frequency weight
                freq_weight = context.lower().count(concept.lower()) / len(context.split())
                
                # Combine weights
                combined_weight = (
                    0.5 * tfidf_weight +
                    0.3 * position_weight +
                    0.2 * freq_weight
                )
                weights.append((concept, combined_weight))
                
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Error weighting concept {concept}: {str(e)}")
                weights.append((concept, 0.0))
                
        return sorted(weights, key=lambda x: x[1], reverse=True)

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts with improved error handling."""
        if not text or not text.strip():
            return []
            
        try:
            # Ensure vectorizer is fitted
            if not self.is_vectorizer_fitted:
                if text not in self.corpus_texts:
                    self.corpus_texts.append(text)
                self._fit_vectorizer()
            
            # Single-threaded processing to avoid thread pool issues
            tokens = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            
            # Enhanced filtering criteria
            filtered_tokens = [
                t for t in tokens 
                if (
                    len(t) > 1 and
                    not t.isnumeric() and
                    not t in stop_words and
                    not all(char in string.punctuation for char in t)
                )
            ]
            
            if not filtered_tokens:
                return tokens[:self.config.min_keywords]
            
            # Calculate weights and select terms
            document = ' '.join(filtered_tokens)
            weighted_terms = self._weight_concepts_by_importance(filtered_tokens, document)
            
            # Apply limits
            num_terms = max(
                self.config.min_keywords,
                min(len(weighted_terms), self.config.max_keywords)
            )
            
            selected_terms = weighted_terms[:num_terms]
            
            if self.config.debug_mode:
                print("\nExtracted concepts:")
                for term, weight in selected_terms:
                    print(f"  • {term}: {weight:.3f}")
            
            return [term for term, _ in selected_terms]
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in concept extraction: {str(e)}")
            return text.split()[:self.config.min_keywords]

    def __del__(self):
        """Cleanup thread pool on object deletion."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False)

    def process_semantic_cascade(self, user_input: str) -> Dict[str, Any]:
        """Enhanced semantic cascade with adaptive processing."""
        try:
            self.last_query = user_input
            
            # Initialize results dictionary
            results = {
                'initial_understanding': '',
                'relationships': '',
                'context_integration': '',
                'final_response': '',
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'concepts_extracted': [],
                    'similarity_scores': {}
                }
            }
            
            if self.config.debug_mode:
                print("\n🤔 Starting semantic cascade...")
                print(f"Corpus size before: {len(self.corpus_texts)}")
            
            # Add user input to corpus
            if user_input not in self.corpus_texts:
                self.corpus_texts.append(user_input)
                
            # Add conversation context to corpus
            context = self.conversation.get_recent_context()
            if context and context not in self.corpus_texts:
                self.corpus_texts.append(context)
                
            # Add knowledge base documents to corpus
            if self.config.use_external_knowledge:
                relevant_docs = self.knowledge_base.get_relevant_documents()
                self.corpus_texts.extend([doc for doc in relevant_docs 
                                        if doc not in self.corpus_texts])
            
            # Process each layer
            results['initial_understanding'] = self._process_layer(
                "initial_understanding",
                user_input,
                "",  # No previous output for first layer
                "Identify ONLY the fundamental concepts and questions."
            )
            
            results['relationships'] = self._process_layer(
                "relationship_analysis",
                user_input,
                results['initial_understanding'],
                "Discover NEW connections between the identified concepts."
            )
            
            results['context_integration'] = self._process_layer(
                "contextual_integration",
                user_input,
                results['relationships'],
                "Add BROADER context and implications not yet discussed."
            )
            
            results['final_response'] = self._process_layer(
                "synthesis",
                user_input,
                results['context_integration'],
                "Create a cohesive response that builds upon all previous layers."
            )
            
            return results
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in semantic cascade: {str(e)}")
            # Return a structured error response
            return {
                'initial_understanding': '',
                'relationships': '',
                'context_integration': '',
                'final_response': f"Error processing query: {str(e)}",
                'error': str(e)
            }

    def _process_with_threshold(
        self,
        user_input: str,
        threshold: float
    ) -> Dict[str, Any]:
        """Process with specific threshold."""
        try:
            temp_config = dataclasses.replace(
                self.config,
                similarity_threshold=threshold
            )
            
            # Process through semantic cascade
            return self.process_semantic_cascade(user_input)
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in threshold processing: {str(e)}")
            return {
                'error': str(e),
                'threshold_used': threshold
            }

    def process_interaction(self, user_input: str) -> Dict[str, Any]:
        """Enhanced interaction processing with category detection."""
        try:
            # Detect category
            category = self._detect_category(user_input)
            
            # Get category-specific config
            category_config = self.category_configs.get(category)
            
            if category_config:
                # Temporarily adjust config for this interaction
                original_config = self.config
                self.config = CSILConfig(
                    **{
                        **dataclasses.asdict(original_config),
                        "similarity_threshold": category_config.similarity_threshold
                    }
                )
                
                try:
                    # Process through semantic cascade directly
                    results = self.process_semantic_cascade(user_input)
                finally:
                    # Restore original config
                    self.config = original_config
                    
                return results
            else:
                # Use default processing if no specific category config
                return self.process_semantic_cascade(user_input)
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in interaction processing: {str(e)}")
            return {
                'error': str(e),
                'initial_understanding': '',
                'relationships': '',
                'context_integration': '',
                'final_response': f"Error processing query: {str(e)}"
            }

    def _calculate_dynamic_temperature(
        self,
        novelty_score: float,
        layer_name: str = None,
        context_complexity: float = None
    ) -> float:
        """
        Enhanced temperature calculation with context complexity.
        
        Args:
            novelty_score: Base novelty score
            layer_name: Processing layer name
            context_complexity: Optional complexity score of current context
            
        Returns:
            float: Optimized temperature value
        """
        try:
            base_temp = self.config.llm_config.temperature
            
            # Enhanced layer-specific modifiers
            layer_modifier = {
                "initial_understanding": {
                    'base': 0.8,
                    'novelty_weight': 0.3,
                    'complexity_weight': 0.2
                },
                "relationship_analysis": {
                    'base': 1.0,
                    'novelty_weight': 0.4,
                    'complexity_weight': 0.3
                },
                "contextual_integration": {
                    'base': 1.2,
                    'novelty_weight': 0.5,
                    'complexity_weight': 0.4
                },
                "synthesis": {
                    'base': 0.9,
                    'novelty_weight': 0.3,
                    'complexity_weight': 0.3
                }
            }.get(layer_name, {
                'base': 1.0,
                'novelty_weight': 0.3,
                'complexity_weight': 0.3
            })
            
            # Calculate complexity factor
            complexity_factor = (
                context_complexity if context_complexity is not None 
                else 0.5
            )
            
            # Enhanced temperature calculation
            adjusted_temp = (
                base_temp * 
                layer_modifier['base'] * 
                (1 + (novelty_score * layer_modifier['novelty_weight'])) *
                (1 + (complexity_factor * layer_modifier['complexity_weight']))
            )
            
            return max(0.1, min(1.0, adjusted_temp))
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Temperature calculation error: {str(e)}")
            return 0.7

    def _call_llm(
        self,
        prompt: str,
        user_input: str,
        config: Dict[str, Any] = None
    ) -> str:
        """
        Make API call to LLM with enhanced debug output.
        
        Args:
            prompt: System prompt
            user_input: User query
            config: Optional configuration overrides
            
        Returns:
            str: LLM response
        """
        try:
            if self.config.debug_mode:
                print("\n🔍 LLM Call Details:")
                print(f"  Temperature: {config.get('temperature', 'default')}")
                print(f"  Model: {config.get('model', 'default')}")
                print(f"  Stream: {config.get('stream', True)}")
                print("\n📝 Prompt:")
                print(f"  {prompt[:200]}...")  # Show first 200 chars
                
            # Use provided config or default
            if config is None:
                config = {
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt
                        },
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ],
                    "temperature": self.config.llm_config.temperature,
                    "stream": self.config.llm_config.stream,
                    "model": self.config.llm_config.model
                }
            
            # Make API request
            if self.config.debug_mode:
                print("\n🌐 Making API request...")
                
            response = requests.post(
                self.config.llm_config.url,
                json=config,
                stream=config.get("stream", True)
            )
            response.raise_for_status()

            if config.get("stream", True):
                if self.config.debug_mode:
                    print("\n💭 Streaming response:")
                    
                client = sseclient.SSEClient(response)
                full_response = ""
                for event in client.events():
                    try:
                        chunk = json.loads(event.data)
                        if 'choices' in chunk and chunk['choices']:
                            content = chunk['choices'][0].get('delta', {}).get(
                                'content', ''
                            )
                            full_response += content
                            if self.config.debug_mode:
                                print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        if self.config.debug_mode:
                            print("⚠️ JSON decode error in stream")
                        continue
                        
                if self.config.debug_mode:
                    print("\n✅ Stream complete")
                return full_response
            else:
                response_json = response.json()
                if self.config.debug_mode:
                    print("\n📤 Non-streaming response received")
                return response_json['choices'][0]['message']['content']

        except Exception as e:
            if self.config.debug_mode:
                print(f"\n❌ LLM API call error: {str(e)}")
                print(f"  URL: {self.config.llm_config.url}")
                print(f"  Config: {json.dumps(config, indent=2)}")
            return f"Error in LLM processing: {str(e)}"

    def _calculate_concept_novelty(
        self, 
        current_concepts: List[str], 
        previous_concepts: set
    ) -> float:
        """
        Calculate novelty score using semantic similarity and concept weights.
        
        Args:
            current_concepts: New concepts from current layer
            previous_concepts: Previously seen concepts
            
        Returns:
            float: Novelty score between 0 and 1, weighted by concept importance
        """
        if not current_concepts or not previous_concepts:
            return 1.0
        
        # Convert concepts to sentence form for better vectorization
        current_texts = [' '.join(c.split('_')) for c in current_concepts]
        previous_texts = [' '.join(c.split('_')) for c in previous_concepts]
        
        # Calculate TF-IDF weights for current concepts
        self._fit_vectorizer()
        current_vectors = self.vectorizer.transform(current_texts)
        
        # Calculate maximum similarity for each current concept
        novelty_scores = []
        for i, current_vec in enumerate(current_vectors):
            similarities = cosine_similarity(
                current_vec, 
                self.vectorizer.transform(previous_texts)
            )[0]
            # Use complement of maximum similarity as novelty
            concept_novelty = 1.0 - (max(similarities) if len(similarities) > 0 else 0.0)
            
            # Weight by concept importance (TF-IDF magnitude)
            concept_weight = np.sqrt(np.sum(current_vec.toarray() ** 2))
            novelty_scores.append(concept_novelty * concept_weight)
        
        # Aggregate weighted novelty scores
        if novelty_scores:
            avg_novelty = sum(novelty_scores) / len(novelty_scores)
            return max(0.1, min(1.0, avg_novelty))
        return 1.0

    def _generate_enhanced_prompt(
        self,
        layer_name: str,
        user_input: str,
        previous_output: str,
        weights: Dict[str, float],
        novelty_score: float,
        knowledge: Dict[str, Any]
    ) -> str:
        """
        Generate enhanced system prompt for LLM based on layer context.
        
        Args:
            layer_name: Current processing layer name
            user_input: Original user query
            previous_output: Output from previous layer
            weights: Layer-specific weights for concept vs practical focus
            novelty_score: Calculated novelty score for concepts
            knowledge: Knowledge base content for the current layer
            
        Returns:
            str: Enhanced system prompt for LLM
        """
        try:
            # Load prompts from knowledge base
            system_prompts = self.knowledge_base.get_relevant_documents('prompts')
            prompts_data = json.loads(system_prompts[0])
            
            # Get base and layer-specific prompts
            base_prompt = prompts_data['base_prompt'][0]
            layer_prompts = prompts_data['layer_prompts'].get(
                layer_name, 
                ["Analyze the given input"]
            )
            
            # Get conversation templates
            templates = self.knowledge_base.get_relevant_documents('templates')
            template_data = json.loads(templates[0])
            connectors = template_data.get('thought_connectors', [])
            
            # Build enhanced prompt
            prompt_parts = [
                base_prompt,
                f"\nContext from previous layer:\n{previous_output}",
                f"\nLayer focus ({layer_name}):",
                *layer_prompts,
                f"\nConcept weight: {weights['concept_weight']:.2f}",
                f"Practical weight: {weights['practical_weight']:.2f}",
                f"Novelty threshold: {novelty_score:.2f}",
                "\nThought connectors to consider:",
                *connectors[:2]  # Use first two connectors
            ]
            
            # Add relevant knowledge to prompt
            if knowledge['concepts']:
                prompt_parts.extend([
                    "\nRelevant Concepts:",
                    *[str(c) for c in knowledge['concepts'][:2]]
                ])
                
            if knowledge['examples']:
                prompt_parts.extend([
                    "\nRelevant Examples:",
                    *knowledge['examples'][:1]
                ])
                
            if knowledge['context']:
                prompt_parts.extend([
                    "\nBroader Context:",
                    *knowledge['context'][:1]
                ])
                
            return "\n".join(prompt_parts)
            
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            # Fallback prompt if there's an error
            return (
                f"You are an expert system. "
                f"Current layer: {layer_name}. "
                f"Previous output: {previous_output}"
            )

    def _integrate_knowledge_base(
        self,
        layer_name: str,
        concepts: List[str]
    ) -> Dict[str, Any]:
        """
        Integrate knowledge base with fallback handling.
        
        Args:
            layer_name: Current processing layer name
            concepts: List of concepts to match against
            
        Returns:
            Dict containing relevant knowledge components
        """
        knowledge = {
            'concepts': [],
            'examples': [],
            'context': []
        }
        
        if not self.config.use_external_knowledge:
            return knowledge
            
        try:
            # Deduplicate concepts
            unique_concepts = list(dict.fromkeys(concepts))
            
            # Process concepts directly without semantic search
            scored_concepts = self._process_concepts(
                self.knowledge_base.get_relevant_documents('concepts'),
                unique_concepts
            )
            knowledge['concepts'] = [c for _, c in scored_concepts[:3]]
            
            # Process examples
            scored_examples = self._process_examples(
                self.knowledge_base.get_relevant_documents('examples'),
                unique_concepts,
                layer_name
            )
            knowledge['examples'] = [e for _, e in scored_examples[:2]]
            
            # Process context
            scored_context = self._process_context(
                self.knowledge_base.get_relevant_documents('context'),
                unique_concepts,
                layer_name
            )
            knowledge['context'] = [c for _, c in scored_context[:2]]
            
            return knowledge
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Knowledge integration error: {str(e)}")
            return knowledge  # Return empty knowledge dict on error

    def _parse_documents(self, docs: List[str]) -> List[Any]:
        """Parse documents handling both JSON and plain text formats."""
        parsed_data = []
        for doc in docs:
            try:
                if isinstance(doc, str):
                    if doc.strip():
                        try:
                            parsed_doc = json.loads(doc)
                            if isinstance(parsed_doc, list):
                                parsed_data.extend(parsed_doc)
                            else:
                                parsed_data.append(parsed_doc)
                        except json.JSONDecodeError:
                            parsed_data.append(doc)
                elif isinstance(doc, (dict, list)):
                    if isinstance(doc, list):
                        parsed_data.extend(doc)
                    else:
                        parsed_data.append(doc)
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Warning: Failed to parse document: {str(e)}")
                continue
        return parsed_data

    def _fit_vectorizer(self) -> None:
        """Optimized vectorizer fitting with batching and caching."""
        try:
            # Only refit if corpus has significantly changed
            current_corpus_hash = hash(tuple(sorted(self.corpus_texts)))
            if (hasattr(self, '_last_corpus_hash') and 
                current_corpus_hash == self._last_corpus_hash):
                return

            if not self.is_vectorizer_fitted or len(self.corpus_texts) > 0:
                valid_texts = [text for text in self.corpus_texts 
                             if text and text.strip()]
                
                if not valid_texts:
                    return
                    
                # Implement batched processing for large corpora
                BATCH_SIZE = 1000
                if len(valid_texts) > BATCH_SIZE:
                    # Process in batches
                    for i in range(0, len(valid_texts), BATCH_SIZE):
                        batch = valid_texts[i:i + BATCH_SIZE]
                        if i == 0:
                            self.vectorizer.fit(batch)
                        else:
                            # Use partial_fit if available, otherwise refit
                            if hasattr(self.vectorizer, 'partial_fit'):
                                self.vectorizer.partial_fit(batch)
                else:
                    self.vectorizer.fit(valid_texts)
                
                self.is_vectorizer_fitted = True
                self._last_corpus_hash = current_corpus_hash
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in vectorizer fitting: {str(e)}")

    def analyze_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Analyze user input text directly against knowledge base and context.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Dict containing semantic analysis results
        """
        # Add user input to corpus
        if user_input not in self.corpus_texts:
            self.corpus_texts.append(user_input)
        
        # Extract concepts from user input
        user_concepts = self._extract_key_concepts(user_input)
        
        # Calculate similarities with knowledge base documents
        kb_similarities = {}
        for category, docs in self.knowledge_base.documents.items():
            similarities = []
            for doc in docs:
                sim = self._calculate_semantic_similarity(user_input, doc)
                similarities.append(sim)
            kb_similarities[category] = max(similarities) if similarities else 0.0
        
        # Calculate similarity with conversation context
        context = self.conversation.get_recent_context()
        context_similarity = self._calculate_semantic_similarity(
            user_input, 
            context
        )
        
        return {
            'user_concepts': user_concepts,
            'knowledge_base_similarities': kb_similarities,
            'context_similarity': context_similarity,
            'corpus_size': len(self.corpus_texts)
        }

    def _detect_category(self, user_input: str) -> str:
        """Detect input category using semantic analysis."""
        category_indicators = {
            "puzzle": ["solve", "puzzle", "riddle", "game"],
            "spatial": ["position", "direction", "location", "space"]
        }
        
        input_lower = user_input.lower()
        max_score = 0
        detected_category = None
        
        for category, indicators in category_indicators.items():
            score = sum(1 for ind in indicators if ind in input_lower)
            if score > max_score:
                max_score = score
                detected_category = category
                
        return detected_category or "general"

    def _build_concept_graph(self, concepts: List[str]) -> None:
        """Build graph of concept relationships."""
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                similarity = self._calculate_semantic_similarity(concept1, concept2)
                if similarity > self.config.similarity_threshold:
                    self.concept_graph.add_edge(concept1, concept2, weight=similarity)
                    
    def _analyze_concept_relationships(self) -> Dict[str, float]:
        """Analyze relationships using graph centrality."""
        return pagerank(self.concept_graph)  # Get concept importance

    def _cluster_concepts(
        self,
        concepts: List[str],
        min_cluster_size: int = 2
    ) -> Dict[str, List[str]]:
        """
        Enhanced concept clustering with dimensional reduction.
        
        Args:
            concepts: List of concepts to cluster
            min_cluster_size: Minimum size for valid clusters
            
        Returns:
            Dict mapping cluster IDs to concept lists
        """
        try:
            if len(concepts) < min_cluster_size:
                return {0: concepts}
                
            # Get concept vectors
            vectors = self.vectorizer.transform(concepts)
            
            # Apply dimensional reduction
            svd = TruncatedSVD(n_components=min(vectors.shape[1], 50))
            reduced_vectors = svd.fit_transform(vectors)
            
            # Adaptive DBSCAN
            nn = NearestNeighbors(n_neighbors=min_cluster_size)
            nn.fit(reduced_vectors)
            distances, _ = nn.kneighbors(reduced_vectors)
            
            # Calculate optimal epsilon
            optimal_eps = np.percentile(distances[:, -1], 75)
            
            # Perform clustering
            clustering = DBSCAN(
                eps=optimal_eps,
                min_samples=min_cluster_size,
                metric='euclidean'
            ).fit(reduced_vectors)
            
            # Group concepts by cluster
            clusters = {}
            for concept, label in zip(concepts, clustering.labels_):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(concept)
                
            return clusters
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Clustering error: {str(e)}")
            return {0: concepts}

    def _process_layer(
        self,
        layer_name: str,
        user_input: str,
        previous_output: str,
        layer_instruction: str
    ) -> str:
        """
        Enhanced layer processing with debug output.
        """
        try:
            if self.config.debug_mode:
                print(f"\n🔄 Processing Layer: {layer_name}")
                print("  ├─ Extracting concepts...")
                
            # Extract concepts for this layer
            concepts = self._extract_key_concepts(user_input)
            
            if self.config.debug_mode:
                print("  ├─ Concepts extracted:")
                for concept in concepts[:5]:  # Show top 5 concepts
                    print(f"    • {concept}")
                print("  ├─ Calculating novelty...")
                
            # Track previously seen concepts
            previous_concepts = set(self._extract_key_concepts(previous_output))
            novelty_score = self._calculate_concept_novelty(
                concepts, 
                previous_concepts
            )
            
            if self.config.debug_mode:
                print(f"  ├─ Novelty score: {novelty_score:.2f}")
                print("  ├─ Integrating knowledge...")
                
            # Get layer-specific configuration
            layer_configs = {
                "initial_understanding": {
                    "instructions": [
                        "Identify and extract key concepts, entities, and relationships.",
                        "Break down complex queries into fundamental components.",
                        "Highlight any ambiguities or unclear aspects that need clarification.",
                    ],
                    "processing_style": {
                        "response_format": "structured",
                        "detail_level": "high",
                        "focus": "analytical",
                        "creativity": 0.3
                    }
                },
                "relationship_analysis": {
                    "instructions": [
                        "Analyze connections between previously identified concepts.",
                        "Discover hidden or implicit relationships in the context.",
                        "Map concept hierarchies and dependencies.",
                    ],
                    "processing_style": {
                        "response_format": "analytical",
                        "detail_level": "medium",
                        "focus": "relationships",
                        "creativity": 0.5
                    }
                },
                "contextual_integration": {
                    "instructions": [
                        "Integrate findings with broader domain knowledge.",
                        "Consider real-world implications and applications.",
                        "Identify relevant examples and analogies.",
                    ],
                    "processing_style": {
                        "response_format": "exploratory",
                        "detail_level": "high",
                        "focus": "contextual",
                        "creativity": 0.7
                    }
                },
                "synthesis": {
                    "instructions": [
                        "Synthesize all previous analyses into a coherent response.",
                        "Ensure practical applicability of the final answer.",
                        "Balance technical accuracy with understandability.",
                    ],
                    "processing_style": {
                        "response_format": "cohesive",
                        "detail_level": "balanced",
                        "focus": "practical",
                        "creativity": 0.4
                    }
                }
            }
            
            # Get layer config or use defaults
            layer_config = layer_configs.get(layer_name, {
                "instructions": [layer_instruction],
                "processing_style": {
                    "response_format": "standard",
                    "detail_level": "medium",
                    "focus": "general",
                    "creativity": 0.5
                }
            })
            
            # Calculate weights based on processing style
            weights = {
                "concept_weight": 0.8 if layer_config["processing_style"]["detail_level"] == "high" else 0.6,
                "practical_weight": 0.8 if layer_config["processing_style"]["focus"] == "practical" else 0.6
            }
            
            # Integrate knowledge base
            knowledge = self._integrate_knowledge_base(layer_name, concepts)
            
            if self.config.debug_mode:
                print("  ├─ Building prompt...")
                
            # Build enhanced prompt with all required arguments
            prompt = self._generate_enhanced_layer_prompt(
                layer_name=layer_name,
                user_input=user_input,
                previous_output=previous_output,
                instructions=layer_config["instructions"],
                weights=weights,
                novelty_score=novelty_score,  # Added missing argument
                knowledge=knowledge,  # Added missing argument
                processing_style=layer_config["processing_style"]  # Added missing argument
            )
            
            if self.config.debug_mode:
                print("  ├─ Calculating temperature...")
                
            # Calculate dynamic temperature
            temperature = self._calculate_dynamic_temperature(
                novelty_score,
                layer_name
            )
            
            if self.config.debug_mode:
                print(f"  └─ Temperature: {temperature:.2f}")
                print("\n🤖 Calling LLM...")
                
            # Process through LLM with style-specific parameters
            return self._call_llm_with_style(
                prompt,
                user_input,
                temperature,
                layer_config["processing_style"]
            )
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"\n❌ Error in layer {layer_name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
            return f"Error processing {layer_name} layer: {str(e)}"

    def _generate_enhanced_layer_prompt(
        self,
        layer_name: str,
        user_input: str,
        previous_output: str,
        instructions: List[str],
        weights: Dict[str, float],
        novelty_score: float,
        knowledge: Dict[str, Any],
        processing_style: Dict[str, Any]
    ) -> str:
        """
        Generate enhanced system prompt for LLM based on layer context.
        
        Args:
            layer_name: Current processing layer name
            user_input: Original user query
            previous_output: Output from previous layer
            instructions: Layer-specific instructions
            weights: Layer-specific weights
            novelty_score: Calculated novelty score
            knowledge: Knowledge base content
            processing_style: Processing style configuration
            
        Returns:
            str: Enhanced system prompt for LLM
        """
        try:
            # Build enhanced prompt
            prompt_parts = [
                f"You are an expert system focused on {layer_name}.",
                "\nProcessing Guidelines:",
                *[f"- {instr}" for instr in instructions],
                f"\nResponse Style: {processing_style['response_format']}",
                f"Detail Level: {processing_style['detail_level']}",
                f"Focus Area: {processing_style['focus']}",
                "\nContext:",
                f"Previous Analysis: {previous_output}",
                f"Original Query: {user_input}",
                f"Concept Weight: {weights['concept_weight']:.2f}",
                f"Practical Weight: {weights['practical_weight']:.2f}",
                f"Novelty Score: {novelty_score:.2f}"
            ]
            
            # Add knowledge base content if available
            if knowledge['concepts']:
                prompt_parts.extend([
                    "\nRelevant Concepts:",
                    *[f"- {c}" for c in knowledge['concepts'][:3]]
                ])
                
            if knowledge['examples']:
                prompt_parts.extend([
                    "\nRelevant Examples:",
                    *[f"- {e}" for e in knowledge['examples'][:2]]
                ])
                
            if knowledge['context']:
                prompt_parts.extend([
                    "\nBroader Context:",
                    *[f"- {c}" for c in knowledge['context'][:2]]
                ])
                
            return "\n".join(prompt_parts)
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error generating layer prompt: {str(e)}")
            return f"Error in prompt generation: {str(e)}"

    def _call_llm_with_style(
        self,
        prompt: str,
        user_input: str,
        temperature: float,
        style: Dict[str, Any]
    ) -> str:
        """
        Enhanced LLM call with style-specific parameters.
        """
        try:
            # Build request configuration
            config = {
                "messages": [
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                "temperature": temperature,
                "stream": self.config.llm_config.stream,
                "model": self.config.llm_config.model
            }
            
            # Add style-specific adjustments
            if style["response_format"] == "creative":
                config["top_p"] = 0.9
            else:
                config["top_p"] = 0.7
                
            if style["detail_level"] == "high":
                config["frequency_penalty"] = 0.3
            else:
                config["frequency_penalty"] = 0.1
                
            if style["focus"] == "exploratory":
                config["presence_penalty"] = 0.3
            else:
                config["presence_penalty"] = 0.1
                
            # Make API call
            return self._call_llm(prompt, user_input, config)
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error in styled LLM call: {str(e)}")
            return f"Error in LLM processing: {str(e)}"

    def _process_concepts(self, concepts_data: List[Any], query_concepts: List[str]) -> List[Tuple[float, Any]]:
        """
        Process and score concept documents.
        
        Args:
            concepts_data: List of concept documents to process
            query_concepts: List of concepts from the query
            
        Returns:
            List of tuples containing (score, concept)
        """
        scored_concepts = []
        for concept_item in concepts_data:
            try:
                concept_text = (
                    json.dumps(concept_item) if isinstance(concept_item, (dict, list))
                    else str(concept_item)
                )
                
                # Calculate similarity with each query concept
                concept_similarities = [
                    self._calculate_semantic_similarity(
                        c.lower(),
                        concept_text.lower()
                    ) for c in query_concepts
                ]
                
                # Use maximum similarity as the score
                max_similarity = max(concept_similarities) if concept_similarities else 0.0
                
                if max_similarity > self.config.similarity_threshold:
                    scored_concepts.append((max_similarity, concept_item))
                    
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Error processing concept: {str(e)}")
                continue
                
        return sorted(scored_concepts, reverse=True)

    def _process_examples(
        self, 
        examples_data: List[Any], 
        concepts: List[str],
        layer_name: str
    ) -> List[Tuple[float, Any]]:
        """
        Process and score example documents.
        
        Args:
            examples_data: List of example documents to process
            concepts: List of concepts to match against
            layer_name: Current processing layer name
            
        Returns:
            List of tuples containing (score, example)
        """
        scored_examples = []
        
        # Get layer-specific weights
        weights = {
            "initial_understanding": 0.8,
            "relationship_analysis": 0.7,
            "contextual_integration": 0.6,
            "synthesis": 0.5
        }.get(layer_name, 0.6)
        
        for example in examples_data:
            try:
                example_text = (
                    json.dumps(example) if isinstance(example, (dict, list))
                    else str(example)
                )
                
                # Calculate weighted similarity scores
                similarities = [
                    weights * self._calculate_semantic_similarity(
                        c.lower(), 
                        example_text.lower()
                    ) for c in concepts
                ]
                
                # Use average of top 2 similarities as score
                top_scores = sorted(similarities, reverse=True)[:2]
                avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
                
                if avg_score > self.config.similarity_threshold:
                    scored_examples.append((avg_score, example))
                    
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Error processing example: {str(e)}")
                continue
                
        return sorted(scored_examples, reverse=True)

    def _process_context(
        self, 
        context_data: List[Any], 
        concepts: List[str],
        layer_name: str
    ) -> List[Tuple[float, Any]]:
        """
        Process and score context documents.
        
        Args:
            context_data: List of context documents to process
            concepts: List of concepts to match against
            layer_name: Current processing layer name
            
        Returns:
            List of tuples containing (score, context)
        """
        scored_contexts = []
        
        # Layer-specific context weights
        weights = {
            "initial_understanding": 0.5,
            "relationship_analysis": 0.7,
            "contextual_integration": 0.9,
            "synthesis": 0.8
        }.get(layer_name, 0.7)
        
        for context in context_data:
            try:
                context_text = (
                    json.dumps(context) if isinstance(context, (dict, list))
                    else str(context)
                )
                
                # Calculate semantic similarity with concepts
                concept_scores = [
                    weights * self._calculate_semantic_similarity(
                        c.lower(),
                        context_text.lower()
                    ) for c in concepts
                ]
                
                # Use weighted average of all similarities
                avg_score = (
                    sum(concept_scores) / len(concept_scores) 
                    if concept_scores else 0
                )
                
                if avg_score > self.config.similarity_threshold:
                    scored_contexts.append((avg_score, context))
                    
            except Exception as e:
                if self.config.debug_mode:
                    print(f"Error processing context: {str(e)}")
                continue
                
        return sorted(scored_contexts, reverse=True)

    def _update_knowledge_graph(
        self, 
        concepts: List[str], 
        relationships: List[Tuple[str, str, float]]
    ) -> None:
        """
        Update internal knowledge graph with new concepts and relationships.
        
        Args:
            concepts: List of concepts to add
            relationships: List of (concept1, concept2, weight) relationships
        """
        # Add new concepts
        for concept in concepts:
            if concept not in self.knowledge_graph:
                self.knowledge_graph.add_node(
                    concept, 
                    first_seen=datetime.now(),
                    frequency=1
                )
            else:
                # Update existing concept
                self.knowledge_graph.nodes[concept]['frequency'] += 1
                
        # Add or update relationships
        for c1, c2, weight in relationships:
            if self.knowledge_graph.has_edge(c1, c2):
                # Update existing relationship weight
                current_weight = self.knowledge_graph[c1][c2]['weight']
                new_weight = (current_weight + weight) / 2
                self.knowledge_graph[c1][c2]['weight'] = new_weight
            else:
                # Add new relationship
                self.knowledge_graph.add_edge(c1, c2, weight=weight)

    def _extract_subgraph(self, concepts: List[str], depth: int = 2) -> nx.DiGraph:
        """
        Extract relevant subgraph for given concepts.
        
        Args:
            concepts: Seed concepts to start from
            depth: How many hops to explore
            
        Returns:
            Subgraph containing relevant concepts and relationships
        """
        relevant_nodes: Set[str] = set(concepts)
        
        # Expand by exploring neighbors up to specified depth
        for _ in range(depth):
            neighbors = set()
            for node in relevant_nodes:
                if node in self.knowledge_graph:
                    neighbors.update(self.knowledge_graph.neighbors(node))
            relevant_nodes.update(neighbors)
            
        return self.knowledge_graph.subgraph(relevant_nodes)

    def analyze_knowledge_graph(self) -> Dict[str, Any]:
        """
        Analyze the current state of the knowledge graph.
        
        Returns:
            Dict containing various graph metrics and insights
        """
        return {
            'num_concepts': self.knowledge_graph.number_of_nodes(),
            'num_relationships': self.knowledge_graph.number_of_edges(),
            'central_concepts': nx.pagerank(self.knowledge_graph),
            'communities': list(nx.community.greedy_modularity_communities(
                self.knowledge_graph.to_undirected()
            )),
            'average_clustering': nx.average_clustering(
                self.knowledge_graph.to_undirected()
            )
        }

    def _analyze_subgraph(self, subgraph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze subgraph for relevant insights."""
        return {
            'central_concepts': nx.pagerank(subgraph),
            'communities': list(nx.community.greedy_modularity_communities(
                subgraph.to_undirected()
            )),
            'concept_clusters': self._cluster_concepts(
                list(subgraph.nodes())
            ),
            'key_relationships': [
                (u, v, d['weight']) 
                for u, v, d in subgraph.edges(data=True)
                if d['weight'] > self.config.similarity_threshold
            ]
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
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}", end=end)

# # Sample document corpus
# document_corpus = [
#     "The quick brown fox jumps over the lazy dog.",
#     "A lazy cat sleeps under the warm sun.",
#     "The dog chases the squirrel up the tree.",
#     "The cat and the dog are best friends.",
# ]

# # User query
# query = "What is the relationship between cats and dogs?"

# # Initialize with custom configuration
# scp = SemanticCascadeProcessor(SCPConfig(
#     min_keywords=2
# ))

# # Load knowledge base
# scp.knowledge_base.load_from_directory('knowledge_base/concepts')
# scp.knowledge_base.load_from_directory('knowledge_base/examples')
# scp.knowledge_base.load_from_json('knowledge_base/prompts/system_prompts.json')
# scp.knowledge_base.load_from_json('knowledge_base/prompts/conversation_templates.json')

# # Process query directly through semantic cascade layers
# results = scp.process_interaction("What is the relationship between cats and dogs?")
# print("\nSemantic Cascade Process:")
# print("1. Concept Extraction:", results['initial_understanding'])
# print("2. Semantic Analysis:", results['relationships'])
# print("3. Context Integration:", results['context_integration'])
# print("4. Response Synthesis:", results['final_response'])
