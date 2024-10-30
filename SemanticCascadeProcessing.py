"""
Semantic Cascade Processing (SCP) Module

This module implements the Semantic Cascade Processing algorithm, which combines
computational semantic analysis with progressive multi-layer processing for
enhanced natural language understanding and response generation.

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
from typing import List, Dict, Optional, Any
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
    url: str = os.getenv('LLM_URL', "http://0.0.0.0:11434/v1/chat/completions")
    model: str = os.getenv('LLM_MODEL', "hf.co/arcee-ai/SuperNova-Medius-GGUF:f16")
    context_window: int = int(os.getenv('LLM_CONTEXT_WINDOW', 8192))
    temperature: float = float(os.getenv('LLM_TEMPERATURE', 0.6))
    max_tokens: int = int(os.getenv('LLM_MAX_TOKENS', 4096))
    stream: bool = os.getenv('LLM_STREAM', 'true').lower() == 'true'
    top_p: float = float(os.getenv('LLM_TOP_P', 0.9))
    frequency_penalty: float = float(os.getenv('LLM_FREQUENCY_PENALTY', 0.0))
    presence_penalty: float = float(os.getenv('LLM_PRESENCE_PENALTY', 0.0))
    stop_sequences: List[str] = field(default_factory=list)
    repeat_penalty: float = float(os.getenv('LLM_REPEAT_PENALTY', 1.1))
    seed: Optional[int] = int(os.getenv('LLM_SEED')) if os.getenv('LLM_SEED') else None

@dataclass
class SCPConfig:
    """Configuration for Semantic Cascade Processing.
    
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
    """
    min_keywords: int = 2
    max_keywords: int = 10
    keyword_weight_threshold: float = 0.1
    similarity_threshold: float = 0.1
    max_results: Optional[int] = None
    llm_config: LLMConfig = LLMConfig()
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

class SemanticCascadeProcessor:
    """
    Main processor implementing the Semantic Cascade Processing algorithm.
    
    The SCP algorithm processes user input through multiple layers of analysis:
    1. Initial Understanding: Basic concept extraction
    2. Semantic Analysis: Relationship discovery
    3. Context Integration: Broader implications
    4. Response Synthesis: Final coherent response
    """
    
    def __init__(self, config: SCPConfig = SCPConfig()):
        """Initialize Semantic Cascade Processing with enhanced semantic analysis."""
        self.config = config
        self.conversation = Conversation()
        self.knowledge_base = KnowledgeBase()
        self.last_query = ""
        # Add semantic processing tools with improved configuration
        self.vectorizer = TfidfVectorizer(
            stop_words=None,  
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,  # Include terms that appear at least once
            strip_accents='unicode',
            token_pattern=r'(?u)\b\w+\b'  # Match any word character
        )
        ensure_nltk_resources()
        self.is_vectorizer_fitted = False
        self.corpus_texts = []
        
    def _extract_key_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts using enhanced NLP techniques.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of key concepts ordered by importance
        """
        if not text or not text.strip():
            return []
            
        # Tokenize and filter with improved preprocessing
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Enhanced filtering criteria with correct punctuation check
        filtered_tokens = [
            t for t in tokens 
            if (
                len(t) > 1 and  # Minimum length
                not t.isnumeric() and  # Not purely numeric
                not t in stop_words and  # Not a stopword
                not all(char in string.punctuation for char in t)  # Correct punctuation check
            )
        ]
        
        if not filtered_tokens:
            return tokens[:self.config.min_keywords]
        
        # Calculate TF-IDF with improved weighting
        document = ' '.join(filtered_tokens)
        try:
            # Use sublinear TF scaling for better weight distribution
            tfidf_matrix = self.vectorizer.fit_transform([document])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get weights considering document frequency
            dense = tfidf_matrix.todense()
            weights = np.asarray(dense)[0]
            
            # Apply sigmoid normalization for more balanced weights
            normalized_weights = 1 / (1 + np.exp(-weights))
            
            # Combine terms with weights
            weighted_terms = list(zip(normalized_weights, feature_names))
            
            # Sort by weight
            weighted_terms.sort(reverse=True)
            
            # Filter based on weight threshold and limits
            significant_terms = [
                (weight, term) 
                for weight, term in weighted_terms 
                if weight >= self.config.keyword_weight_threshold
            ]
            
            # Apply min/max limits while respecting weight threshold
            num_terms = max(
                self.config.min_keywords,
                min(len(significant_terms), self.config.max_keywords)
            )
            
            # Return terms with their weights for debugging
            selected_terms = significant_terms[:num_terms]
            
            if self.config.debug_mode:
                print("\nExtracted concepts:")
                for weight, term in selected_terms:
                    print(f"  • {term}: {weight:.3f}")
            
            return [term for _, term in selected_terms]
            
        except ValueError as e:
            if self.config.debug_mode:
                print(f"Error in concept extraction: {str(e)}")
            return filtered_tokens[:self.config.min_keywords]

    def _calculate_semantic_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """Calculate semantic similarity with improved metrics."""
        # Ensure vectorizer is fitted before calculating similarity
        self._fit_vectorizer()
        
        try:
            # Handle empty or whitespace-only inputs
            if not text1.strip() or not text2.strip():
                return 0.0
                
            # Get TF-IDF vectors
            vectors = self.vectorizer.transform([text1, text2])
            
            # Calculate cosine similarity
            cos_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Calculate Jaccard similarity for word overlap
            words1 = set(word_tokenize(text1.lower()))
            words2 = set(word_tokenize(text2.lower()))
            jaccard = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
            
            # Combine similarities with weighted average
            combined_sim = (0.7 * cos_sim) + (0.3 * jaccard)
            
            # Add debug logging
            if self.config.debug_mode:
                print(f"\nSimilarity Calculation:")
                print(f"- Cosine similarity: {cos_sim:.3f}")
                print(f"- Jaccard similarity: {jaccard:.3f}")
                print(f"- Combined similarity: {combined_sim:.3f}")
            
            # Normalize to [0,1] range
            return max(0.0, min(1.0, combined_sim))
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"Similarity calculation error: {str(e)}")
                print(f"Text1 length: {len(text1)}")
                print(f"Text2 length: {len(text2)}")
            return 0.0

    def process_semantic_cascade(self, user_input: str) -> Dict[str, Any]:
        """Process user input through semantic cascade layers."""
        self.last_query = user_input
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
        
        if self.config.debug_mode:
            print(f"Corpus size after: {len(self.corpus_texts)}")
            print(f"Vectorizer fitted: {self.is_vectorizer_fitted}")
        
        # Refit vectorizer with updated corpus
        self._fit_vectorizer()
        
        if self.config.debug_mode:
            print_colored("\n1️⃣ Layer 1: Concept Extraction", 'blue')
        initial_understanding = self._process_layer(
            "initial_understanding",
            user_input,
            context,
            """Identify ONLY the fundamental concepts and questions. 
            Keep this layer focused on raw concept identification."""
        )
        
        if self.config.debug_mode:
            print_colored(f"➡️ {initial_understanding}\n", 'green')
            print_colored("2️⃣ Layer 2: Semantic Analysis", 'blue')
        
        # Layer 2: Semantic Analysis
        relationships = self._process_layer(
            "relationship_analysis",
            user_input,
            initial_understanding,
            """Discover NEW connections between the identified concepts. 
            Focus ONLY on relationships not mentioned in the previous layer."""
        )
        
        if self.config.debug_mode:
            print_colored(f"➡️ {relationships}\n", 'green')
            print_colored("3️⃣ Layer 3: Context Integration", 'blue')
        
        # Layer 3: Contextual Integration
        context_integration = self._process_layer(
            "contextual_integration",
            user_input,
            relationships,
            """Add BROADER context and implications not yet discussed. 
            Do not repeat specific examples unless adding new perspective."""
        )
        
        if self.config.debug_mode:
            print_colored(f"➡️ {context_integration}\n", 'green')
            print_colored("4️⃣ Layer 4: Response Synthesis", 'blue')
        
        # Layer 4: Synthesis and Response Formation
        final_response = self._process_layer(
            "synthesis",
            user_input,
            context_integration,
            """Create a cohesive response that builds upon all previous 
            layers without repeating their specific details. Focus on 
            synthesizing the insights into a clear, actionable conclusion."""
        )
        
        if self.config.debug_mode:
            print_colored(f"➡️ {final_response}\n", 'green')
        
        return {
            'initial_understanding': initial_understanding,
            'relationships': relationships,
            'context_integration': context_integration,
            'final_response': final_response
        }
    
    def _process_layer(
        self, 
        layer_name: str, 
        user_input: str,
        previous_layer_output: str,
        layer_instruction: str
    ) -> str:
        """Process a single layer with enhanced semantic analysis."""
        # Extract concepts from both user input and previous layer
        user_concepts = self._extract_key_concepts(user_input)
        layer_concepts = self._extract_key_concepts(previous_layer_output)
        combined_concepts = list(set(user_concepts + layer_concepts))
        
        # Calculate semantic similarity between user input and previous layer
        similarity = self._calculate_semantic_similarity(
            user_input, 
            previous_layer_output
        )
        
        # Calculate novelty considering both user input and previous concepts
        all_previous_concepts = self.conversation.context.get(
            'processed_concepts', 
            set()
        )
        novelty_score = self._calculate_concept_novelty(
            combined_concepts,
            all_previous_concepts
        )
        
        # Update processing based on layer position
        layer_weights = {
            "initial_understanding": {
                "concept_weight": 0.9,
                "practical_weight": 0.3,
                "knowledge_weight": 0.5
            },
            "relationship_analysis": {
                "concept_weight": 0.8,
                "practical_weight": 0.4,
                "knowledge_weight": 0.7
            },
            "contextual_integration": {
                "concept_weight": 0.7,
                "practical_weight": 0.6,
                "knowledge_weight": 0.8
            },
            "synthesis": {
                "concept_weight": 0.6,
                "practical_weight": 0.8,
                "knowledge_weight": 0.7
            }
        }
        
        weights = layer_weights.get(layer_name, {
            "concept_weight": 0.5,
            "practical_weight": 0.5
        })
        
        # Integrate knowledge base content
        knowledge = self._integrate_knowledge_base(layer_name, layer_concepts)
        
        # Generate enhanced prompt with knowledge
        system_prompt = self._generate_enhanced_prompt(
            layer_name,
            user_input,
            previous_layer_output,
            weights,
            novelty_score,
            knowledge
        )
        
        # Store processed concepts
        self.conversation.context['processed_concepts'] = (
            all_previous_concepts.union(set(layer_concepts))
        )
        
        return self._call_llm(
            system_prompt, 
            user_input,
            self._calculate_dynamic_temperature(novelty_score, layer_name)
        )

    def process_interaction(self, user_input: str) -> Dict[str, Any]:
        """Process a complete interaction with semantic cascade."""
        # Add user input to conversation history
        self.conversation.add_message("user", user_input)
        
        # Process through semantic cascade
        cascade_results = self.process_semantic_cascade(user_input)
        
        # Add final response to conversation history
        self.conversation.add_message(
            "assistant", 
            cascade_results['final_response']
        )
        
        return cascade_results

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        """Make an API call to the LLM."""
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.config.llm_config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": self.config.llm_config.max_tokens,
            "stream": self.config.llm_config.stream,
            "top_p": self.config.llm_config.top_p,
            "frequency_penalty": self.config.llm_config.frequency_penalty,
            "presence_penalty": self.config.llm_config.presence_penalty,
            "stop": self.config.llm_config.stop_sequences,
            "repeat_penalty": self.config.llm_config.repeat_penalty
        }
        
        # Add seed if specified
        if self.config.llm_config.seed is not None:
            data["seed"] = self.config.llm_config.seed

        try:
            response = requests.post(
                self.config.llm_config.url,
                headers=headers,
                data=json.dumps(data),
                stream=True
            )
            response.raise_for_status()

            if self.config.llm_config.stream:
                # Handle streaming response
                client = sseclient.SSEClient(response)
                full_response = ""
                if self.config.debug_mode:
                    print_colored("\nThinking: ", 'blue', end='')
                for event in client.events():
                    try:
                        chunk = json.loads(event.data)
                        if 'choices' in chunk and chunk['choices']:
                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                            full_response += content
                            if self.config.debug_mode:
                                print(content, end='', flush=True)
                    except json.JSONDecodeError:
                        continue
                if self.config.debug_mode:
                    print("\n")
                return full_response
            else:
                # Handle non-streaming response
                return response.json()['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM API call failed: {str(e)}")

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

    def _calculate_dynamic_temperature(
        self, 
        novelty_score: float,
        layer_name: str = None
    ) -> float:
        """Calculate temperature with improved dynamic adjustment."""
        base_temp = self.config.llm_config.temperature
        
        # Layer-specific temperature modifiers
        layer_modifiers = {
            "initial_understanding": 0.8,
            "relationship_analysis": 1.0,
            "contextual_integration": 1.2,
            "synthesis": 0.9
        }
        
        # Apply layer modifier if specified
        modifier = layer_modifiers.get(layer_name, 1.0)
        
        # Sigmoid function for smooth temperature scaling
        k = 12  # Steepness
        x0 = 0.5  # Midpoint
        scaled_novelty = 1 / (1 + np.exp(-k * (novelty_score - x0)))
        
        # Calculate final temperature
        min_temp = 0.1
        max_temp = 1.0
        adjusted_temp = (
            base_temp * (1 - scaled_novelty) + 
            max_temp * scaled_novelty
        ) * modifier
        
        return max(min_temp, min(max_temp, adjusted_temp))

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
        Integrate knowledge base with improved relevance scoring.
        
        Args:
            layer_name: Current processing layer name
            concepts: List of extracted concepts
            
        Returns:
            Dict containing relevant concepts, examples, and context
        """
        knowledge = {
            'concepts': [],
            'examples': [],
            'context': []
        }
        
        if self.config.debug_mode:
            print(f"Knowledge base integration enabled: {self.config.use_external_knowledge}")
        
        if not self.config.use_external_knowledge:
            if self.config.debug_mode:
                print("Skipping knowledge base integration - disabled in config")
            return knowledge
            
        try:
            # Process concepts
            concept_docs = self.knowledge_base.get_relevant_documents('concepts')
            if concept_docs:
                concepts_data = []
                for doc in concept_docs:
                    try:
                        if isinstance(doc, str):
                            if doc.strip():  # Check if doc is not empty
                                parsed_doc = json.loads(doc)
                                concepts_data.append(parsed_doc)
                        elif isinstance(doc, dict):
                            concepts_data.append(doc)
                    except json.JSONDecodeError as e:
                        if self.config.debug_mode:
                            print(f"Failed to parse concept doc: {str(e)}")
                        continue
                
                scored_concepts = []
                for concept in concepts_data:
                    try:
                        concept_text = (
                            json.dumps(concept) if isinstance(concept, dict)
                            else str(concept)
                        )
                        max_similarity = max(
                            self._calculate_semantic_similarity(
                                c.lower(), 
                                concept_text.lower()
                            ) for c in concepts
                        )
                        if max_similarity > self.config.similarity_threshold:
                            scored_concepts.append((max_similarity, concept))
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"Error processing concept: {str(e)}")
                        continue
                
                scored_concepts.sort(reverse=True)
                knowledge['concepts'] = [c for _, c in scored_concepts[:3]]
            
            # Process examples
            example_docs = self.knowledge_base.get_relevant_documents('examples')
            if example_docs:
                examples_data = []
                for doc in example_docs:
                    try:
                        if isinstance(doc, str):
                            if doc.strip():  # Check if doc is not empty
                                try:
                                    parsed_doc = json.loads(doc)
                                    if isinstance(parsed_doc, list):
                                        examples_data.extend(parsed_doc)
                                    else:
                                        examples_data.append(parsed_doc)
                                except json.JSONDecodeError:
                                    # Treat as plain text if not valid JSON
                                    examples_data.append(doc)
                        elif isinstance(doc, (dict, list)):
                            if isinstance(doc, list):
                                examples_data.extend(doc)
                            else:
                                examples_data.append(doc)
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"Failed to process example doc: {str(e)}")
                        continue
                
                scored_examples = []
                for example in examples_data:
                    try:
                        example_text = (
                            json.dumps(example) if isinstance(example, (dict, list))
                            else str(example)
                        )
                        
                        concept_similarities = [
                            self._calculate_semantic_similarity(
                                c.lower(), 
                                example_text.lower()
                            ) for c in concepts
                        ]
                        
                        layer_similarity = self._calculate_semantic_similarity(
                            layer_name.lower(),
                            example_text.lower()
                        )
                        
                        combined_score = (
                            0.7 * max(concept_similarities) if concept_similarities else 0.0 +
                            0.3 * layer_similarity
                        )
                        
                        if combined_score > self.config.similarity_threshold:
                            scored_examples.append((combined_score, example))
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"Error processing example: {str(e)}")
                        continue
                
                scored_examples.sort(reverse=True)
                knowledge['examples'] = [e for _, e in scored_examples[:2]]
            
            # Process broader context
            context_docs = self.knowledge_base.get_relevant_documents('context')
            if context_docs:
                context_data = []
                for doc in context_docs:
                    try:
                        if isinstance(doc, str):
                            if doc.strip():  # Check if doc is not empty
                                try:
                                    parsed_doc = json.loads(doc)
                                    if isinstance(parsed_doc, list):
                                        context_data.extend(parsed_doc)
                                    else:
                                        context_data.append(parsed_doc)
                                except json.JSONDecodeError:
                                    # Treat as plain text if not valid JSON
                                    context_data.append(doc)
                        elif isinstance(doc, (dict, list)):
                            if isinstance(doc, list):
                                context_data.extend(doc)
                            else:
                                context_data.append(doc)
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"Failed to process context doc: {str(e)}")
                        continue
                
                scored_context = []
                for context_item in context_data:
                    try:
                        context_text = (
                            json.dumps(context_item) if isinstance(context_item, (dict, list))
                            else str(context_item)
                        )
                        
                        layer_weights = {
                            "initial_understanding": 0.3,
                            "relationship_analysis": 0.5,
                            "contextual_integration": 0.8,
                            "synthesis": 0.6
                        }
                        layer_weight = layer_weights.get(layer_name, 0.5)
                        
                        concept_similarities = [
                            self._calculate_semantic_similarity(
                                c.lower(),
                                context_text.lower()
                            ) for c in concepts
                        ]
                        
                        layer_similarity = self._calculate_semantic_similarity(
                            layer_name.lower(),
                            context_text.lower()
                        )
                        
                        combined_score = (
                            (1 - layer_weight) * max(concept_similarities) if concept_similarities else 0.0 +
                            layer_weight * layer_similarity
                        )
                        
                        if combined_score > self.config.similarity_threshold:
                            scored_context.append((combined_score, context_item))
                    except Exception as e:
                        if self.config.debug_mode:
                            print(f"Error processing context item: {str(e)}")
                        continue
                
                scored_context.sort(reverse=True)
                knowledge['context'] = [c for _, c in scored_context[:2]]
            
            # Add metadata about knowledge integration
            knowledge['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'layer': layer_name,
                'concepts_found': len(knowledge['concepts']),
                'examples_found': len(knowledge['examples']),
                'context_found': len(knowledge['context']),
                'threshold_used': self.config.similarity_threshold
            }
            
        except Exception as e:
            if self.config.debug_mode:
                print(f"Knowledge base integration error: {str(e)}")
                print(f"Layer: {layer_name}")
                print(f"Concepts: {concepts}")
            knowledge['error'] = str(e)
        
        return knowledge

    def _fit_vectorizer(self) -> None:
        """Fit vectorizer on accumulated corpus texts."""
        try:
            if not self.is_vectorizer_fitted and self.corpus_texts:
                if self.config.debug_mode:
                    print(f"Fitting vectorizer on {len(self.corpus_texts)} documents")
                self.vectorizer.fit(self.corpus_texts)
                self.is_vectorizer_fitted = True
                if self.config.debug_mode:
                    print("Vectorizer fitted successfully")
        except Exception as e:
            if self.config.debug_mode:
                print(f"Error fitting vectorizer: {str(e)}")

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




