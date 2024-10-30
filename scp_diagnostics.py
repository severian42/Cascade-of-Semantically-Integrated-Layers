"""
SCP Diagnostics Module

Provides real-time visibility into the Semantic Cascade Processing layers,
semantic calculations, and decision-making process.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn
from rich.syntax import Syntax
import json
import time

@dataclass
class SemanticMetrics:
    """Stores detailed semantic analysis metrics"""
    similarity_scores: Dict[str, float]
    novelty_scores: Dict[str, float]
    concept_weights: Dict[str, float]
    temperature_adjustments: Dict[str, float]
    layer_confidences: Dict[str, float]

@dataclass
class LayerDiagnostics:
    """Captures processing details for each layer"""
    concepts_extracted: List[str]
    semantic_metrics: SemanticMetrics
    knowledge_integration: Dict[str, Any]
    processing_time: float
    layer_output: str

class SCPDiagnostics:
    def __init__(self, scp_processor):
        """Initialize diagnostics for an SCP processor"""
        self.scp = scp_processor
        self.console = Console()
        self.layer_history: Dict[str, List[LayerDiagnostics]] = {}
        self._current_metrics = {}
        
        # Store original methods
        self._original_calculate_similarity = self.scp._calculate_semantic_similarity
        self._original_extract_concepts = self.scp._extract_key_concepts
        
        # Override with instrumented versions
        self.scp._calculate_semantic_similarity = self._instrument_similarity
        self.scp._extract_key_concepts = self._instrument_concept_extraction

    def _instrument_similarity(self, text1: str, text2: str) -> float:
        """Instrumented version of similarity calculation"""
        similarity = self._original_calculate_similarity(text1, text2)
        vectors = self.scp.vectorizer.transform([text1, text2])
        
        # Calculate detailed metrics
        vector_magnitudes = np.sqrt(np.sum(vectors.toarray() ** 2, axis=1))
        term_contributions = vectors.multiply(vectors).sum(axis=1).A1
        
        self._current_metrics = {
            'raw_similarity': similarity,
            'vector_magnitudes': vector_magnitudes.tolist(),
            'term_contributions': term_contributions.tolist()
        }
        
        return similarity

    def _instrument_concept_extraction(self, text: str) -> List[str]:
        """
        Instrumented version of concept extraction with detailed metrics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted concepts
        """
        start_time = time.time()
        
        # Call original method
        concepts = self._original_extract_concepts(text)
        
        # Calculate additional metrics
        processing_time = time.time() - start_time
        
        # Store metrics for current extraction
        self._current_metrics.update({
            'concept_extraction_time': processing_time,
            'num_concepts': len(concepts),
            'concept_lengths': [len(c) for c in concepts],
            'extraction_confidence': 1.0 if concepts else 0.0
        })
        
        return concepts

    def visualize_layer_processing(self, layer_name: str, diagnostics: LayerDiagnostics):
        """Create rich visualization of layer processing"""
        self.console.print(f"\n[bold blue]Layer: {layer_name}[/bold blue]")
        
        # Concepts Table
        concept_table = Table(title="Extracted Concepts")
        concept_table.add_column("Concept")
        concept_table.add_column("Weight")
        concept_table.add_column("Novelty")
        
        for idx, concept in enumerate(diagnostics.concepts_extracted):
            # Handle both dictionary and list formats for weights
            if isinstance(diagnostics.semantic_metrics.concept_weights, dict):
                weight = diagnostics.semantic_metrics.concept_weights.get(concept, 0)
            else:
                weight = diagnostics.semantic_metrics.concept_weights[idx] if idx < len(
                    diagnostics.semantic_metrics.concept_weights
                ) else 0
                
            novelty = diagnostics.semantic_metrics.novelty_scores.get(
                concept, 
                0
            ) if isinstance(
                diagnostics.semantic_metrics.novelty_scores, 
                dict
            ) else 0
            
            concept_table.add_row(
                concept,
                f"{weight:.3f}",
                f"{novelty:.3f}"
            )
        
        self.console.print(concept_table)
        
        # Semantic Metrics
        metrics_table = Table(title="Semantic Metrics")
        metrics_table.add_column("Metric")
        metrics_table.add_column("Value")
        
        if isinstance(diagnostics.semantic_metrics.similarity_scores, dict):
            for metric, value in diagnostics.semantic_metrics.similarity_scores.items():
                metrics_table.add_row(metric, f"{value:.3f}")
        else:
            metrics_table.add_row(
                "Similarity", 
                f"{diagnostics.semantic_metrics.similarity_scores:.3f}"
            )
        
        self.console.print(metrics_table)

    def analyze_corpus_state(self) -> Dict[str, Any]:
        """Analyze current state of the semantic corpus"""
        # Initialize default empty state
        empty_state = {
            "corpus_size": 0,
            "vocabulary_size": 0,
            "avg_document_length": 0.0,
            "unique_terms_ratio": 0.0,
            "term_frequency_distribution": [],
            "status": "empty"
        }
        
        if not hasattr(self.scp, 'corpus_texts') or not self.scp.corpus_texts:
            return empty_state
            
        try:
            vectorizer = self.scp.vectorizer
            corpus_vectors = vectorizer.transform(self.scp.corpus_texts)
            
            # Calculate corpus statistics
            document_lengths = np.sum(corpus_vectors.toarray() > 0, axis=1)
            term_frequencies = np.sum(corpus_vectors.toarray(), axis=0)
            vocabulary_size = len(vectorizer.vocabulary_)
            
            return {
                "corpus_size": len(self.scp.corpus_texts),
                "vocabulary_size": vocabulary_size,
                "avg_document_length": float(np.mean(document_lengths)),
                "unique_terms_ratio": vocabulary_size / sum(document_lengths),
                "term_frequency_distribution": term_frequencies.tolist(),
                "status": "active"
            }
        except Exception as e:
            print(f"Warning: Error analyzing corpus state: {str(e)}")
            return empty_state

    def export_layer_metrics(self, format: str = 'json') -> str:
        """Export layer processing metrics in specified format"""
        metrics = {
            layer: {
                "processing_time": diag.processing_time,
                "concepts_found": len(diag.concepts_extracted),
                "similarity_scores": diag.semantic_metrics.similarity_scores,
                "novelty_scores": diag.semantic_metrics.novelty_scores,
                "temperature": diag.semantic_metrics.temperature_adjustments
            }
            for layer, diag in self.layer_history.items()
        }
        
        if format == 'json':
            return json.dumps(metrics, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def monitor_interaction(self, user_input: str):
        """Real-time monitoring of SCP processing"""
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Processing", total=4)
            
            # Monitor each layer
            for layer in ["initial_understanding", "relationship_analysis", 
                         "contextual_integration", "synthesis"]:
                progress.update(task, description=f"Processing {layer}")
                
                # Capture layer processing
                start_time = time.time()
                layer_output = self.scp._process_layer(
                    layer,
                    user_input,
                    "",
                    f"Processing {layer}"
                )
                processing_time = time.time() - start_time
                
                # Store diagnostics
                self.layer_history[layer] = LayerDiagnostics(
                    concepts_extracted=self.scp._extract_key_concepts(layer_output),
                    semantic_metrics=self._capture_current_metrics(),
                    knowledge_integration=self.scp._integrate_knowledge_base(
                        layer, 
                        self.scp._extract_key_concepts(layer_output)
                    ),
                    processing_time=processing_time,
                    layer_output=layer_output
                )
                
                # Visualize current layer
                self.visualize_layer_processing(layer, self.layer_history[layer])
                
                progress.advance(task)

    def _capture_current_metrics(self) -> SemanticMetrics:
        """Capture current processing metrics"""
        # Convert list to dictionary for term contributions
        term_contributions = getattr(self, '_current_metrics', {}).get(
            'term_contributions', []
        )
        concept_weights = {}
        if isinstance(term_contributions, list):
            for idx, weight in enumerate(term_contributions):
                concept_weights[str(idx)] = weight
        
        return SemanticMetrics(
            similarity_scores=getattr(self, '_current_metrics', {}).get(
                'raw_similarity', {}
            ),
            novelty_scores=getattr(self, '_current_metrics', {}).get(
                'novelty_scores', {}
            ),
            concept_weights=concept_weights,  # Now using the converted dictionary
            temperature_adjustments=getattr(self, '_current_metrics', {}).get(
                'temperature', {}
            ),
            layer_confidences=getattr(self, '_current_metrics', {}).get(
                'confidences', {}
            )
        )

    def _generate_report_content(self) -> List[str]:
        """Generate the main content of the analysis report"""
        report = []
        
        # 1. Corpus Analysis
        corpus_state = self.analyze_corpus_state()
        report.append("## 1. Knowledge Base Status")
        report.append("```")
        report.append(f"Documents Analyzed: {corpus_state['corpus_size']}")
        report.append(f"Vocabulary Size: {corpus_state['vocabulary_size']}")
        report.append(f"Average Document Length: {corpus_state['avg_document_length']:.2f} terms")
        report.append(f"Knowledge Density: {corpus_state.get('unique_terms_ratio', 0):.3f}")
        report.append("```\n")
        
        # 2. Layer-by-Layer Analysis
        report.append("## 2. Processing Layers Analysis")
        for layer, diagnostics in self.layer_history.items():
            report.append(f"\n### {layer.title()}")
            
            # Concepts and their weights
            report.append("#### Extracted Concepts")
            report.append("```")
            for concept in diagnostics.concepts_extracted:
                weight = diagnostics.semantic_metrics.concept_weights.get(concept, 0)
                novelty = diagnostics.semantic_metrics.novelty_scores.get(concept, 0)
                report.append(f"• {concept}")
                report.append(f"  - Relevance Weight: {weight:.3f}")
                report.append(f"  - Novelty Score: {novelty:.3f}")
            report.append("```")
            
            # Semantic Analysis
            report.append("\n#### Semantic Analysis")
            report.append("```")
            report.append(f"Processing Time: {diagnostics.processing_time:.3f}s")
            report.append(f"Similarity Score: {diagnostics.semantic_metrics.similarity_scores}")
            report.append(f"Temperature: {list(diagnostics.semantic_metrics.temperature_adjustments.values())[0] if diagnostics.semantic_metrics.temperature_adjustments else 'N/A'}")
            report.append("```")
            
            # Layer Output
            report.append("\n#### Layer Understanding")
            report.append(f"```\n{diagnostics.layer_output}\n```\n")
        
        # 3. Overall Processing Metrics
        report.append("## 3. Processing Summary")
        total_time = sum(d.processing_time for d in self.layer_history.values())
        total_concepts = sum(len(d.concepts_extracted) for d in self.layer_history.values())
        report.append("```")
        report.append(f"Total Processing Time: {total_time:.3f}s")
        report.append(f"Total Concepts Analyzed: {total_concepts}")
        report.append(f"Average Processing Time per Layer: {total_time/len(self.layer_history):.3f}s")
        report.append("```")
        
        return report

    def generate_detailed_report(self) -> str:
        """Generate and save comprehensive analysis of semantic processing"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report = []
        
        # Add metadata header
        report.append("---")
        report.append("title: SCP Analysis Report")
        report.append(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"query: {getattr(self.scp, 'last_query', 'No query recorded')}")
        report.append("---\n")
        
        # Add existing report content
        report.extend(self._generate_report_content())
        
        # Save report
        report_text = "\n".join(report)
        filename = f"scp_analysis_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        return report_text
