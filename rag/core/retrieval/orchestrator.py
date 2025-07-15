
"""
Vuln_RAG Retrieval Orchestrator

High-level coordinator that turns a pre-processed query into a complete Large-Language-Model context. Pipeline:
1. Pre-processing (performed upstream)
2. Hybrid search on the three knowledge bases (KB1 Whoosh, KB2 FAISS CPG, KB3 FAISS Code)
3. Reciprocal-Rank-Fusion (RRF)
4. Document assembly and enrichment
5. Structured context construction for vulnerability detection and (optionally) patch generation
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .fusion_controller import Vuln_RAGRetrievalController, FusionCandidate, create_default_controller
from .document_assembler import DocumentAssembler, EnrichedDocument
from ..generation.context_builder import ContextBuilder

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Complete result of a Vuln_RAG retrieval run"""
    # Input
    original_code: str
    query_data: Dict[str, Any]
    
    # Fusion results
    fusion_candidates: List[FusionCandidate]
    enriched_documents: List[EnrichedDocument]
    
    # LLM context
    detection_context: str
    patch_context: Optional[str] = None
    
    # Performance metadata
    total_time_ms: float = 0.0
    search_time_ms: float = 0.0
    assembly_time_ms: float = 0.0
    context_time_ms: float = 0.0
    
    # Statistics
    stats: Dict[str, Any] = None

class Vuln_RAGRetrievalOrchestrator:
    """
    Main orchestrator for Vuln_RAG retrieval
    
    Full pipeline:
    1. Code pre-processing → query_data
    2. Hybrid search (KB1 + KB2 + KB3)
    3. RRF fusion
    4. Document assembly
    5. LLM context construction
    """
    
    def __init__(self,
                 controller: Optional[Vuln_RAGRetrievalController] = None,
                 assembler: Optional[DocumentAssembler] = None,
                 top_k_search: int = 10,
                 top_k_context: int = 5):
        """
        Args:
            controller: Search controller (None = default)
            assembler: Document assembler (None = default)
            top_k_search: Number of results per engine
            top_k_context: Number of documents kept in final context
        """
        self.controller = controller or create_default_controller()
        self.assembler = assembler or DocumentAssembler()
        self.top_k_search = top_k_search
        self.top_k_context = top_k_context
        
        logger.info("Vuln_RAG Retrieval Orchestrator initialised")
    
    def retrieve_context(self,
                        original_code: str,
                        query_data: Dict[str, Any],
                        include_patch_context: bool = False) -> RetrievalResult:
        """
        Full contextual retrieval pipeline
        
        Args:
            original_code: Source code to analyse
            query_data: Pre-processed query data {
                "kb1_purpose": str,
                "kb1_function": str, 
                "kb2_vector": List[float],
                "kb3_code": str
            }
            include_patch_context: Also generate patch context
            
        Returns:
            RetrievalResult with full LLM context
        """
        start_time = time.time()
        
        logger.info(f"Starting contextual retrieval for {len(original_code)} characters of code")
        
        # 1. Hybrid search + RRF fusion
        search_start = time.time()
        fusion_candidates = self.controller.search_from_preprocessed_query(
            query_data=query_data,
            top_k=self.top_k_search
        )
        search_time_ms = (time.time() - search_start) * 1000
        
        logger.info(f"RRF fusion produced {len(fusion_candidates)} candidates in {search_time_ms:.1f} ms")
        
        # 2. Document assembly
        assembly_start = time.time()
        enriched_documents = self.assembler.assemble_documents(
            candidates=fusion_candidates,
            top_k=self.top_k_context
        )
        assembly_time_ms = (time.time() - assembly_start) * 1000
        
        logger.info(f"Assembly: {len(enriched_documents)} documents enriched in {assembly_time_ms:.1f} ms")
        
        # 3. Detection context construction
        context_start = time.time()
        detection_context = ContextBuilder.build_detection_context(
            original_code=original_code,
            enriched_docs=enriched_documents,
            max_context_length=4000
        )
        
        # 4. Construction du contexte de patch (optionnel)
        patch_context = None
        if include_patch_context:
            # Simulate detection result to build patch context
            dummy_detection = {
                "is_vulnerable": True,
                "confidence": 0.8,
                "cwe": enriched_documents[0].cwe if enriched_documents else "CWE-Unknown",
                "explanation": "Vulnerability detected by RAG system"
            }
            patch_context = ContextBuilder.build_patch_context(
                original_code=original_code,
                detection_result=dummy_detection,
                enriched_docs=enriched_documents
            )
        
        context_time_ms = (time.time() - context_start) * 1000
        total_time_ms = (time.time() - start_time) * 1000
        
        # 5. Statistiques
        stats = self._compute_stats(fusion_candidates, enriched_documents)
        
        result = RetrievalResult(
            original_code=original_code,
            query_data=query_data,
            fusion_candidates=fusion_candidates,
            enriched_documents=enriched_documents,
            detection_context=detection_context,
            patch_context=patch_context,
            total_time_ms=total_time_ms,
            search_time_ms=search_time_ms,
            assembly_time_ms=assembly_time_ms,
            context_time_ms=context_time_ms,
            stats=stats
        )
        
        logger.info(f"Retrieval completed in {total_time_ms:.1f} ms")
        logger.info(f"Detection context length: {len(detection_context)} characters")
        if patch_context:
            logger.info(f"Patch context length: {len(patch_context)} characters")
            
        return result
    
    def _compute_stats(self, 
                      candidates: List[FusionCandidate], 
                      documents: List[EnrichedDocument]) -> Dict[str, Any]:
        """Compute retrieval statistics"""
        if not candidates:
            return {"candidates": 0, "documents": 0, "coverage": {}}
            
        # Source distribution
        kb1_count = sum(1 for c in candidates if c.kb1_rank is not None)
        kb2_count = sum(1 for c in candidates if c.kb2_rank is not None) 
        kb3_count = sum(1 for c in candidates if c.kb3_rank is not None)
        
        # CWE distribution
        cwe_distribution = {}
        for doc in documents:
            cwe = doc.cwe or "Unknown"
            cwe_distribution[cwe] = cwe_distribution.get(cwe, 0) + 1
            
        # Scores
        scores = [c.final_score for c in candidates[:10]]
        
        return {
            "candidates": len(candidates),
            "documents": len(documents),
            "coverage": {
                "kb1": kb1_count,
                "kb2": kb2_count, 
                "kb3": kb3_count
            },
            "cwe_distribution": cwe_distribution,
            "score_stats": {
                "max": max(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "avg": sum(scores) / len(scores) if scores else 0
            }
        }
    
    def quick_search(self,
                    code: str,
                    purpose: str = "",
                    function: str = "") -> RetrievalResult:
        """
        Quick search with simplified parameters
        
        Args:
            code: Source code to analyse
            purpose: Purpose description (optional)
            function: Function description (optional)
            
        Returns:
            RetrievalResult with detection context
        """
        # Build simplified query
        query_data = {
            "kb1_purpose": purpose,
            "kb1_function": function,
            "kb2_vector": None,  # Pas de vecteur CPG
            "kb3_code": code
        }
        
        return self.retrieve_context(
            original_code=code,
            query_data=query_data,
            include_patch_context=False
        )

class RetrievalResultAnalyzer:
    """Analyzer for Vuln_RAG retrieval results"""
    
    @staticmethod
    def print_summary(result: RetrievalResult, detailed: bool = False):
        """Print a summary of the retrieval result"""
        print("=" * 60)
        print("Vuln_RAG RETRIEVAL SUMMARY")
        print("=" * 60)
        
        # Timing
        print(f"⏱️  Total Time: {result.total_time_ms:.1f}ms")
        print(f"   ├─ Search: {result.search_time_ms:.1f}ms")
        print(f"   ├─ Assembly: {result.assembly_time_ms:.1f}ms")
        print(f"   └─ Context: {result.context_time_ms:.1f}ms")
        
        # Results
        stats = result.stats or {}
        print(f"\n📊 Results:")
        print(f"   ├─ Fusion Candidates: {stats.get('candidates', 0)}")
        print(f"   ├─ Enriched Documents: {stats.get('documents', 0)}")
        print(f"   └─ Context Length: {len(result.detection_context)} chars")
        
        # Coverage
        coverage = stats.get('coverage', {})
        print(f"\n🎯 KB Coverage:")
        print(f"   ├─ KB1 (Whoosh): {coverage.get('kb1', 0)} hits")
        print(f"   ├─ KB2 (FAISS CPG): {coverage.get('kb2', 0)} hits") 
        print(f"   └─ KB3 (FAISS Code): {coverage.get('kb3', 0)} hits")
        
        # Top candidates
        if result.fusion_candidates:
            print(f"\n🏆 Top Candidates:")
            for i, candidate in enumerate(result.fusion_candidates[:3], 1):
                print(f"   {i}. {candidate.key}")
                print(f"      Score: {candidate.final_score:.3f}")
                print(f"      Sources: KB1({candidate.kb1_rank}), KB2({candidate.kb2_rank}), KB3({candidate.kb3_rank})")
        
        # CWE distribution
        cwe_dist = stats.get('cwe_distribution', {})
        if cwe_dist:
            print(f"\n🔍 CWE Distribution:")
            for cwe, count in sorted(cwe_dist.items()):
                print(f"   ├─ {cwe}: {count}")
        
        # Details if requested
        if detailed and result.enriched_documents:
            print(f"\n📋 Document Details:")
            for i, doc in enumerate(result.enriched_documents[:3], 1):
                print(f"\n   Document {i}: {doc.key}")
                print(f"   ├─ Purpose: {doc.gpt_purpose[:100]}...")
                print(f"   ├─ Function: {doc.gpt_function[:100]}...")
                if doc.dangerous_functions:
                    print(f"   └─ Dangerous: {', '.join(doc.dangerous_functions[:3])}")
        
        print("=" * 60)
    
    @staticmethod
    def export_context(result: RetrievalResult, output_path: str):
        """Export the context to a file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Vuln_RAG Detection Context\n\n")
            f.write(result.detection_context)
            
            if result.patch_context:
                f.write("\n\n" + "="*60 + "\n")
                f.write("# Vuln_RAG Patch Context\n\n")
                f.write(result.patch_context)
                
        logger.info(f"Context exported to {output_file}")
   