# core/retrieval/fusion_controller.py
"""
Reciprocal Rank Fusion (RRF) Controller for Vuln_RAG -
Combines the results of the 3 search engines: KB1 (Whoosh), KB2 (FAISS CPG), KB3 (FAISS Code)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result individual from a search engine"""
    key: str
    score: float
    rank: int
    source: str  # "kb1", "kb2", "kb3"
    metadata: Dict[str, Any] = None

@dataclass
class FusionCandidate:
    """Candidate after RRF fusion with complete provenance"""
    key: str
    final_score: float
    
    # Scores and ranks individually
    kb1_rank: Optional[int] = None
    kb2_rank: Optional[int] = None
    kb3_rank: Optional[int] = None
    kb1_score: Optional[float] = None
    kb2_score: Optional[float] = None
    kb3_score: Optional[float] = None
    
    # Performance metadata
    search_time_ms: float = 0.0
    total_sources: int = 0

class ReciprocalRankFusion:
    """
    Implémentation du Reciprocal Rank Fusion (RRF) pour 3 sources
    
    RRF Score = Σ(weight_i * 1 / (k + rank_i)) pour chaque source i
    où k est un paramètre de lissage (typiquement 60)
    """
    
    def __init__(self, k: int = 60, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            k: Paramètre de lissage RRF (60 par défaut selon littérature)
            weights: Poids optionnels par source {"kb1": 0.3, "kb2": 0.4, "kb3": 0.3}
        """
        self.k = k
        self.weights = weights or {"kb1": 1.0, "kb2": 1.0, "kb3": 1.0}
        
        # Normaliser les poids
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        logger.info(f"RRF initialisé: k={k}, weights={self.weights}")
        
    def fuse(self, 
             kb1_results: List[Dict], 
             kb2_results: List[Dict], 
             kb3_results: List[Dict]) -> List[FusionCandidate]:
        """
        Fusion RRF des résultats des 3 search engines
        
        Args:
            kb1_results: Results from Whoosh [{"key": str, "score": float}, ...]
            kb2_results: Results from FAISS CPG [{"key": str, "score": float}, ...]  
            kb3_results: Results from FAISS Code [{"key": str, "score": float}, ...]
            
        Returns:
            List of FusionCandidate sorted by RRF score descending
        """
        # Convertir en SearchResult avec rangs
        search_results = []
        
        # KB1 Results
        for rank, result in enumerate(kb1_results, 1):
            search_results.append(SearchResult(
                key=str(result["key"]),  # Assurer que key est string
                score=float(result.get("score", 0.0)),
                rank=rank,
                source="kb1",
                metadata=result
            ))
            
        # KB2 Results
        for rank, result in enumerate(kb2_results, 1):
            search_results.append(SearchResult(
                key=str(result["key"]),
                score=float(result.get("score", 0.0)),
                rank=rank,
                source="kb2",
                metadata=result
            ))
            
        # KB3 Results
        for rank, result in enumerate(kb3_results, 1):
            search_results.append(SearchResult(
                key=str(result["key"]),
                score=float(result.get("score", 0.0)),
                rank=rank,
                source="kb3",
                metadata=result
            ))
        
        # Grouper par clé
        key_groups = {}
        for result in search_results:
            if result.key not in key_groups:
                key_groups[result.key] = {}
            key_groups[result.key][result.source] = result
            
        # Calculer scores RRF
        fusion_candidates = []
        
        for key, sources in key_groups.items():
            rrf_score = 0.0
            total_sources = len(sources)
            
            # Provenance data
            kb1_rank = kb1_score = kb2_rank = kb2_score = kb3_rank = kb3_score = None
            
            # Weighted RRF calculation
            for source, result in sources.items():
                weight = self.weights.get(source, 1.0)
                contribution = weight * (1.0 / (self.k + result.rank))
                rrf_score += contribution
                
                # Stocker provenance
                if source == "kb1":
                    kb1_rank, kb1_score = result.rank, result.score
                elif source == "kb2":
                    kb2_rank, kb2_score = result.rank, result.score  
                elif source == "kb3":
                    kb3_rank, kb3_score = result.rank, result.score
                    
            candidate = FusionCandidate(
                key=key,
                final_score=rrf_score,
                kb1_rank=kb1_rank, kb1_score=kb1_score,
                kb2_rank=kb2_rank, kb2_score=kb2_score, 
                kb3_rank=kb3_rank, kb3_score=kb3_score,
                total_sources=total_sources
            )
            
            fusion_candidates.append(candidate)
            
        # Sort by final score
        fusion_candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        # Log top results
        logger.info(f"Top-3: {[(c.key, f'{c.final_score:.4f}') for c in fusion_candidates[:3]]}")
        
        return fusion_candidates

class Vuln_RAGRetrievalController:
    """
    Main controller for hybrid Vuln_RAG search
    Coordinates 3 engines + RRF fusion
    """
    
    def __init__(self, 
                 kb1_index_path: str,
                 kb2_index_path: str, 
                 kb2_metadata_path: str,
                 kb3_index_path: str,
                 kb3_metadata_path: str,
                 rrf_k: int = 60,
                 rrf_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            kb1_index_path: Path to Whoosh index
            kb2_index_path: Path to FAISS KB2 index  
            kb2_metadata_path: Path to KB2 metadata
            kb3_index_path: Path to FAISS KB3 index
            kb3_metadata_path: Path to KB3 metadata
            rrf_k: RRF parameter
            rrf_weights: Poids par moteur
        """
        self.kb1_index_path = kb1_index_path
        self.kb2_index_path = kb2_index_path
        self.kb2_metadata_path = kb2_metadata_path
        self.kb3_index_path = kb3_index_path
        self.kb3_metadata_path = kb3_metadata_path
        
        # Initialiseur RRF
        self.rrf = ReciprocalRankFusion(k=rrf_k, weights=rrf_weights)
        
        # Lazy loading des searchers
        self._kb1_searcher = None
        self._kb2_searcher = None
        self._kb3_searcher = None
        
        logger.info("Vuln_RAG Retrieval Controller initialized")
        
    def _get_kb1_searcher(self):
        """Lazy load KB1 searcher"""
        if self._kb1_searcher is None:
            sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "retrieval"))
            try:
                from search_kb1 import search_kb1_
                self._kb1_searcher = search_kb1_
            except ImportError:
                logger.error("Unable to import search_kb1")
                raise
        return self._kb1_searcher
    
    def _get_kb2_searcher(self):
        """Lazy load KB2 searcher"""
        if self._kb2_searcher is None:
            sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "retrieval"))
            try:
                from search_kb2_faiss import get_kb2_structure_searcher
                self._kb2_searcher = get_kb2_structure_searcher(
                    self.kb2_index_path, self.kb2_metadata_path
                )
            except ImportError:
                logger.error("Unable to import search_kb2_faiss")
                raise
        return self._kb2_searcher
    
    def _get_kb3_searcher(self):
        """Lazy load KB3 searcher"""
        if self._kb3_searcher is None:
            sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "retrieval"))
            try:
                from search_kb3_code_faiss import get_kb3_searcher
                self._kb3_searcher = get_kb3_searcher(
                    self.kb3_index_path, self.kb3_metadata_path
                )
            except ImportError:
                logger.error("Unable to import search_kb3_code_faiss")
                raise
        return self._kb3_searcher
        
    def search_hybrid(self, 
                     kb1_purpose: str = "",
                     kb1_function: str = "", 
                     kb2_vector: Optional[List[float]] = None,
                     kb3_code: str = "",
                     top_k: int = 10) -> List[FusionCandidate]:
        """
        Hybrid search in the 3 KBs with RRF fusion
        
        Args:
            kb1_purpose: Text for KB1 search (purpose)
            kb1_function: Text for KB1 search (function)
            kb2_vector: Embedding vector for KB2
            kb3_code: Source code for KB3 search
            top_k: Number of results per engine
            
        Returns:
            List of fusionned and sorted candidates
        """
        start_time = time.time()
        
        # Parallel search in the 3 KBs
        kb1_results = []
        kb2_results = []
        kb3_results = []
        
        # KB1 - Textual search Whoosh
        if kb1_purpose or kb1_function:
            try:
                kb1_searcher = self._get_kb1_searcher()
                kb1_results = kb1_searcher(
                    purpose_text=kb1_purpose,
                    function_text=kb1_function, 
                    top_k=top_k,
                    index_dir=self.kb1_index_path
                )
                logger.info(f"KB1 Whoosh: {len(kb1_results)} résultats")
            except Exception as e:
                logger.warning(f"Erreur KB1: {e}")
                
        # KB2 - Vector search CPG  
        if kb2_vector:
            try:
                kb2_searcher = self._get_kb2_searcher()
                kb2_results, _ = kb2_searcher.search(
                    embedding_vector=kb2_vector,
                    top_k=top_k
                )
                logger.info(f"KB2 FAISS CPG: {len(kb2_results)} résultats")
            except Exception as e:
                logger.warning(f"Erreur KB2: {e}")
                
        # KB3 - Vector search Code
        if kb3_code:
            try:
                kb3_searcher = self._get_kb3_searcher()
                kb3_results, _ = kb3_searcher.search(
                    code_snippet=kb3_code,
                    top_k=top_k
                )
                logger.info(f"KB3 FAISS Code: {len(kb3_results)} résultats")
            except Exception as e:
                logger.warning(f"Erreur KB3: {e}")
                
        # RRF fusion
        fusion_candidates = self.rrf.fuse(kb1_results, kb2_results, kb3_results)
        
        # Add timing
        search_time_ms = (time.time() - start_time) * 1000
        for candidate in fusion_candidates:
            candidate.search_time_ms = search_time_ms
            
        logger.info(f"Recherche hybride terminée en {search_time_ms:.1f}ms")
        logger.info(f"Top-3 candidats: {[(c.key, f'{c.final_score:.3f}') for c in fusion_candidates[:3]]}")
        
        return fusion_candidates
    
    def search_from_preprocessed_query(self, query_data: Dict, top_k: int = 10) -> List[FusionCandidate]:
        """
        Recherche à partir d'un objet query préprocessé
        
        Args:
            query_data: Dictionary with keys "kb1_purpose", "kb1_function", "kb2_vector", "kb3_code"
            top_k: Number of results
            
        Returns:
            List of fusionned candidates
        """
        return self.search_hybrid(
            kb1_purpose=query_data.get("kb1_purpose", ""),
            kb1_function=query_data.get("kb1_function", ""),
            kb2_vector=query_data.get("kb2_vector"),
            kb3_code=query_data.get("kb3_code", ""),
            top_k=top_k
        )

# Utility functions
def create_default_controller() -> Vuln_RAGRetrievalController:
    """Create a controller with default paths"""
    import os   
    
    return Vuln_RAGRetrievalController(
        kb1_index_path=os.getenv("KB1_INDEX_PATH", "data/KBs/kb1_index"),
        kb2_index_path=os.getenv("KB2_INDEX_PATH", "data/KBs/kb2_index/kb2_code.index"), 
        kb2_metadata_path=os.getenv("KB2_METADATA_PATH", "data/KBs/kb2_index/kb2_metadata.json"),
        kb3_index_path=os.getenv("KB3_INDEX_PATH", "data/KBs/kb3_index/kb3_code.index"),
        kb3_metadata_path=os.getenv("KB3_METADATA_PATH", "data/KBs/kb3_index/kb3_metadata.json"),
        rrf_weights={"kb1": 0.3, "kb2": 0.4, "kb3": 0.3}  # Privilégier structure
    )

def analyze_fusion_performance(candidates: List[FusionCandidate]) -> Dict[str, Any]:
    """Analyze the performance of the RRF fusion"""
    if not candidates:
        return {"error": "No candidates"}
    
    # Source coverage
    source_coverage = {
        "kb1_coverage": sum(1 for c in candidates if c.kb1_rank is not None) / len(candidates),
        "kb2_coverage": sum(1 for c in candidates if c.kb2_rank is not None) / len(candidates),
        "kb3_coverage": sum(1 for c in candidates if c.kb3_rank is not None) / len(candidates)
    }
    
    # Diversité des sources (combien de candidats viennent de plusieurs sources)
    multi_source = sum(1 for c in candidates if c.total_sources > 1)
    diversity_ratio = multi_source / len(candidates)
    
    # Distribution des scores
    scores = [c.final_score for c in candidates[:10]]
    score_stats = {
        "max_score": max(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "score_range": max(scores) - min(scores) if scores else 0,
        "score_variance": sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores) if scores else 0
    }
    
    return {
        "total_candidates": len(candidates),
        "source_coverage": source_coverage,
        "diversity_ratio": diversity_ratio,
        "score_statistics": score_stats,
        "top_3_scores": scores[:3]
    }

