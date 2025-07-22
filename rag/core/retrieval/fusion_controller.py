# core/retrieval/fusion_controller.py
"""
Reciprocal Rank Fusion (RRF) Controller for Vuln_RAG -
Combines the results of the 3 search engines: KB1 (Whoosh), KB2 (HNSW CPG), KB3 (HNSW Code)
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

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) algorithm
    Combines multiple ranked lists into a single ranked list
    """
    
    def __init__(self, k: int = 60, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            k: RRF parameter (default: 60)
            weights: Weights for each search engine
        """
        self.k = k
        self.weights = weights or {"kb1": 1.0, "kb2": 1.0, "kb3": 1.0}
        
    def fuse(self, results: Dict[str, List[SearchResult]]) -> List[SearchResult]:
        """
        Fuse multiple ranked lists using RRF
        
        Args:
            results: Dictionary of search results by engine
            
        Returns:
            Fused ranked list
        """
        # Collect all unique keys
        all_keys = set()
        for engine_results in results.values():
            for result in engine_results:
                all_keys.add(result.key)
        
        # Calculate RRF scores
        rrf_scores = {}
        for key in all_keys:
            rrf_score = 0.0
            for engine, engine_results in results.items():
                weight = self.weights.get(engine, 1.0)
                # Find rank of this key in this engine's results
                rank = None
                for result in engine_results:
                    if result.key == key:
                        rank = result.rank
                        break
                
                if rank is not None:
                    rrf_score += weight / (self.k + rank)
            
            rrf_scores[key] = rrf_score
        
        # Create fused results
        fused_results = []
        for key, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            # Find the best metadata for this key
            best_metadata = None
            best_source = None
            for engine, engine_results in results.items():
                for result in engine_results:
                    if result.key == key:
                        if best_metadata is None or result.score > best_metadata.get('score', 0):
                            best_metadata = result.metadata
                            best_source = engine
            
            fused_results.append(SearchResult(
                key=key,
                score=score,
                rank=len(fused_results) + 1,
                source=best_source or "fused",
                metadata=best_metadata
            ))
        
        return fused_results

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
            kb2_index_path: Path to HNSW KB2 index  
            kb2_metadata_path: Path to KB2 metadata
            kb3_index_path: Path to HNSW KB3 index
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
                from search_kb2_hnsw import get_kb2_structure_searcher
                self._kb2_searcher = get_kb2_structure_searcher(
                    self.kb2_index_path, self.kb2_metadata_path
                )
            except ImportError:
                logger.error("Unable to import search_kb2_hnsw")
                raise
        return self._kb2_searcher
    
    def _get_kb3_searcher(self):
        """Lazy load KB3 searcher"""
        if self._kb3_searcher is None:
            sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "retrieval"))
            try:
                from search_kb3_code_hnsw import get_kb3_searcher
                self._kb3_searcher = get_kb3_searcher(
                    self.kb3_index_path, self.kb3_metadata_path
                )
            except ImportError:
                logger.error("Unable to import search_kb3_code_hnsw")
                raise
        return self._kb3_searcher
    
    def search(self, 
               purpose_text: str = "",
               function_text: str = "",
               embedding_vector: Optional[List[float]] = None,
               code_snippet: str = "",
               top_k: int = 10,
               verbose: bool = False) -> Dict[str, Any]:
        """
        Perform hybrid search across all 3 knowledge bases
        
        Args:
            purpose_text: Text for KB1 search
            function_text: Additional text for KB1 search
            embedding_vector: Vector for KB2 search
            code_snippet: Code for KB3 search
            top_k: Number of results per engine
            verbose: Verbose output
            
        Returns:
            Dictionary with fused results and individual engine results
        """
        start_time = time.time()
        
        # Search each engine
        results = {}
        
        # KB1 search (Whoosh)
        if purpose_text or function_text:
            try:
                kb1_searcher = self._get_kb1_searcher()
                kb1_results = kb1_searcher(
                    purpose_text=purpose_text,
                    function_text=function_text,
                    top_k=top_k
                )
                
                # Convert to SearchResult format
                kb1_search_results = []
                for i, result in enumerate(kb1_results):
                    kb1_search_results.append(SearchResult(
                        key=result.get('key', f'kb1_{i}'),
                        score=result.get('score', 0.0),
                        rank=i + 1,
                        source='kb1',
                        metadata=result
                    ))
                results['kb1'] = kb1_search_results
                
                if verbose:
                    logger.info(f"KB1 found {len(kb1_search_results)} results")
                    
            except Exception as e:
                logger.error(f"KB1 search failed: {e}")
                results['kb1'] = []
        
        # KB2 search (HNSW)
        if embedding_vector:
            try:
                kb2_searcher = self._get_kb2_searcher()
                kb2_results, _ = kb2_searcher.search(embedding_vector, top_k=top_k)
                
                # Convert to SearchResult format
                kb2_search_results = []
                for result in kb2_results:
                    kb2_search_results.append(SearchResult(
                        key=result['key'],
                        score=result['score'],
                        rank=result['rank'],
                        source='kb2',
                        metadata=result
                    ))
                results['kb2'] = kb2_search_results
                
                if verbose:
                    logger.info(f"KB2 found {len(kb2_search_results)} results")
                    
            except Exception as e:
                logger.error(f"KB2 search failed: {e}")
                results['kb2'] = []
        
        # KB3 search (HNSW)
        if code_snippet:
            try:
                kb3_searcher = self._get_kb3_searcher()
                kb3_results, _ = kb3_searcher.search(code_snippet, top_k=top_k)
                
                # Convert to SearchResult format
                kb3_search_results = []
                for result in kb3_results:
                    kb3_search_results.append(SearchResult(
                        key=result['key'],
                        score=result['score'],
                        rank=result['rank'],
                        source='kb3',
                        metadata=result
                    ))
                results['kb3'] = kb3_search_results
                
                if verbose:
                    logger.info(f"KB3 found {len(kb3_search_results)} results")
                    
            except Exception as e:
                logger.error(f"KB3 search failed: {e}")
                results['kb3'] = []
        
        # Fuse results
        fused_results = self.rrf.fuse(results)
        
        # Calculate timing
        total_time = time.time() - start_time
        
        return {
            'fused_results': fused_results,
            'engine_results': results,
            'timing': {
                'total_time': total_time,
                'engines_used': list(results.keys())
            }
        }

