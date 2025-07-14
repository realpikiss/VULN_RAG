# core/retrieval/fusion_controller.py
"""
Reciprocal Rank Fusion (RRF) Controller pour VulRAG
Combine les résultats des 3 moteurs de recherche : KB1 (Whoosh), KB2 (FAISS CPG), KB3 (FAISS Code)
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys

# Import des searchers existants
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts" / "retrieval"))
from search_kb1 import search_kb1
from search_kb2_faiss import search_kb2_faiss  
from search_kb3_code_faiss import search_kb3_code

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Résultat individuel d'un moteur de recherche"""
    key: str
    score: float
    rank: int
    source: str  # "kb1", "kb2", "kb3"
    metadata: Dict[str, Any] = None

@dataclass
class FusionCandidate:
    """Candidat après fusion RRF avec provenance complète"""
    key: str
    final_score: float
    
    # Scores et rangs individuels
    kb1_rank: Optional[int] = None
    kb2_rank: Optional[int] = None
    kb3_rank: Optional[int] = None
    kb1_score: Optional[float] = None
    kb2_score: Optional[float] = None
    kb3_score: Optional[float] = None
    
    # Métadonnées de performance
    search_time_ms: float = 0.0
    total_sources: int = 0

class ReciprocalRankFusion:
    """
    Implémentation du Reciprocal Rank Fusion (RRF) pour 3 sources
    
    RRF Score = Σ(1 / (k + rank_i)) pour chaque source i
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
        
    def fuse(self, 
             kb1_results: List[Dict], 
             kb2_results: List[Dict], 
             kb3_results: List[Dict]) -> List[FusionCandidate]:
        """
        Fusion RRF des résultats des 3 moteurs
        
        Args:
            kb1_results: Résultats Whoosh [{"key": str, "score": float}, ...]
            kb2_results: Résultats FAISS CPG [{"key": str, "score": float}, ...]  
            kb3_results: Résultats FAISS Code [{"key": str, "score": float}, ...]
            
        Returns:
            Liste de FusionCandidate triés par score RRF décroissant
        """
        # Convertir en SearchResult avec rangs
        search_results = []
        
        for rank, result in enumerate(kb1_results, 1):
            search_results.append(SearchResult(
                key=result["key"],
                score=result["score"], 
                rank=rank,
                source="kb1",
                metadata=result
            ))
            
        for rank, result in enumerate(kb2_results, 1):
            search_results.append(SearchResult(
                key=result["key"],
                score=result["score"],
                rank=rank, 
                source="kb2",
                metadata=result
            ))
            
        for rank, result in enumerate(kb3_results, 1):
            search_results.append(SearchResult(
                key=result["key"],
                score=result["score"],
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
            
            # Données de provenance
            kb1_rank = kb1_score = kb2_rank = kb2_score = kb3_rank = kb3_score = None
            
            # Calcul RRF pondéré
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
            
        # Trier par score RRF décroissant
        fusion_candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        logger.info(f"RRF Fusion: {len(fusion_candidates)} candidats uniques générés")
        return fusion_candidates

class VulRAGRetrievalController:
    """
    Contrôleur principal pour la recherche hybride VulRAG
    Coordonne les 3 moteurs + fusion RRF
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
            kb1_index_path: Chemin vers l'index Whoosh
            kb2_index_path: Chemin vers l'index FAISS KB2  
            kb2_metadata_path: Chemin vers métadonnées KB2
            kb3_index_path: Chemin vers l'index FAISS KB3
            kb3_metadata_path: Chemin vers métadonnées KB3
            rrf_k: Paramètre RRF
            rrf_weights: Poids par moteur
        """
        self.kb1_index_path = kb1_index_path
        self.kb2_index_path = kb2_index_path
        self.kb2_metadata_path = kb2_metadata_path
        self.kb3_index_path = kb3_index_path
        self.kb3_metadata_path = kb3_metadata_path
        
        # Initialiseur RRF
        self.rrf = ReciprocalRankFusion(k=rrf_k, weights=rrf_weights)
        
        logger.info("VulRAG Retrieval Controller initialisé")
        
    def search_hybrid(self, 
                     kb1_purpose: str = "",
                     kb1_function: str = "", 
                     kb2_vector: Optional[List[float]] = None,
                     kb3_code: str = "",
                     top_k: int = 10) -> List[FusionCandidate]:
        """
        Recherche hybride dans les 3 KBs avec fusion RRF
        
        Args:
            kb1_purpose: Texte pour recherche KB1 (purpose)
            kb1_function: Texte pour recherche KB1 (function)
            kb2_vector: Vecteur d'embedding pour KB2
            kb3_code: Code source pour recherche KB3
            top_k: Nombre de résultats par moteur
            
        Returns:
            Liste fusionnée et triée de candidats
        """
        start_time = time.time()
        
        # Recherche parallèle dans les 3 KBs
        kb1_results = []
        kb2_results = []
        kb3_results = []
        
        # KB1 - Recherche textuelle Whoosh
        if kb1_purpose or kb1_function:
            try:
                kb1_results = search_kb1(
                    purpose_text=kb1_purpose,
                    function_text=kb1_function, 
                    top_k=top_k,
                    index_dir=self.kb1_index_path
                )
                logger.info(f"KB1 Whoosh: {len(kb1_results)} résultats")
            except Exception as e:
                logger.warning(f"Erreur KB1: {e}")
                
        # KB2 - Recherche vectorielle CPG  
        if kb2_vector:
            try:
                kb2_results = search_kb2_faiss(
                    query_vector=kb2_vector,
                    top_k=top_k,
                    index_path=self.kb2_index_path,
                    metadata_path=self.kb2_metadata_path
                )
                logger.info(f"KB2 FAISS CPG: {len(kb2_results)} résultats")
            except Exception as e:
                logger.warning(f"Erreur KB2: {e}")
                
        # KB3 - Recherche vectorielle Code
        if kb3_code:
            try:
                kb3_results = search_kb3_code(
                    code_query=kb3_code,
                    top_k=top_k,
                    index_path=self.kb3_index_path,
                    metadata_path=self.kb3_metadata_path
                )
                logger.info(f"KB3 FAISS Code: {len(kb3_results)} résultats")
            except Exception as e:
                logger.warning(f"Erreur KB3: {e}")
                
        # Fusion RRF
        fusion_candidates = self.rrf.fuse(kb1_results, kb2_results, kb3_results)
        
        # Ajouter timing
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
            query_data: Dictionnaire avec clés "kb1_purpose", "kb1_function", "kb2_vector", "kb3_code"
            top_k: Nombre de résultats
            
        Returns:
            Candidats fusionnés
        """
        return self.search_hybrid(
            kb1_purpose=query_data.get("kb1_purpose", ""),
            kb1_function=query_data.get("kb1_function", ""),
            kb2_vector=query_data.get("kb2_vector"),
            kb3_code=query_data.get("kb3_code", ""),
            top_k=top_k
        )

# Fonctions utilitaires
def create_default_controller() -> VulRAGRetrievalController:
    """Crée un contrôleur avec les chemins par défaut correspondant à votre structure"""
    return VulRAGRetrievalController(
        kb1_index_path="data/KBs/kb1_index",
        kb2_index_path="data/KBs/kb2_index/kb2_code.index", 
        kb2_metadata_path="data/KBs/kb2_index/kb2_metadata.json",
        kb3_index_path="data/KBs/kb3_index/kb3_code.index",
        kb3_metadata_path="data/KBs/kb3_index/kb3_metadata.json",
        rrf_weights={"kb1": 0.3, "kb2": 0.4, "kb3": 0.3}  # Privilégier structure
    )

def test_fusion_controller():
    """Test du contrôleur de fusion"""
    controller = create_default_controller()
    
    # Test avec requête simple
    test_code = """
    char buffer[10];
    strcpy(buffer, user_input);
    """
    
    results = controller.search_hybrid(
        kb1_purpose="buffer overflow vulnerability",
        kb1_function="string copy without bounds checking", 
        kb3_code=test_code,
        top_k=5
    )
    
    print(f"Résultats fusion: {len(results)}")
    for i, candidate in enumerate(results[:3], 1):
        print(f"{i}. {candidate.key}: score={candidate.final_score:.3f}")
        print(f"   Provenance: KB1({candidate.kb1_rank}), KB2({candidate.kb2_rank}), KB3({candidate.kb3_rank})")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_fusion_controller()