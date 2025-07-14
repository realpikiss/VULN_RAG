# core/retrieval/orchestrator.py
"""
Orchestrateur principal pour le système de récupération VulRAG
Coordonne le preprocessing -> recherche hybride -> fusion RRF -> assemblage contexte
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from .fusion_controller import VulRAGRetrievalController, FusionCandidate, create_default_controller
from .document_assembler import DocumentAssembler, ContextBuilder, EnrichedDocument

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Résultat complet d'une recherche VulRAG"""
    # Input
    original_code: str
    query_data: Dict[str, Any]
    
    # Résultats fusion
    fusion_candidates: List[FusionCandidate]
    enriched_documents: List[EnrichedDocument]
    
    # Contexte LLM
    detection_context: str
    patch_context: Optional[str] = None
    
    # Métadonnées performance
    total_time_ms: float = 0.0
    search_time_ms: float = 0.0
    assembly_time_ms: float = 0.0
    context_time_ms: float = 0.0
    
    # Statistiques
    stats: Dict[str, Any] = None

class VulRAGRetrievalOrchestrator:
    """
    Orchestrateur principal pour la récupération VulRAG
    
    Pipeline complet:
    1. Preprocessing du code → query_data
    2. Recherche hybride (KB1 + KB2 + KB3)  
    3. Fusion RRF
    4. Assemblage des documents
    5. Construction du contexte LLM
    """
    
    def __init__(self,
                 controller: Optional[VulRAGRetrievalController] = None,
                 assembler: Optional[DocumentAssembler] = None,
                 top_k_search: int = 10,
                 top_k_context: int = 5):
        """
        Args:
            controller: Contrôleur de recherche (None = défaut)
            assembler: Assembleur de documents (None = défaut)
            top_k_search: Nombre de résultats par moteur
            top_k_context: Nombre de documents dans le contexte final
        """
        self.controller = controller or create_default_controller()
        self.assembler = assembler or DocumentAssembler()
        self.top_k_search = top_k_search
        self.top_k_context = top_k_context
        
        logger.info("VulRAG Retrieval Orchestrator initialisé")
    
    def retrieve_context(self,
                        original_code: str,
                        query_data: Dict[str, Any],
                        include_patch_context: bool = False) -> RetrievalResult:
        """
        Pipeline complet de récupération contextuelle
        
        Args:
            original_code: Code source vulnérable à analyser
            query_data: Données de requête préprocessées {
                "kb1_purpose": str,
                "kb1_function": str, 
                "kb2_vector": List[float],
                "kb3_code": str
            }
            include_patch_context: Générer aussi le contexte pour patch
            
        Returns:
            RetrievalResult avec contexte complet pour LLM
        """
        start_time = time.time()
        
        logger.info(f"Début récupération contextuelle pour {len(original_code)} chars de code")
        
        # 1. Recherche hybride + Fusion RRF
        search_start = time.time()
        fusion_candidates = self.controller.search_from_preprocessed_query(
            query_data=query_data,
            top_k=self.top_k_search
        )
        search_time_ms = (time.time() - search_start) * 1000
        
        logger.info(f"Fusion RRF: {len(fusion_candidates)} candidats en {search_time_ms:.1f}ms")
        
        # 2. Assemblage des documents
        assembly_start = time.time()
        enriched_documents = self.assembler.assemble_documents(
            candidates=fusion_candidates,
            top_k=self.top_k_context
        )
        assembly_time_ms = (time.time() - assembly_start) * 1000
        
        logger.info(f"Assemblage: {len(enriched_documents)} documents en {assembly_time_ms:.1f}ms")
        
        # 3. Construction du contexte de détection
        context_start = time.time()
        detection_context = ContextBuilder.build_detection_context(
            original_code=original_code,
            enriched_docs=enriched_documents,
            max_context_length=4000
        )
        
        # 4. Construction du contexte de patch (optionnel)
        patch_context = None
        if include_patch_context:
            # Simuler résultat de détection pour le contexte patch
            dummy_detection = {
                "is_vulnerable": True,
                "confidence": 0.8,
                "cwe": enriched_documents[0].cwe if enriched_documents else "CWE-Unknown",
                "explanation": "Vulnérabilité détectée par système RAG"
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
        
        logger.info(f"Récupération terminée en {total_time_ms:.1f}ms")
        logger.info(f"Contexte détection: {len(detection_context)} chars")
        if patch_context:
            logger.info(f"Contexte patch: {len(patch_context)} chars")
            
        return result
    
    def _compute_stats(self, 
                      candidates: List[FusionCandidate], 
                      documents: List[EnrichedDocument]) -> Dict[str, Any]:
        """Calcule les statistiques de la recherche"""
        if not candidates:
            return {"candidates": 0, "documents": 0, "coverage": {}}
            
        # Distribution des sources
        kb1_count = sum(1 for c in candidates if c.kb1_rank is not None)
        kb2_count = sum(1 for c in candidates if c.kb2_rank is not None) 
        kb3_count = sum(1 for c in candidates if c.kb3_rank is not None)
        
        # Distribution CWE
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
        Recherche rapide avec paramètres simplifiés
        
        Args:
            code: Code source à analyser
            purpose: Description de l'objectif (optionnel)
            function: Description de la fonction (optionnel)
            
        Returns:
            RetrievalResult avec contexte de détection
        """
        # Construction query simplifiée
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
    """Analyseur pour les résultats de recherche VulRAG"""
    
    @staticmethod
    def print_summary(result: RetrievalResult, detailed: bool = False):
        """Affiche un résumé du résultat de recherche"""
        print("=" * 60)
        print("VulRAG RETRIEVAL SUMMARY")
        print("=" * 60)
        
        # Timing
        print(f"⏱️  Total Time: {result.total_time_ms:.1f}ms")
        print(f"   ├─ Search: {result.search_time_ms:.1f}ms")
        print(f"   ├─ Assembly: {result.assembly_time_ms:.1f}ms")
        print(f"   └─ Context: {result.context_time_ms:.1f}ms")
        
        # Résultats
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
        
        # Détails si demandé
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
        """Exporte le contexte vers un fichier"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# VulRAG Detection Context\n\n")
            f.write(result.detection_context)
            
            if result.patch_context:
                f.write("\n\n" + "="*60 + "\n")
                f.write("# VulRAG Patch Context\n\n")
                f.write(result.patch_context)
                
        logger.info(f"Contexte exporté vers {output_file}")

# Tests et exemples
def test_orchestrator():
    """Test complet de l'orchestrateur"""
    orchestrator = VulRAGRetrievalOrchestrator()
    
    # Code de test
    test_code = """
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    char buffer[10];
    strcpy(buffer, argv[1]);  // Vulnérabilité buffer overflow
    printf("Input: %s\\n", buffer);
    return 0;
}
"""
    
    # Test recherche rapide
    result = orchestrator.quick_search(
        code=test_code,
        purpose="copy user input to buffer",
        function="string manipulation without bounds checking"
    )
    
    # Analyse des résultats
    RetrievalResultAnalyzer.print_summary(result, detailed=True)
    
    # Export optionnel
    # RetrievalResultAnalyzer.export_context(result, "output/test_context.md")
    
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_orchestrator()