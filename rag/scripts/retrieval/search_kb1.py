# scripts/retrieval/search_kb1_.py
"""
Searcher pour KB1 enrichi dans Whoosh
Récupère des documents COMPLETS directement depuis l'index
"""

import argparse
import os
import json
from pathlib import Path
from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser, QueryParser, AndGroup
from whoosh import scoring
from whoosh.query import Term, And, Or
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def _open_index(index_dir: str):
    """Open Whoosh index, raising if missing."""
    if index_dir is None:
        index_dir = os.getenv("KB1_INDEX_PATH")
    if index_dir is None or not Path(index_dir).exists():
        raise FileNotFoundError(f"❌ Index not found: {index_dir}")
    return open_dir(index_dir)

def search_kb1_(purpose_text: str = "", 
                       function_text: str = "",
                       key_filter: List[str] = None,
                       cwe_filter: str = None,
                       top_k: int = 10, 
                       index_dir: str = None) -> List[Dict[str, Any]]:
    """
    Recherche enrichie dans KB1 Whoosh avec documents complets
    
    Args:
        purpose_text: Texte pour recherche dans gpt_purpose
        function_text: Texte pour recherche dans gpt_function
        key_filter: Liste de clés spécifiques à rechercher (pour filtrage FAISS)
        cwe_filter: Filtrer par CWE spécifique
        top_k: Nombre de résultats
        index_dir: Répertoire de l'index
        
    Returns:
        Liste de documents complets avec tous les champs
    """
    ix = _open_index(index_dir)
    
    with ix.searcher(weighting=scoring.BM25F()) as searcher:
        
        # Construction de la requête
        queries = []
        
        # 1. Requête textuelle sur purpose/function
        if purpose_text or function_text:
            parser = MultifieldParser(["gpt_purpose", "gpt_function"], schema=ix.schema)
            
            # Construire la requête textuelle
            text_parts = []
            if purpose_text:
                text_parts.append(f'gpt_purpose:("{purpose_text}")')
            if function_text:
                text_parts.append(f'gpt_function:("{function_text}")')
            
            if text_parts:
                text_query_str = " AND ".join(text_parts)
                text_query = parser.parse(text_query_str)
                queries.append(text_query)
        
        # 2. Filtre par clés spécifiques (venant de FAISS)
        if key_filter:
            key_terms = [Term("key", str(key)) for key in key_filter]
            key_query = Or(key_terms)
            queries.append(key_query)
        
        # 3. Filtre par CWE
        if cwe_filter:
            cwe_query = Term("cwe", cwe_filter)
            queries.append(cwe_query)
        
        # Combiner toutes les requêtes
        if queries:
            if len(queries) == 1:
                final_query = queries[0]
            else:
                final_query = And(queries)
        else:
            # Si aucune requête, rechercher tous les documents
            from whoosh.query import Every
            final_query = Every()
        
        # Exécuter la recherche
        results = searcher.search(final_query, limit=top_k)
        
        # Extraire les documents complets
        docs = []
        for hit in results:
            # Document complet depuis Whoosh avec données KB1 + KB2
            doc = {
                "key": hit["key"],
                "score": hit.score,
                
                # Champs textuels KB1
                "gpt_purpose": hit.get("gpt_purpose", ""),
                "gpt_function": hit.get("gpt_function", ""),
                "gpt_analysis": hit.get("gpt_analysis", ""),
                "solution": hit.get("solution", ""),
                
                # Code source KB1
                "code_before_change": hit.get("code_before_change", ""),
                "code_after_change": hit.get("code_after_change", ""),
                
                # Analyses CPG KB1
                "cpg_vulnerability_pattern": hit.get("cpg_vulnerability_pattern", ""),
                "patch_transformation_analysis": hit.get("patch_transformation_analysis", ""),
                "detection_cpg_signature": hit.get("detection_cpg_signature", ""),
                "remediation_graph_guidance": hit.get("remediation_graph_guidance", ""),
                
                # Identifiants
                "cve_id": hit.get("cve_id", ""),
                "cwe": hit.get("cwe", ""),
                
                # Données CPG depuis KB2 (enrichissement)
                "dangerous_functions": _parse_json_field(hit.get("dangerous_functions", "")),
                "dangerous_functions_count": hit.get("dangerous_functions_count", 0),
                "dangerous_functions_detected": hit.get("dangerous_functions_detected", False),
                "risk_class": hit.get("risk_class", "unknown"),
                "complexity_class": hit.get("complexity_class", "unknown"),
                "embedding_text": hit.get("embedding_text", ""),
                
                # Analyse patch depuis KB2
                "dangerous_functions_added": _parse_json_field(hit.get("dangerous_functions_added", "")),
                "dangerous_functions_removed": _parse_json_field(hit.get("dangerous_functions_removed", "")),
                "net_dangerous_change": hit.get("net_dangerous_change", 0),
                
                # Métadonnées calculées
                "code_lines_count": hit.get("code_lines_count", 0),
                "has_code_before": hit.get("has_code_before", False),
                "has_code_after": hit.get("has_code_after", False),
                
                # Champs structurés KB1 (JSON strings → dicts)
                "vulnerability_behavior": _parse_json_field(hit.get("vulnerability_behavior", "")),
                "modified_lines": _parse_json_field(hit.get("modified_lines", ""))
            }
            docs.append(doc)
        
        if not docs:
            logger.warning("KB1 : no results found")
        else:
            logger.info(f"KB1 : {len(docs)} documents retrieved")
            
        return docs

def search_kb1_by_keys(keys: List[str], 
                      index_dir: str = None) -> List[Dict[str, Any]]:
    """
    Recherche spécifique par liste de clés (optimisé pour récupération FAISS)
    
    Args:
        keys: Liste des clés à récupérer
        index_dir: Répertoire de l'index
        
    Returns:
        Documents complets correspondant aux clés
    """
    return search_kb1_(
        key_filter=keys,
        top_k=len(keys),
        index_dir=index_dir
    )

def search_kb1_hybrid(purpose_text: str = "",
                     function_text: str = "",
                     faiss_candidates: List[str] = None,
                     boost_faiss: float = 1.2,
                     top_k: int = 10,
                     index_dir: str = None) -> List[Dict[str, Any]]:
    """
    Recherche hybride : combine recherche textuelle + candidats FAISS
    
    Args:
        purpose_text: Recherche textuelle
        function_text: Recherche textuelle
        faiss_candidates: Clés candidates de FAISS (optionnel)
        boost_faiss: Boost pour les candidats FAISS
        top_k: Nombre de résultats
        index_dir: Répertoire index
        
    Returns:
        Documents avec scores ajustés
    """
    # Recherche textuelle normale
    text_results = search_kb1_(
        purpose_text=purpose_text,
        function_text=function_text,
        top_k=top_k * 2,  # Récupérer plus pour avoir du choix
        index_dir=index_dir
    )
    
    # Si pas de candidats FAISS, retourner résultats textuels
    if not faiss_candidates:
        return text_results[:top_k]
    
    # Boost des scores pour les candidats FAISS
    faiss_set = set(faiss_candidates)
    for doc in text_results:
        if doc["key"] in faiss_set:
            doc["score"] *= boost_faiss
            doc["boosted"] = True
        else:
            doc["boosted"] = False
    
    # Re-trier par score ajusté
    text_results.sort(key=lambda x: x["score"], reverse=True)
    
    return text_results[:top_k]

def _parse_json_field(json_str: str) -> Dict:
    """Parse un champ JSON string vers dict"""
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except Exception:
        return {}

# Fonctions de compatibilité avec l'ancien searcher
def search_kb1(purpose_text: str, function_text: str, top_k: int = 10, index_dir: str = None):
    """Fonction de compatibilité avec l'ancien searcher (format simple)"""
    results = search_kb1_(purpose_text, function_text, top_k=top_k, index_dir=index_dir)
    
    # Convertir au format simple pour compatibilité
    simple_results = []
    for doc in results:
        simple_results.append({
            "key": doc["key"],
            "score": doc["score"],
            "gpt_purpose": doc["gpt_purpose"],
            "gpt_function": doc["gpt_function"]
        })
    return simple_results

# CLI et tests
def test__search():
    """Test du nouveau searcher enrichi"""
    print("🧪 Test KB1  Search")
    
    # Test 1: Recherche textuelle simple
    results = search_kb1_(
        purpose_text="buffer overflow",
        function_text="string copy",
        top_k=3
    )
    
    print(f"\n📊 Test 1 - Recherche textuelle: {len(results)} résultats")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc['key']} (score: {doc['score']:.3f})")
        print(f"   CWE: {doc['cwe']}, Lines: {doc['code_lines_count']}")
        print(f"   Purpose: {doc['gpt_purpose'][:100]}...")
    
    # Test 2: Recherche par clés spécifiques
    if results:
        test_keys = [doc['key'] for doc in results[:2]]
        key_results = search_kb1_by_keys(test_keys)
        
        print(f"\n📊 Test 2 - Recherche par clés: {len(key_results)} résultats")
        for doc in key_results:
            print(f"- {doc['key']}: {len(doc['code_before_change'])} chars de code")
    
    # Test 3: Recherche hybride
    hybrid_results = search_kb1_hybrid(
        purpose_text="vulnerability",
        faiss_candidates=[results[0]['key']] if results else [],
        top_k=3
    )
    
    print(f"\n📊 Test 3 - Recherche hybride: {len(hybrid_results)} résultats")
    for doc in hybrid_results:
        boost_status = "🚀 Boosted" if doc.get('boosted', False) else "📄 Normal"
        print(f"- {doc['key']}: {doc['score']:.3f} {boost_status}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test__search()