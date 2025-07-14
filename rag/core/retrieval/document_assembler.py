# core/retrieval/document_assembler.py
"""
Document Assembler pour VulRAG
Récupère le contenu complet des documents à partir des clés RRF
et assemble le contexte enrichi pour le LLM.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .fusion_controller import FusionCandidate

logger = logging.getLogger(__name__)

@dataclass 
class EnrichedDocument:
    """Document complet assemblé depuis les 3 KBs"""
    key: str
    final_score: float
    
    # Contenu KB1 (texte)
    gpt_purpose: str = ""
    gpt_function: str = ""
    code_before_change: str = ""
    code_after_change: str = ""
    cve_id: str = ""
    
    # Contenu KB2 (structure)  
    embedding_text: str = ""
    dangerous_functions: List[str] = None
    cpg_signature: Dict = None
    
    # Métadonnées
    cwe: str = ""
    provenance: Dict = None

class DocumentAssembler:
    """Assemble les documents complets à partir des clés RRF"""
    
    def __init__(self, 
                 kb1_path: str = "data/KBs/JSON_FORMAT_KB/kb1.json",
                 kb2_metadata_path: str = "data/KBs/kb2_index/kb2_metadata.json", 
                 kb3_metadata_path: str = "data/KBs/kb3_index/kb3_metadata.json"):
        """
        Args:
            kb1_path: Chemin vers KB1 JSON complet
            kb2_metadata_path: Chemin vers métadonnées KB2
            kb3_metadata_path: Chemin vers métadonnées KB3
        """
        self.kb1_path = Path(kb1_path)
        self.kb2_metadata_path = Path(kb2_metadata_path)
        self.kb3_metadata_path = Path(kb3_metadata_path)
        
        # Charger les données sources
        self._load_kb_data()
    
    def _load_kb_data(self):
        """Charge les données des 3 KBs en mémoire"""
        logger.info("Chargement des données KB...")
        
        # KB1 - Données complètes
        if self.kb1_path.exists():
            with open(self.kb1_path, 'r', encoding='utf-8') as f:
                self.kb1_data = json.load(f)
        else:
            logger.warning(f"KB1 non trouvé: {self.kb1_path}")
            self.kb1_data = {}
            
        # KB2 - Métadonnées  
        if self.kb2_metadata_path.exists():
            with open(self.kb2_metadata_path, 'r', encoding='utf-8') as f:
                self.kb2_metadata = json.load(f)
                self.kb2_lookup = {item['key']: item for item in self.kb2_metadata}
        else:
            logger.warning(f"KB2 metadata non trouvé: {self.kb2_metadata_path}")
            self.kb2_lookup = {}
            
        # KB3 - Métadonnées
        if self.kb3_metadata_path.exists():
            with open(self.kb3_metadata_path, 'r', encoding='utf-8') as f:
                self.kb3_metadata = json.load(f)
                self.kb3_lookup = {item['key']: item for item in self.kb3_metadata}
        else:
            logger.warning(f"KB3 metadata non trouvé: {self.kb3_metadata_path}")
            self.kb3_lookup = {}
            
        logger.info(f"KB1: {len(self.kb1_data)} entrées")
        logger.info(f"KB2: {len(self.kb2_lookup)} entrées") 
        logger.info(f"KB3: {len(self.kb3_lookup)} entrées")
    
    def assemble_documents(self, 
                          candidates: List[FusionCandidate],
                          top_k: int = 5) -> List[EnrichedDocument]:
        """
        Assemble les documents enrichis à partir des candidats RRF
        
        Args:
            candidates: Liste des candidats avec scores RRF
            top_k: Nombre de documents à assembler
            
        Returns:
            Liste des documents enrichis triés par score final
        """
        enriched_docs = []
        
        for candidate in candidates[:top_k]:
            try:
                doc = self._assemble_single_document(candidate)
                if doc:
                    enriched_docs.append(doc)
            except Exception as e:
                logger.warning(f"Erreur assemblage {candidate.key}: {e}")
                continue
                
        logger.info(f"Documents assemblés: {len(enriched_docs)}/{top_k}")
        return enriched_docs
    
    def _assemble_single_document(self, candidate: FusionCandidate) -> Optional[EnrichedDocument]:
        """Assemble un document unique depuis les 3 sources"""
        key = candidate.key
        
        # Récupérer depuis KB1 (source principale)
        kb1_entry = self.kb1_data.get(key)
        if not kb1_entry:
            logger.warning(f"Clé {key} introuvable dans KB1")
            return None
            
        # Récupérer métadonnées KB2 et KB3
        kb2_meta = self.kb2_lookup.get(key, {})
        kb3_meta = self.kb3_lookup.get(key, {})
        
        # Extraction sécurisée des champs
        gpt_purpose = kb1_entry.get("GPT_purpose", "")
        gpt_function = kb1_entry.get("GPT_function", "")
        
        # Code avant/après
        code_before = kb1_entry.get("code_before_change", "")
        code_after = kb1_entry.get("code_after_change", "")
        
        # CVE ID
        cve_id = kb1_entry.get("CVE_id", kb1_entry.get("cve_id", ""))
        
        # Embedding text depuis KB2
        embedding_text = kb2_meta.get("embedding_text", "")
        
        # Fonctions dangereuses depuis KB2 (si disponible)
        dangerous_functions = kb2_meta.get("dangerous_functions", [])
        if isinstance(dangerous_functions, str):
            dangerous_functions = [dangerous_functions]
            
        # CWE depuis la clé
        cwe = key.split("_")[0] if "_" in key else ""
        
        # Construire le document enrichi
        doc = EnrichedDocument(
            key=key,
            final_score=candidate.final_score,
            
            # Contenu KB1 (descriptif neutre)
            gpt_purpose=gpt_purpose,
            gpt_function=gpt_function,
            code_before_change=code_before,
            code_after_change=code_after,
            cve_id=cve_id,
            
            # Contenu KB2 (structure)
            embedding_text=embedding_text,
            dangerous_functions=dangerous_functions,
            
            # Métadonnées
            cwe=cwe,
            
            # Provenance pour debugging
            provenance={
                "kb1_rank": candidate.kb1_rank,
                "kb2_rank": candidate.kb2_rank, 
                "kb3_rank": candidate.kb3_rank,
                "kb1_score": candidate.kb1_score,
                "kb2_score": candidate.kb2_score,
                "kb3_score": candidate.kb3_score,
                "search_time_ms": candidate.search_time_ms,
                "total_sources": candidate.total_sources
            }
        )
        
        return doc

class ContextBuilder:
    """Construit le contexte structuré pour le LLM"""
    
    @staticmethod
    def build_detection_context(original_code: str, 
                              enriched_docs: List[EnrichedDocument],
                              max_context_length: int = 4000) -> str:
        """
        Construit un prompt structuré pour la détection de vulnérabilité
        
        Args:
            original_code: Code source à analyser
            enriched_docs: Documents similaires trouvés
            max_context_length: Limite de contexte en caractères
            
        Returns:
            Prompt Markdown structuré pour le LLM
        """
        context_parts = []
        
        # En-tête avec le code à analyser
        context_parts.append("# Analyse de Vulnérabilité C/C++")
        context_parts.append("\n## Code Source à Analyser")
        context_parts.append("```c")
        context_parts.append(original_code.strip())
        context_parts.append("```\n")
        
        # Contexte des documents similaires
        if enriched_docs:
            context_parts.append("## Exemples Similaires Détectés")
            
            for i, doc in enumerate(enriched_docs[:3], 1):  # Top 3 seulement
                context_parts.append(f"\n### Exemple {i} (Score: {doc.final_score:.3f}, CWE: {doc.cwe})")
                
                if doc.gpt_purpose:
                    context_parts.append(f"**Objectif**: {doc.gpt_purpose}")
                    
                if doc.gpt_function:
                    context_parts.append(f"**Fonction**: {doc.gpt_function}")
                    
                if doc.embedding_text:
                    # Tronquer le texte d'embedding
                    embedding_preview = doc.embedding_text[:200]
                    if len(doc.embedding_text) > 200:
                        embedding_preview += "..."
                    context_parts.append(f"**Structure**: {embedding_preview}")
                    
                if doc.dangerous_functions:
                    funcs_str = ", ".join(doc.dangerous_functions[:5])
                    context_parts.append(f"**Fonctions dangereuses**: {funcs_str}")
                    
                # Code similaire (tronqué)
                if doc.code_before_change:
                    context_parts.append("**Code similaire vulnérable**:")
                    context_parts.append("```c")
                    code_preview = doc.code_before_change[:300]
                    if len(doc.code_before_change) > 300:
                        code_preview += "\n// ..."
                    context_parts.append(code_preview)
                    context_parts.append("```")
        else:
            context_parts.append("## Aucun Exemple Similaire Trouvé")
            context_parts.append("Analysez le code de manière autonome.")
        
        # Instructions pour le LLM
        context_parts.append("\n## Instructions")
        context_parts.append("Analysez le code source et déterminez s'il contient une vulnérabilité.")
        context_parts.append("Basez-vous sur les exemples similaires si disponibles.")
        context_parts.append("Répondez au format JSON uniquement:")
        context_parts.append('```json')
        context_parts.append('{')
        context_parts.append('  "is_vulnerable": true/false,')
        context_parts.append('  "confidence": 0.0-1.0,')
        context_parts.append('  "cwe": "CWE-XXX",')
        context_parts.append('  "explanation": "Description détaillée de la vulnérabilité"')
        context_parts.append('}')
        context_parts.append('```')
        
        full_context = "\n".join(context_parts)
        
        # Tronquer si trop long
        if len(full_context) > max_context_length:
            logger.warning(f"Contexte tronqué: {len(full_context)} -> {max_context_length} chars")
            full_context = full_context[:max_context_length] + "\n\n[...contexte tronqué...]"
            
        return full_context
    
    @staticmethod
    def build_patch_context(original_code: str,
                           detection_result: Dict,
                           enriched_docs: List[EnrichedDocument],
                           max_context_length: int = 5000) -> str:
        """Construit le contexte pour la génération de patch"""
        context_parts = []
        
        context_parts.append("# Génération de Correctif C/C++")
        context_parts.append(f"\n## Vulnérabilité Détectée")
        context_parts.append(f"**Type**: {detection_result.get('cwe', 'Unknown')}")
        context_parts.append(f"**Confiance**: {detection_result.get('confidence', 0):.2f}")
        context_parts.append(f"**Explication**: {detection_result.get('explanation', '')}")
        
        context_parts.append("\n## Code Vulnérable")
        context_parts.append("```c")
        context_parts.append(original_code.strip())
        context_parts.append("```")
        
        # Exemples de correctifs
        if enriched_docs:
            context_parts.append("\n## Exemples de Correctifs")
            patch_count = 0
            for i, doc in enumerate(enriched_docs[:3], 1):
                if doc.code_after_change and patch_count < 2:  # Max 2 exemples
                    patch_count += 1
                    context_parts.append(f"\n### Correctif Exemple {patch_count}")
                    context_parts.append("```c")
                    code_patch = doc.code_after_change[:500]
                    if len(doc.code_after_change) > 500:
                        code_patch += "\n// ..."
                    context_parts.append(code_patch)
                    context_parts.append("```")
        
        context_parts.append("\n## Instructions")
        context_parts.append("Générez un correctif sécurisé pour le code vulnérable.")
        context_parts.append("Inspirez-vous des exemples de correctifs fournis.")
        context_parts.append("Répondez uniquement avec le code corrigé complet, sans explication supplémentaire.")
        
        full_context = "\n".join(context_parts)
        
        # Tronquer si nécessaire
        if len(full_context) > max_context_length:
            logger.warning(f"Contexte patch tronqué: {len(full_context)} -> {max_context_length}")
            full_context = full_context[:max_context_length] + "\n// [contexte tronqué...]"
            
        return full_context

# Test/Debug functions
def test_document_assembler():
    """Test de l'assembleur de documents"""
    from .fusion_controller import FusionCandidate
    
    # Simuler des candidats RRF
    candidates = [
        FusionCandidate(
            key="CWE-119_CVE-2016-1234_1",
            final_score=0.85,
            kb1_rank=1, kb2_rank=2, kb3_rank=1,
            kb1_score=0.9, kb2_score=0.8, kb3_score=0.85
        ),
        FusionCandidate(
            key="CWE-119_CVE-2017-5678_2", 
            final_score=0.72,
            kb1_rank=3, kb2_rank=1, kb3_rank=5,
            kb1_score=0.7, kb2_score=0.9, kb3_score=0.6
        )
    ]
    
    assembler = DocumentAssembler()
    enriched_docs = assembler.assemble_documents(candidates)
    
    print(f"Documents assemblés: {len(enriched_docs)}")
    for doc in enriched_docs:
        print(f"- {doc.key}: score={doc.final_score:.3f}, CWE={doc.cwe}")
        print(f"  Purpose: {doc.gpt_purpose[:100]}...")
        print(f"  Provenance: KB1({doc.provenance['kb1_rank']}), KB2({doc.provenance['kb2_rank']}), KB3({doc.provenance['kb3_rank']})")
    
    # Test contexte
    original_code = """
char buffer[10];
strcpy(buffer, user_input);  // Vulnérabilité potentielle
printf("Data: %s", buffer);
"""
    
    context = ContextBuilder.build_detection_context(original_code, enriched_docs)
    print("\n" + "="*60)
    print("CONTEXTE DE DÉTECTION GÉNÉRÉ:")
    print("="*60)
    print(context[:1200] + "..." if len(context) > 1200 else context)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_document_assembler()