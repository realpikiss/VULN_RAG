# core/retrieval/document_assembler.py
"""
Document Assembler for Vuln_RAG - VERSION WHOOSH
Retrieves the complete content of documents from the enriched Whoosh index
and assembles the enriched context for the LLM.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

from whoosh.index import open_dir
from whoosh.query import Term, Or

logger = logging.getLogger(__name__)

@dataclass 
class EnrichedDocument:
    """Complete document assembled from the enriched Whoosh index""" 
    key: str
    final_score: float
    
    # Content KB1 (descriptive text)
    gpt_purpose: str = ""
    gpt_function: str = ""
    gpt_analysis: str = ""
    solution: str = ""
    code_before_change: str = ""
    code_after_change: str = ""
    cve_id: str = ""
    
    # CPG KB1 analysis
    cpg_vulnerability_pattern: str = ""
    patch_transformation_analysis: str = ""
    detection_cpg_signature: str = ""
    remediation_graph_guidance: str = ""
    
    # KB2 enriched content (from Whoosh)
    embedding_text: str = ""
    dangerous_functions: List[str] = field(default_factory=list)
    dangerous_functions_count: int = 0
    dangerous_functions_detected: bool = False
    risk_class: str = "unknown"
    complexity_class: str = "unknown"
    
    #   Patch analysis from KB2
    dangerous_functions_added: List[str] = field(default_factory=list)
    dangerous_functions_removed: List[str] = field(default_factory=list)
    net_dangerous_change: int = 0
    
    # Enriched metadata
    cwe: str = ""
    code_lines_count: int = 0
    has_code_before: bool = False
    has_code_after: bool = False
    vulnerability_behavior: Dict = field(default_factory=dict)
    modified_lines: Dict = field(default_factory=dict)
    
    # Provenance details
    provenance: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validation and post-initialization calculations"""
        # Calculate derived indicators
        self.has_dangerous_functions = self.dangerous_functions_count > 0
        self.vulnerability_severity = self._calculate_severity()
    
    def _calculate_severity(self) -> str:
        """Calculate severity based on available indicators"""
        if self.risk_class in ["high", "critical"]:
            return "high"
        elif self.dangerous_functions_count >= 3:
            return "medium"
        elif self.dangerous_functions_count > 0:
            return "low"
        else:
            return "unknown"

class DocumentAssembler:
    """Assemble the complete documents from the enriched Whoosh index"""
    
    def __init__(self, index_path: Optional[str] = None):
        """
        Args:
            index_path: Path to the enriched Whoosh index (KB1 + KB2)
        """
        self.index_path = index_path or os.getenv("KB1_INDEX_PATH", "data/KBs/kb1_index")
        
        # Lazy loading of the index
        self._index = None
        
        # Statistics
        self._stats = {
            "index_loaded": False,
            "documents_assembled": 0,
            "assembly_errors": 0,
            "cache_hits": 0
        }
        
        logger.info(f"DocumentAssembler initialisé avec index: {self.index_path}")
    
    @property
    def index(self):
        """Lazy loading of the Whoosh index"""
        if self._index is None:
            self._load_index()
        return self._index
    
    def _load_index(self):
        """Load the Whoosh index"""
        try:
            if not Path(self.index_path).exists():
                raise FileNotFoundError(f"Index Whoosh not found: {self.index_path}")
            
            self._index = open_dir(self.index_path)
            self._stats["index_loaded"] = True
            
            with self._index.searcher() as searcher:
                doc_count = searcher.doc_count()
                logger.info(f"Index Whoosh loaded: {doc_count} documents from {self.index_path}")
                
        except Exception as e:
            logger.error(f"Error loading Whoosh index: {e}")
            raise
    
    def assemble_documents(self, 
                          candidates: List,  # FusionCandidate objects
                          top_k: int = 5,
                          include_code: bool = True,
                          include_cpg: bool = True) -> List[EnrichedDocument]:
        """
        Assemble the enriched documents from the Whoosh index
        
        Args:
            candidates: List of FusionCandidate objects with RRF scores
            top_k: Number of documents to assemble
            include_code: Include the source code (always True for Whoosh)
            include_cpg: Include CPG data (always True for Whoosh)
            
        Returns:
            List of EnrichedDocument objects sorted by final score
        """
        if not candidates:
            logger.warning("No candidates provided for assembly")
            return []
        
        # Extraire les clés des candidats
        candidate_keys = [candidate.key for candidate in candidates[:top_k]]
        
        # Récupérer les documents depuis Whoosh en batch
        whoosh_docs = self._batch_retrieve_from_whoosh(candidate_keys)
        
        # Assembler les documents enrichis
        enriched_docs = []
        assembly_errors = 0
        
        for candidate in candidates[:top_k]:
            try:
                whoosh_doc = whoosh_docs.get(candidate.key)
                if whoosh_doc:
                    doc = self._assemble_from_whoosh_doc(candidate, whoosh_doc)
                    if doc:
                        enriched_docs.append(doc)
                        self._stats["documents_assembled"] += 1
                    else:
                        assembly_errors += 1
                else:
                    logger.warning(f"Document {candidate.key} non trouvé dans index Whoosh")
                    assembly_errors += 1
                    
            except Exception as e:
                logger.warning(f"Erreur assemblage {candidate.key}: {e}")
                assembly_errors += 1
                continue
        
        self._stats["assembly_errors"] += assembly_errors
        
        if assembly_errors > 0:
            logger.warning(f"{assembly_errors} erreurs d'assemblage sur {len(candidates[:top_k])}")
        
        logger.info(f"Documents assemblés: {len(enriched_docs)}/{top_k} depuis index Whoosh")
        return enriched_docs
    
    def _batch_retrieve_from_whoosh(self, keys: List[str]) -> Dict[str, Dict]:
        """Retrieve multiple documents from Whoosh in a single query"""
        if not keys:
            return {}
        
        try:
            with self.index.searcher() as searcher:
                # Construire requête OR pour toutes les clés
                key_terms = [Term("key", key) for key in keys]
                query = Or(key_terms)
                
                # Exécuter la requête
                results = searcher.search(query, limit=len(keys))
                
                # Construire le dictionnaire de résultats
                docs = {}
                for hit in results:
                    key = hit["key"]
                    # Convert hit Whoosh to dictionary
                    doc_data = dict(hit)
                    docs[key] = doc_data
                
                logger.info(f"Retrieved {len(docs)}/{len(keys)} documents from Whoosh")
                return docs
                
        except Exception as e:
            logger.error(f"Error during Whoosh batch retrieval: {e}")
            return {}
    
    def _assemble_from_whoosh_doc(self, candidate, whoosh_doc: Dict) -> Optional[EnrichedDocument]:
        """Assemble an EnrichedDocument from a Whoosh document"""
        try:
            # Build results dictionary
            doc_data = {
                "key": whoosh_doc.get("key", ""),
                "cwe": whoosh_doc.get("cwe", ""),
                "gpt_purpose": whoosh_doc.get("gpt_purpose", ""),
                "gpt_function": whoosh_doc.get("gpt_function", ""),
                "dangerous_functions": whoosh_doc.get("dangerous_functions", []),
                "dangerous_functions_count": whoosh_doc.get("dangerous_functions_count", 0),
                "risk_class": whoosh_doc.get("risk_class", "unknown"),
                "embedding_text": whoosh_doc.get("embedding_text", ""),
                "code_before_change": whoosh_doc.get("code_before_change", ""),
                "code_after_change": whoosh_doc.get("code_after_change", ""),
                "cpg_vulnerability_pattern": whoosh_doc.get("cpg_vulnerability_pattern", ""),
                "patch_transformation_analysis": whoosh_doc.get("patch_transformation_analysis", ""),
                "final_score": candidate.final_score,
            }
            
            # Extract fields with secure default values
            try:
                doc_data["dangerous_functions"] = self._parse_json_field(
                    whoosh_doc.get("dangerous_functions", "[]"), default=[]
                )
            except Exception:
                doc_data["dangerous_functions"] = []
            
            # Patch analysis from KB2
            try:
                patch_analysis = self._parse_json_field(
                    whoosh_doc.get("patch_analysis", "{}"), default={}
                )
                doc_data["patch_analysis"] = patch_analysis
            except Exception:
                doc_data["patch_analysis"] = {}
            
            # Parsing JSON fields stored in Whoosh
            doc_data["dangerous_functions_added"] = self._parse_json_field(
                whoosh_doc.get("dangerous_functions_added", ""), default=[]
            )
            doc_data["dangerous_functions_removed"] = self._parse_json_field(
                whoosh_doc.get("dangerous_functions_removed", ""), default=[]
            )
            doc_data["vulnerability_behavior"] = self._parse_json_field(
                whoosh_doc.get("vulnerability_behavior", ""), default={}
            )
            doc_data["modified_lines"] = self._parse_json_field(
                whoosh_doc.get("modified_lines", ""), default={}
            )
            
            # Provenance information
            doc_data["provenance"] = {
                "kb1_rank": getattr(candidate, 'kb1_rank', None),
                "kb2_rank": getattr(candidate, 'kb2_rank', None), 
                "kb3_rank": getattr(candidate, 'kb3_rank', None),
                "kb1_score": getattr(candidate, 'kb1_score', None),
                "kb2_score": getattr(candidate, 'kb2_score', None),
                "kb3_score": getattr(candidate, 'kb3_score', None),
                "search_time_ms": getattr(candidate, 'search_time_ms', 0.0),
                "total_sources": getattr(candidate, 'total_sources', 0),
                "source": "whoosh_enriched_index"
            }
            
            # Create the enriched document
            doc = EnrichedDocument(**doc_data)
            return doc
            
        except Exception as e:
            logger.error(f"Error assembling document {candidate.key}: {e}")
            return None
    
    def _parse_json_field(self, json_str: str, default=None):
        """Parse a JSON string field into a Python object"""
        if not json_str or json_str == "":
            return default if default is not None else {}
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Error parsing JSON: {json_str[:50]}...")
            return default if default is not None else {}
        except Exception as e:
            logger.warning(f"Error parsing JSON: {e}")
            return default if default is not None else {}
    
    def get_document_by_key(self, key: str) -> Optional[EnrichedDocument]:
        """Retrieve a single document by its key"""
        try:
            with self.index.searcher() as searcher:
                query = Term("key", key)
                results = searcher.search(query, limit=1)
                
                if results:
                    # Create a fake candidate for assembly
                    from dataclasses import dataclass
                    
                    @dataclass
                    class MockCandidate:
                        key: str
                        final_score: float = 1.0
                        kb1_rank: int = 1
                        kb2_rank: Optional[int] = None
                        kb3_rank: Optional[int] = None
                        kb1_score: float = 1.0
                        kb2_score: Optional[float] = None
                        kb3_score: Optional[float] = None
                        search_time_ms: float = 0.0
                        total_sources: int = 1
                    
                    mock_candidate = MockCandidate(key=key)
                    doc_data = dict(results[0])
                    
                    return self._assemble_from_whoosh_doc(mock_candidate, doc_data)
                else:
                    logger.warning(f"Document {key} not found in index")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving document {key}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Return assembly statistics"""
        return self._stats.copy()