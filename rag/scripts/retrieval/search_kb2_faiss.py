import os
import json
import faiss
import numpy as np
import logging
from functools import lru_cache
from typing import List, Dict, Tuple, Union

# === Configuration ===
KB2_INDEX_PATH = os.getenv("KB2_INDEX_PATH")
KB2_METADATA_PATH = os.getenv("KB2_METADATA_PATH")

# === Logger ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KB2StructureFaissSearcher:
    def __init__(self, index_path: str = KB2_INDEX_PATH, metadata_path: str = KB2_METADATA_PATH):
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.metadata = []

        self._load_index()
        self._load_metadata()

    def _load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index FAISS introuvable: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        logger.info(f"📦 Index FAISS chargé: {self.index_path}")

    def _load_metadata(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Fichier metadata introuvable: {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"📁 {len(self.metadata)} métadonnées chargées")

    def search(self, embedding_vector: Union[np.ndarray, List[float]], top_k: int = 3, verbose: bool = False) -> Tuple[List[Dict], float]:
        if isinstance(embedding_vector, list):
            vec = np.asarray(embedding_vector, dtype='float32').reshape(1, -1)
        else:
            vec = embedding_vector.astype('float32')
            if vec.ndim == 1:
                vec = vec.reshape(1, -1)

        faiss.normalize_L2(vec)
        D, I = self.index.search(vec, top_k)

        results = []
        for rank, idx in enumerate(I[0]):
            if idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            result = {
                "key": meta.get("key"),
                "score": float(D[0][rank]),
                "rank": rank + 1,
                "cwe": meta.get("cwe"),
                "cve_id": meta.get("cve_id"),
                "preview": meta.get("embedding_text", "")[:200]
            }
            results.append(result)
            if verbose:
                print(f"[{rank+1}] Key: {result['key']}, Score: {result['score']:.4f}, CWE: {result['cwe']}")
                print(f"     Preview: {result['preview']}")
                print("-")

        best_score = float(D[0][0]) if len(D[0]) > 0 else 0.0
        if not results:
            logger.warning("KB2: no results found. Consider querying KB1 or KB3 as fallback.")
        return results, best_score

# Singleton pour usage pipeline
@lru_cache(maxsize=1)
def get_kb2_structure_searcher(index_path=None, metadata_path=None) -> KB2StructureFaissSearcher:
    return KB2StructureFaissSearcher(
        index_path or KB2_INDEX_PATH,
        metadata_path or KB2_METADATA_PATH
    )

