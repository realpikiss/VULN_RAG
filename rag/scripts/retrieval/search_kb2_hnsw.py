import os
import json
import hnswlib
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

class KB2StructureHnswSearcher:
    def __init__(self, index_path: str = KB2_INDEX_PATH, metadata_path: str = KB2_METADATA_PATH):
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.index = None
        self.metadata = []

        self._load_index()
        self._load_metadata()

    def _load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index HNSW introuvable: {self.index_path}")
        
        # Charger l'index HNSW
        self.index = hnswlib.Index(space='cosine', dim=384)  # Dimension par défaut
        self.index.load_index(self.index_path)
        logger.info(f"📦 Index HNSW chargé: {self.index_path}")

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

        # Normalisation L2
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        
        # Recherche HNSW
        labels, distances = self.index.knn_query(vec, k=top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(labels[0], distances[0])):
            if idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            # Convertir distance en score de similarité
            score = 1.0 - dist  # HNSW retourne des distances, on les convertit en similarité
            result = {
                "key": meta.get("key"),
                "score": float(score),
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

        best_score = float(1.0 - distances[0][0]) if len(distances[0]) > 0 else 0.0
        if not results:
            logger.warning("KB2: no results found. Consider querying KB1 or KB3 as fallback.")
        return results, best_score

@lru_cache(maxsize=1)
def get_kb2_structure_searcher(index_path: str = None, metadata_path: str = None):
    """Return a cached singleton instance of KB2StructureHnswSearcher."""
    return KB2StructureHnswSearcher(
        index_path or KB2_INDEX_PATH,
        metadata_path or KB2_METADATA_PATH,
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HNSW search on KB2 structure embeddings")
    parser.add_argument("--vectorfile", type=str, required=True, help="JSON file with embedding vector")
    parser.add_argument("--topk", type=int, default=3, help="Number of results to return")
    parser.add_argument("--verbose", action="store_true", help="Display results")
    args = parser.parse_args()

    with open(args.vectorfile, "r", encoding="utf-8") as f:
        data = json.load(f)
        vec = data["embedding"] if isinstance(data, dict) and "embedding" in data else data

    searcher = get_kb2_structure_searcher()
    results, _ = searcher.search(vec, top_k=args.topk, verbose=args.verbose)

    with open("kb2_hnsw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✅ Results saved in 'kb2_hnsw_results.json'")

