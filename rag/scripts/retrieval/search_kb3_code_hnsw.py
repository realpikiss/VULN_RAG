# search_kb3_code_hnsw.py

import os
import json
import hnswlib
import numpy as np
import logging
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# === Configuration ===
KB3_INDEX_PATH = os.getenv("KB3_INDEX_PATH")
KB3_METADATA_PATH = os.getenv("KB3_METADATA_PATH")
EMBEDDING_MODEL = os.getenv("KB3_MODEL")

# === Logger ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KB3-Code-Search")

@lru_cache(maxsize=1)
def get_kb3_searcher(index_path: str = None, metadata_path: str = None, model_name: str = None):
    """Return a cached singleton instance of KB3CodeHnswSearcher."""
    return KB3CodeHnswSearcher(
        index_path or KB3_INDEX_PATH,
        metadata_path or KB3_METADATA_PATH,
        model_name or EMBEDDING_MODEL,
    )

class KB3CodeHnswSearcher:
    def __init__(self,
                 index_path: str = KB3_INDEX_PATH,
                 metadata_path: str = KB3_METADATA_PATH,
                 model_name: str = EMBEDDING_MODEL):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        self.index = None
        self.metadata = []
        self.model = None

        self._load_index()
        self._load_metadata()
        self._load_model()

    def _load_index(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"HNSW index not found: {self.index_path}")
        
        # Charger l'index HNSW
        self.index = hnswlib.Index(space='cosine', dim=384)  # Dimension par défaut
        self.index.load_index(self.index_path)
        logger.info(f"📦 HNSW index loaded: {self.index_path}")

    def _load_metadata(self):
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"📁 {len(self.metadata)} metadata loaded")

    def _load_model(self):
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"🧠 Encoding model loaded: {self.model_name}")

    def encode_code(self, code_text: str) -> np.ndarray:
        vec = self.model.encode([code_text])
        # Normalisation L2
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        return vec.astype('float32')

    def search(self, code_snippet: str, top_k: int = 3, verbose: bool = False) -> Tuple[List[Dict], float]:
        vec = self.encode_code(code_snippet)
        labels, distances = self.index.knn_query(vec, k=top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(labels[0], distances[0])):
            if idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            # Convertir distance en score de similarité
            score = 1.0 - dist
            result = {
                "key": meta.get("key"),
                "score": float(score),
                "rank": rank + 1,
                "lines": meta.get("lines"),
                "cwe": meta.get("cwe"),
                "cve_id": meta.get("cve_id"),
                "chars": meta.get("chars")
            }
            results.append(result)
            if verbose:
                print(f"[{rank+1}] Key: {result['key']}, Score: {result['score']:.4f}, CWE: {result['cwe']}, Lines: {result['lines']}")
                print("-")

        return results, float(1.0 - distances[0][0]) if len(distances[0]) > 0 else 0.0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HNSW search on raw code KB3")
    parser.add_argument("--codefile", type=str, help="File containing source code")
    parser.add_argument("--topk", type=int, default=3, help="Number of results to return")
    parser.add_argument("--verbose", action="store_true", help="Display results")
    args = parser.parse_args()

    if not args.codefile or not os.path.exists(args.codefile):
        raise FileNotFoundError("Code file not found")

    with open(args.codefile, "r", encoding="utf-8") as f:
        code_input = f.read()

    searcher = KB3CodeHnswSearcher()
    results, _ = searcher.search(code_input, top_k=args.topk, verbose=args.verbose)

    with open("kb3_code_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Results saved in 'kb3_code_results.json'")
