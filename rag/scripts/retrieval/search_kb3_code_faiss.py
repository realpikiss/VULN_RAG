# search_kb3_code_faiss.py

import os
import json
import faiss
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
    """Return a cached singleton instance of KB3CodeFaissSearcher."""
    return KB3CodeFaissSearcher(
        index_path or KB3_INDEX_PATH,
        metadata_path or KB3_METADATA_PATH,
        model_name or EMBEDDING_MODEL,
    )

class KB3CodeFaissSearcher:
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
            raise FileNotFoundError(f"FAISS index not found: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        logger.info(f"📦 FAISS index loaded: {self.index_path}")

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
        faiss.normalize_L2(vec)
        return vec.astype('float32')

    def search(self, code_snippet: str, top_k: int = 3, verbose: bool = False) -> Tuple[List[Dict], float]:
        vec = self.encode_code(code_snippet)
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
                "lines": meta.get("lines"),
                "cwe": meta.get("cwe"),
                "cve_id": meta.get("cve_id"),
                "chars": meta.get("chars")
            }
            results.append(result)
            if verbose:
                print(f"[{rank+1}] Key: {result['key']}, Score: {result['score']:.4f}, CWE: {result['cwe']}, Lines: {result['lines']}")
                print("-")

        return results, float(D[0][0]) if len(D[0]) > 0 else 0.0

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FAISS search on raw code KB3")
    parser.add_argument("--codefile", type=str, help="File containing source code")
    parser.add_argument("--topk", type=int, default=3, help="Number of results to return")
    parser.add_argument("--verbose", action="store_true", help="Display results")
    args = parser.parse_args()

    if not args.codefile or not os.path.exists(args.codefile):
        raise FileNotFoundError("Code file not found")

    with open(args.codefile, "r", encoding="utf-8") as f:
        code_input = f.read()

    searcher = KB3CodeFaissSearcher()
    results, _ = searcher.search(code_input, top_k=args.topk, verbose=args.verbose)

    with open("kb3_code_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved in 'kb3_code_results.json'")
