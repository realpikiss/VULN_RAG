import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from retrieval.search_kb2_faiss import get_kb2_structure_searcher

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FAISS search KB2")
    parser.add_argument("--vectorfile", type=str, required=True, help="JSON file with embedding vector")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.vectorfile, "r", encoding="utf-8") as f:
        data = json.load(f)
        vec = data["embedding"] if isinstance(data, dict) and "embedding" in data else data

    searcher = get_kb2_structure_searcher()
    results, _ = searcher.search(vec, top_k=args.topk, verbose=args.verbose)

    with open("kb2_faiss_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("✅ Results saved in kb2_faiss_results.json")