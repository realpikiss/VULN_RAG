import sys
from pathlib import Path
# Add 'scripts' directory to PYTHONPATH to import retrieval
sys.path.append(str(Path(__file__).resolve().parents[1]))
from retrieval.search_kb3_code_faiss import get_kb3_searcher

snippet_path = Path(__file__).resolve().parent / "test_snippet.c"
with open(snippet_path, "r", encoding="utf-8") as f:
    code = f.read()

searcher = get_kb3_searcher()
results, _ = searcher.search(code_snippet=code, top_k=5, verbose=True)

print("🔍 KB3 Results (HNSW on raw code)")
for res in results:
    print(f"[{res['rank']}] Key: {res['key']} | Score: {res['score']:.4f} | CWE: {res['cwe']}")
