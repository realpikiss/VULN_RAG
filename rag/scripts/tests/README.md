# Test Utilities

Quick-and-dirty CLI test scripts that validate the behaviour and performance of
each knowledge-base searcher.

| Script | What it tests |
| ------ | ------------- |
| **`test_kb1_search.py`** | End-to-end smoke test for the *Whoosh* KB1 searcher. Performs multiple queries (simple, network, general, CWE filter) and prints diagnostics + recommended troubleshooting steps. Includes a *debug* mode that prints selected fields of the top document. |
| **`test_kb2_search.py`** | Calls the KB2 HNSW searcher using a sample embedding (`sample_vector.json`) and prints the top-k structural matches. Demonstrates the singleton cache. |
| **`test_kb3_search.py`** | Encodes a tiny C snippet (`test_snippet.c`) with the KB3 code searcher and prints similarity results (key, score, CWE). |
| **`sample_vector.json`** | 384-dimensional embedding used by `test_kb2_search.py`. |
| **`test_snippet.c`** | Small C programme used by `test_kb3_search.py` as a query example. |

## Running All Tests

Activate the project virtual-environment and simply run:

```bash
python rag/scripts/tests/test_kb1_search.py
python rag/scripts/tests/test_kb2_search.py
python rag/scripts/tests/test_kb3_search.py
```

Each script is fully self-contained and prints helpful troubleshooting tips if
the corresponding index or model is missing.
