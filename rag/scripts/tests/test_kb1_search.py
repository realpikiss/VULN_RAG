# test_kb1_corrected.py

import sys
from pathlib import Path
import logging
import os

# Add the 'scripts' directory to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Vérifier que les variables d'environnement sont définies
if not os.getenv("KB1_INDEX_PATH"):
    os.environ["KB1_INDEX_PATH"] = "data/KBs/kb1_index"

try:
    # ✅ CORRECTION : Import du nouveau searcher enrichi
    from retrieval.search_kb1 import search_kb1_ as search_kb1_, search_kb1_by_keys, search_kb1_hybrid
    print("✅ Nouveau searcher enrichi importé avec succès")
except ImportError as e:
    # Fallback vers l'ancien searcher pour tester
    print("⚠️ Nouveau searcher non trouvé, fallback vers l'ancien")
    from retrieval.search_kb1 import search_kb1
    
    def search_kb1_(purpose_text="", function_text="", **kwargs):
        return search_kb1(purpose_text, function_text, kwargs.get('top_k', 10))
    
    def search_kb1_by_keys(keys, **kwargs):
        return []
    
    def search_kb1_hybrid(purpose_text="", **kwargs):
        return search_kb1(purpose_text, "", kwargs.get('top_k', 10))

def test_kb1_search():
    """Test KB1 searcher with different strategies"""
    print("🧪 KB1 Search Test - Complete Diagnostic")
    print("=" * 50)
    
    # Check index
    index_path = os.getenv("KB1_INDEX_PATH", "data/KBs/kb1_index")
    print(f"📁 Index path: {index_path}")
    
    if not Path(index_path).exists():
        print(f"❌ Index not found at {index_path}")
        print("   You must first create the index with migrate_kb1_to_whoosh.py")
        return
    else:
        print(f"✅ Index found at {index_path}")
    
    # Test 1: Buffer Overflow (CWE-119)
    print(f"\n🔍 Test 1 - Buffer Overflow (CWE-119)")
    buffer_results = search_kb1_(
        purpose_text="buffer overflow",
        top_k=5
    )
    
    print(f"📊 Results: {len(buffer_results)}")
    for i, doc in enumerate(buffer_results[:3], 1):
        print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
        if 'cwe' in doc:
            print(f"   CWE: {doc['cwe']}")
        if 'gpt_purpose' in doc:
            print(f"   Purpose: {doc['gpt_purpose'][:80]}...")
    
    # Test 2: Race Condition (CWE-362)
    print(f"\n🔍 Test 2 - Race Condition (CWE-362)")
    race_results = search_kb1_(
        purpose_text="race condition",
        top_k=5
    )
    
    print(f"📊 Results: {len(race_results)}")
    for i, doc in enumerate(race_results[:3], 1):
        print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
        if 'cwe' in doc:
            print(f"   CWE: {doc['cwe']}")
        if 'gpt_purpose' in doc:
            print(f"   Purpose: {doc['gpt_purpose'][:80]}...")
    
    # Test 3: Null Pointer (CWE-476)
    print(f"\n🔍 Test 3 - Null Pointer (CWE-476)")
    null_results = search_kb1_(
        purpose_text="null pointer",
        top_k=5
    )
    
    print(f"📊 Results: {len(null_results)}")
    for i, doc in enumerate(null_results[:3], 1):
        print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
        if 'cwe' in doc:
            print(f"   CWE: {doc['cwe']}")
        if 'gpt_purpose' in doc:
            print(f"   Purpose: {doc['gpt_purpose'][:80]}...")
    
    # Test 4: Out-of-bounds Write (CWE-787)
    print(f"\n🔍 Test 4 - Out-of-bounds Write (CWE-787)")
    bounds_results = search_kb1_(
        purpose_text="out of bounds write",
        top_k=5
    )
    
    print(f"📊 Results: {len(bounds_results)}")
    for i, doc in enumerate(bounds_results[:3], 1):
        print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
        if 'cwe' in doc:
            print(f"   CWE: {doc['cwe']}")
        if 'gpt_purpose' in doc:
            print(f"   Purpose: {doc['gpt_purpose'][:80]}...")
    
    # Test 5: Specific CWE search (CWE-416 - Use After Free)
    print(f"\n🔍 Test 5 - Use After Free (CWE-416)")
    try:
        cwe_results = search_kb1_(
            purpose_text="",
            function_text="",
            cwe_filter="CWE-416",
            top_k=3
        )
        
        print(f"📊 CWE-416 Results: {len(cwe_results)}")
        for i, doc in enumerate(cwe_results[:2], 1):
            print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
            print(f"   Purpose: {doc.get('gpt_purpose', 'N/A')[:80]}...")
    except Exception as e:
        print(f"⚠️ CWE test failed: {e}")
    
    # Index diagnostic
    print(f"\n🔍 Index Diagnostic")
    try:
        from whoosh.index import open_dir
        ix = open_dir(index_path)
        
        print(f"📊 Schema fields: {list(ix.schema.names())}")
        
        with ix.searcher() as searcher:
            doc_count = searcher.doc_count()
            print(f"📊 Total documents in index: {doc_count}")
            
            # Document samples
            print(f"📋 Document samples:")
            for i, doc in enumerate(searcher.documents(), 1):
                if i <= 3:
                    print(f"  {i}. Key: {doc.get('key', 'N/A')}")
                    print(f"     Purpose: {doc.get('gpt_purpose', 'N/A')[:60]}...")
                else:
                    break
                    
    except Exception as e:
        print(f"❌ Diagnostic error: {e}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if not buffer_results and not race_results and not null_results and not bounds_results:
        print("❌ No results found. Possible issues:")
        print("  1. Empty or corrupted index")
        print("  2. Incompatible schema")
        print("  3. Poorly indexed data")
        print("\n🔧 Suggested actions:")
        print("  1. Recreate index with migrate_kb1_to_whoosh.py")
        print("  2. Check KB1_PATH and KB1_INDEX_PATH environment variables")
        print("  3. Verify that KB1 JSON file exists and contains data")
    else:
        print("✅ Search functional!")
        print("💡 Tests adapted to your 10 CWE base:")
        print("  - CWE-119: Buffer Overflow")
        print("  - CWE-362: Race Condition")
        print("  - CWE-476: Null Pointer")
        print("  - CWE-787: Out-of-bounds Write")
        print("  - CWE-416: Use After Free (most frequent)")

def test_with_debug():
    """Test with detailed debug"""
    print("\n🐛 TEST DEBUG")
    print("-" * 30)
    
    # Very simple test
    try:
        results = search_kb1_(
            purpose_text="buffer",
            top_k=1
        )
        print(f"Debug - Results for 'buffer': {len(results)}")
        
        if results:
            doc = results[0]
            print("Debug - Selected fields:\n")
            fields = [
                "key", "score", "cwe", "risk_class", "complexity_class", "dangerous_functions_count",
                "gpt_purpose", "gpt_function", "solution", "dangerous_functions", "code_after_change"
            ]
            for field in fields:
                value = doc.get(field, "N/A")
                print(f"{field:15}: {value}")
            
    except Exception as e:
        print(f"Debug - Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    test_kb1_search()
    test_with_debug()