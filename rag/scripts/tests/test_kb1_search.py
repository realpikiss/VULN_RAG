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
    """Test du searcher KB1 avec différentes stratégies"""
    print("🧪 Test KB1 Search - Diagnostic Complet")
    print("=" * 50)
    
    # Vérifier l'index
    index_path = os.getenv("KB1_INDEX_PATH", "data/KBs/kb1_index")
    print(f"📁 Index path: {index_path}")
    
    if not Path(index_path).exists():
        print(f"❌ Index not found at {index_path}")
        print("   Vous devez d'abord créer l'index avec migrate_kb1_to_whoosh.py")
        return
    else:
        print(f"✅ Index found at {index_path}")
    
    # Test 1: Recherche simple avec mots-clés courts
    print(f"\n🔍 Test 1 - Recherche simple (mots-clés courts)")
    simple_results = search_kb1_(
        purpose_text="network status update, thread dispatching, memory management, cellular data handling",
        function_text="string copy",
        top_k=5
    )
    
    print(f"📊 Résultats: {len(simple_results)}")
    for i, doc in enumerate(simple_results[:3], 1):
        print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
        if 'cwe' in doc:
            print(f"   CWE: {doc['cwe']}")
        if 'gpt_purpose' in doc:
            print(f"   Purpose: {doc['gpt_purpose'][:80]}...")
    
    # Test 2: Recherche avec mots-clés réseau (adapté à votre exemple)
    print(f"\n🔍 Test 2 - Recherche réseau")
    network_results = search_kb1_(
        purpose_text="network state thread",
        function_text="system network info",
        top_k=5
    )
    
    print(f"📊 Résultats: {len(network_results)}")
    for i, doc in enumerate(network_results[:3], 1):
        print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
        if 'gpt_purpose' in doc:
            print(f"   Purpose: {doc['gpt_purpose'][:80]}...")
    
    # Test 3: Recherche générale
    print(f"\n🔍 Test 3 - Recherche générale")
    general_results = search_kb1_(
        purpose_text="vulnerability",
        top_k=5
    )
    
    print(f"📊 Résultats: {len(general_results)}")
    for i, doc in enumerate(general_results[:3], 1):
        print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
        if 'gpt_purpose' in doc:
            print(f"   Purpose: {doc['gpt_purpose'][:80]}...")
    
    # Test 4: Recherche par CWE spécifique
    print(f"\n🔍 Test 4 - Recherche par CWE")
    try:
        cwe_results = search_kb1_(
            purpose_text="",
            function_text="",
            cwe_filter="CWE-119",
            top_k=3
        )
        
        print(f"📊 Résultats CWE-119: {len(cwe_results)}")
        for i, doc in enumerate(cwe_results[:2], 1):
            print(f"{i}. {doc['key']} (score: {doc.get('score', 0):.3f})")
    except Exception as e:
        print(f"⚠️ Test CWE échoué (normal si ancien index): {e}")
    
    # Diagnostic de l'index
    print(f"\n🔍 Diagnostic Index")
    try:
        from whoosh.index import open_dir
        ix = open_dir(index_path)
        
        print(f"📊 Schema fields: {list(ix.schema.names())}")
        
        with ix.searcher() as searcher:
            doc_count = searcher.doc_count()
            print(f"📊 Total documents in index: {doc_count}")
            
            # Échantillon de documents
            print(f"📋 Échantillon de documents:")
            for i, doc in enumerate(searcher.documents(), 1):
                if i <= 3:
                    print(f"  {i}. Key: {doc.get('key', 'N/A')}")
                    print(f"     Purpose: {doc.get('gpt_purpose', 'N/A')[:60]}...")
                else:
                    break
                    
    except Exception as e:
        print(f"❌ Erreur diagnostic: {e}")
    
    # Recommandations
    print(f"\n💡 Recommandations:")
    
    if not simple_results and not network_results and not general_results:
        print("❌ Aucun résultat trouvé. Problèmes possibles:")
        print("  1. Index vide ou corrompu")
        print("  2. Schema incompatible")
        print("  3. Données mal indexées")
        print("\n🔧 Actions suggérées:")
        print("  1. Recréer l'index avec migrate_kb1_to_whoosh.py")
        print("  2. Vérifier les variables d'environnement KB1_PATH et KB1_INDEX_PATH")
        print("  3. Vérifier que le fichier KB1 JSON existe et contient des données")
    else:
        print("✅ Recherche fonctionnelle !")
        print("💡 Pour de meilleurs résultats:")
        print("  - Utilisez des mots-clés courts et précis")
        print("  - Combinez purpose_text et function_text")
        print("  - Testez différentes variantes de mots-clés")

def test_with_debug():
    """Test avec debug détaillé"""
    print("\n🐛 TEST DEBUG")
    print("-" * 30)
    
    # Test très simple
    try:
        results = search_kb1_(
            purpose_text="buffer",
            top_k=1
        )
        print(f"Debug - Résultats pour 'buffer': {len(results)}")
        
        if results:
            doc = results[0]
            print("Debug - Champs sélectionnés:\n")
            fields = [
                "key", "score", "cwe", "risk_class", "complexity_class", "dangerous_functions_count",
                "gpt_purpose", "gpt_function", "solution", "dangerous_functions", "code_after_change"
            ]
            for field in fields:
                value = doc.get(field, "N/A")
                print(f"{field:15}: {value}")
            
    except Exception as e:
        print(f"Debug - Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    test_kb1_search()
    test_with_debug()