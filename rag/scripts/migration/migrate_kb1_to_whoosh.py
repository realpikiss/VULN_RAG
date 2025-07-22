# migrate_kb1_to_whoosh_.py

import json
import os
import logging
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer
from whoosh.index import create_in
from pathlib import Path

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("KB1-Migration")

# Load environment variables from .env file
def load_env():
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

load_env()

# Input/output parameters
KB1_JSON_PATH = Path(os.getenv("KB1_PATH"))
KB2_JSON_PATH = Path(os.getenv("KB2_PATH"))  
INDEX_DIR = Path(os.getenv("KB1_INDEX_PATH"))

# 🧱 Schema enrichi utilisant données KB1 + KB2 existantes
schema = Schema(
    # Identifiants
    key=ID(stored=True, unique=True),
    cve_id=ID(stored=True),
    cwe=ID(stored=True),
    
    # Champs textuels KB1 (pour recherche)
    gpt_purpose=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    gpt_function=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    gpt_analysis=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    solution=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    
    # Code source KB1 (analysé mais avec analyzer plus simple)
    code_before_change=TEXT(stored=True, analyzer=StandardAnalyzer()),
    code_after_change=TEXT(stored=True, analyzer=StandardAnalyzer()),
    
    # Analyses CPG KB1 (analysées)
    cpg_vulnerability_pattern=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    patch_transformation_analysis=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    detection_cpg_signature=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    remediation_graph_guidance=TEXT(stored=True, analyzer=StemmingAnalyzer()),
    
    # Champs structurés KB1 (stockés comme JSON strings)
    vulnerability_behavior=STORED(),  # Dict converti en JSON
    modified_lines=STORED(),          # Dict converti en JSON
    
    # Données CPG depuis KB2 (évite recalcul)
    dangerous_functions=STORED(),           # Liste depuis KB2
    dangerous_functions_count=STORED(),     # Count depuis KB2
    dangerous_functions_detected=STORED(),  # Boolean depuis KB2
    risk_class=STORED(),                   # Classe de risque depuis KB2
    complexity_class=STORED(),             # Complexité depuis KB2
    embedding_text=TEXT(stored=True),      # Pour debug/explication
    
    # Métadonnées patch depuis KB2
    dangerous_functions_added=STORED(),     # Patch analysis
    dangerous_functions_removed=STORED(),   # Patch analysis
    net_dangerous_change=STORED(),         # Net change in dangerous functions
    
    # Métadonnées calculées simples
    code_lines_count=STORED(),             # Nombre de lignes du code
    has_code_before=STORED(),              # Boolean si code_before existe
    has_code_after=STORED()                # Boolean si code_after existe
)

def extract_cwe_from_key(key: str) -> str:
    """Extrait le CWE de la clé"""
    return key.split("_")[0] if "_" in key else "Unknown"

def count_code_lines(code: str) -> int:
    """Compte les lignes de code non vides"""
    if not code:
        return 0
    return len([line for line in code.splitlines() if line.strip()])

def safe_json_dumps(obj) -> str:
    """Convertit un objet en JSON string de manière sûre"""
    if obj is None:
        return ""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def extract_and_clean_field(entry: dict, field: str, default: str = "") -> str:
    """Extrait et nettoie un champ textuel"""
    value = entry.get(field, default)
    if value is None:
        return default
    return str(value).strip()

def extract_kb2_dangerous_functions(kb2_entry: dict) -> tuple:
    """Extrait les données de fonctions dangereuses depuis KB2"""
    cpg_features = kb2_entry.get("cpg_features", {})
    
    # Fonctions dangereuses (différents chemins possibles)
    dangerous_funcs = []
    if "dangerous_functions" in cpg_features:
        dangerous_funcs = cpg_features["dangerous_functions"]
    elif "vuln_signature" in cpg_features and "dangerous_functions" in cpg_features["vuln_signature"]:
        dangerous_funcs = cpg_features["vuln_signature"]["dangerous_functions"]
    elif "patch_signature" in cpg_features and "dangerous_functions" in cpg_features["patch_signature"]:
        dangerous_funcs = cpg_features["patch_signature"]["dangerous_functions"]
    
    # Count
    dangerous_count = 0
    if "dangerous_count" in cpg_features:
        dangerous_count = cpg_features["dangerous_count"]
    elif "vuln_signature" in cpg_features and "dangerous_count" in cpg_features["vuln_signature"]:
        dangerous_count = cpg_features["vuln_signature"]["dangerous_count"]
    
    # Detected boolean
    dangerous_detected = bool(dangerous_funcs) or cpg_features.get("dangerous_functions_detected", False)
    
    return dangerous_funcs, dangerous_count, dangerous_detected

def extract_kb2_risk_metrics(kb2_entry: dict) -> tuple:
    """Extrait les métriques de risque depuis KB2"""
    cpg_features = kb2_entry.get("cpg_features", {})
    
    # Risk class
    risk_class = "unknown"
    if "risk_class" in cpg_features:
        risk_class = cpg_features["risk_class"]
    elif "vuln_signature" in cpg_features and "risk_class" in cpg_features["vuln_signature"]:
        risk_class = cpg_features["vuln_signature"]["risk_class"]
    elif "patch_signature" in cpg_features and "risk_class" in cpg_features["patch_signature"]:
        risk_class = cpg_features["patch_signature"]["risk_class"]
    
    # Complexity class
    complexity_class = "unknown"
    if "complexity_class" in cpg_features:
        complexity_class = cpg_features["complexity_class"]
    elif "vuln_signature" in cpg_features and "complexity_class" in cpg_features["vuln_signature"]:
        complexity_class = cpg_features["vuln_signature"]["complexity_class"]
    elif "patch_signature" in cpg_features and "complexity_class" in cpg_features["patch_signature"]:
        complexity_class = cpg_features["patch_signature"]["complexity_class"]
    
    return risk_class, complexity_class

def extract_kb2_patch_analysis(kb2_entry: dict) -> tuple:
    """Extrait l'analyse de patch depuis KB2"""
    patch_analysis = kb2_entry.get("patch_analysis", {})
    
    dangerous_added = patch_analysis.get("dangerous_functions_added", [])
    dangerous_removed = patch_analysis.get("dangerous_functions_removed", [])
    net_change = patch_analysis.get("net_dangerous_change", 0)
    
    return dangerous_added, dangerous_removed, net_change

# 📦 Creating or opening the index
    logger.info(f"Creating Whoosh index at {INDEX_DIR}")
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

# Supprimer l'index existant s'il existe pour recréer avec nouveau schema
if index.exists_in(INDEX_DIR):
    import shutil
    shutil.rmtree(INDEX_DIR)
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

ix = create_in(INDEX_DIR, schema)

# 📤 Loading KB1
if not KB1_JSON_PATH.exists():
    raise FileNotFoundError(f"KB1 file not found: {KB1_JSON_PATH}")

with open(KB1_JSON_PATH, "r", encoding="utf-8") as f:
    kb1_entries = json.load(f)

    logger.info(f"→ Loaded {len(kb1_entries)} entries from KB1: {KB1_JSON_PATH}")

# 📤 Loading KB2 (pour enrichissement)
kb2_entries = {}
if KB2_JSON_PATH and KB2_JSON_PATH.exists():
    with open(KB2_JSON_PATH, "r", encoding="utf-8") as f:
        kb2_data = json.load(f)
        # Si KB2 est une liste de dict avec key
        if isinstance(kb2_data, list):
            kb2_entries = {item["key"]: item for item in kb2_data if "key" in item}
        # Si KB2 est un dict avec key comme clé
        elif isinstance(kb2_data, dict):
            kb2_entries = kb2_data
    logger.info(f"→ Loaded {len(kb2_entries)} entries from KB2: {KB2_JSON_PATH}")
else:
    logger.warning("⚠️  KB2 not found, proceeding without CPG enrichment")

# 🧾 Indexing avec tous les champs enrichis
with ix.writer() as writer:
    indexed = 0
    errors = 0
    kb2_ = 0
    
    for key, kb1_entry in kb1_entries.items():
        try:
            # Récupérer l'entrée KB2 correspondante
            kb2_entry = kb2_entries.get(str(key), {})
            if kb2_entry:
                kb2_ += 1
            
            # Extraction des champs textuels KB1
            gpt_purpose = extract_and_clean_field(kb1_entry, "GPT_purpose")
            gpt_function = extract_and_clean_field(kb1_entry, "GPT_function") 
            gpt_analysis = extract_and_clean_field(kb1_entry, "GPT_analysis")
            solution = extract_and_clean_field(kb1_entry, "solution")
            
            # Code source KB1
            code_before = extract_and_clean_field(kb1_entry, "code_before_change")
            code_after = extract_and_clean_field(kb1_entry, "code_after_change")
            
            # Analyses CPG KB1
            cpg_pattern = extract_and_clean_field(kb1_entry, "cpg_vulnerability_pattern")
            patch_analysis = extract_and_clean_field(kb1_entry, "patch_transformation_analysis")
            detection_sig = extract_and_clean_field(kb1_entry, "detection_cpg_signature")
            remediation = extract_and_clean_field(kb1_entry, "remediation_graph_guidance")
            
            # Identifiants
            cve_id = extract_and_clean_field(kb1_entry, "CVE_id")
            cwe = extract_cwe_from_key(str(key))
            
            # Champs structurés KB1
            vuln_behavior = safe_json_dumps(kb1_entry.get("vulnerability_behavior"))
            modified_lines = safe_json_dumps(kb1_entry.get("modified_lines"))
            
            # Extraction des données CPG depuis KB2
            dangerous_funcs, dangerous_count, dangerous_detected = extract_kb2_dangerous_functions(kb2_entry)
            risk_class, complexity_class = extract_kb2_risk_metrics(kb2_entry)
            dangerous_added, dangerous_removed, net_change = extract_kb2_patch_analysis(kb2_entry)
            
            # Embedding text depuis KB2
            embedding_text = kb2_entry.get("embedding_text", "")
            
            # Métadonnées calculées simples
            code_lines = count_code_lines(code_before)
            has_code_before = bool(code_before.strip())
            has_code_after = bool(code_after.strip())
            
            # Ajouter le document à l'index
            writer.add_document(
                # Identifiants
                key=str(key),
                cve_id=cve_id,
                cwe=cwe,
                
                # Champs textuels KB1
                gpt_purpose=gpt_purpose,
                gpt_function=gpt_function,
                gpt_analysis=gpt_analysis,
                solution=solution,
                
                # Code KB1
                code_before_change=code_before,
                code_after_change=code_after,
                
                # Analyses CPG KB1
                cpg_vulnerability_pattern=cpg_pattern,
                patch_transformation_analysis=patch_analysis,
                detection_cpg_signature=detection_sig,
                remediation_graph_guidance=remediation,
                
                # Données structurées KB1
                vulnerability_behavior=vuln_behavior,
                modified_lines=modified_lines,
                
                # Données CPG depuis KB2 (pas recalculées !)
                dangerous_functions=safe_json_dumps(dangerous_funcs),
                dangerous_functions_count=dangerous_count,
                dangerous_functions_detected=dangerous_detected,
                risk_class=str(risk_class),
                complexity_class=str(complexity_class),
                embedding_text=embedding_text,
                
                # Analyse patch depuis KB2
                dangerous_functions_added=safe_json_dumps(dangerous_added),
                dangerous_functions_removed=safe_json_dumps(dangerous_removed),
                net_dangerous_change=net_change,
                
                # Métadonnées simples
                code_lines_count=code_lines,
                has_code_before=has_code_before,
                has_code_after=has_code_after
            )
            
            indexed += 1
            
            if indexed % 100 == 0:
                logger.info(f"   Processed {indexed} documents... (KB2 : {kb2_})")
                
        except Exception as e:
            logger.error(f"[⚠️] Error with key {key}: {e}")
            errors += 1
            continue

logger.info(f"✅ {indexed} documents successfully indexed in Whoosh ({INDEX_DIR})")
logger.info(f"✅ {kb2_} documents  with KB2 data")
if errors > 0:
    logger.warning(f"⚠️ {errors} errors encountered during indexing")

# Statistiques de l'index créé
logger.info(f"\nIndex Statistics:")
logger.info(f"   Schema fields: {len(schema.names())} fields")
logger.info(f"   KB1 source documents: {len(kb1_entries)}")
logger.info(f"   KB2 enrichment coverage: {kb2_}/{len(kb1_entries)} ({100*kb2_/len(kb1_entries):.1f}%)")
logger.info(f"   Total indexed: {indexed}")

# Test de l'index créé
logger.info(f"\nTesting index...")
with ix.searcher() as searcher:
    # Test simple
    from whoosh.qparser import QueryParser
    parser = QueryParser("gpt_purpose", ix.schema)
    query = parser.parse("buffer overflow")
    results = searcher.search(query, limit=3)
    
    logger.info(f"   Test query 'buffer overflow' found {len(results)} results")
    for i, hit in enumerate(results[:2], 1):
        dangerous_count = hit.get('dangerous_functions_count', 0)
        risk_class = hit.get('risk_class', 'unknown')
        logger.info(f"   {i}. {hit['key']}")
        logger.info(f"      Purpose: {hit['gpt_purpose'][:80]}...")
        logger.info(f"      Dangerous functions: {dangerous_count}, Risk: {risk_class}")

logger.info(f"\n KB1 Whoosh index successfully created!")
logger.info(f"   Location: {INDEX_DIR}")
logger.info(f"   Features: KB1 text + KB2 CPG analysis + unified storage")
logger.info(f"   Ready for high-performance document retrieval!")