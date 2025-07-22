#!/bin/bash
# Script de configuration automatique pour ComputeCanada Cedar

set -e  # Arrêter en cas d'erreur

echo "=== Configuration VulnRAG pour ComputeCanada Cedar ==="

# Vérifier que nous sommes sur Cedar
if [[ ! "$(hostname)" =~ cedar ]]; then
    echo "ATTENTION: Ce script est conçu pour ComputeCanada Cedar"
    echo "Hostname actuel: $(hostname)"
    read -p "Continuer quand même? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Configuration des variables
USERNAME=$(whoami)
PROJECT_DIR="/scratch/$USERNAME/vulnrag"
CACHE_DIR="/scratch/$USERNAME/vulnrag/huggingface"

echo "Utilisateur: $USERNAME"
echo "Répertoire projet: $PROJECT_DIR"
echo "Répertoire cache: $CACHE_DIR"

# Créer la structure des répertoires
echo "Création de la structure des répertoires..."
mkdir -p "$PROJECT_DIR"/{data,models,logs,results}
mkdir -p "$PROJECT_DIR"/kb1_index
mkdir -p "$PROJECT_DIR"/kb2_index
mkdir -p "$PROJECT_DIR"/kb3_index
mkdir -p "$CACHE_DIR"/{transformers,datasets}
mkdir -p "$PROJECT_DIR"/joern

# Charger les modules nécessaires
echo "Chargement des modules..."
module load python/3.9
module load gcc/9.3.0
module load llvm/12.0.0
module load java/11

# Vérifier que les modules sont chargés
echo "Vérification des modules..."
python --version
gcc --version
clang-tidy --version
java -version

# Créer l'environnement virtuel
echo "Création de l'environnement virtuel..."
cd "$PROJECT_DIR"
python -m venv venv
source venv/bin/activate

# Installer les dépendances
echo "Installation des dépendances Python..."
pip install --upgrade pip
pip install -r requirements.txt

# Installer les outils statiques
echo "Installation des outils statiques..."
pip install flawfinder semgrep

# Vérifier l'installation de cppcheck
if ! command -v cppcheck &> /dev/null; then
    echo "Installation de cppcheck..."
    conda install -c conda-forge cppcheck -y
fi

# Installation de Joern (CRITIQUE pour CPG)
echo "Installation de Joern..."
if ! command -v joern-parse &> /dev/null; then
    echo "Joern non trouvé, installation via coursier..."
    
    # Installer coursier
    curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-apple-darwin.gz | gzip -d > cs
    chmod +x cs
    ./cs setup
    
    # Installer Joern
    ./cs install joern
    
    # Ajouter au PATH
    export PATH="$HOME/.local/share/coursier/bin:$PATH"
    
    # Vérifier l'installation
    if command -v joern-parse &> /dev/null; then
        echo "✓ Joern installé avec succès"
    else
        echo "✗ Échec de l'installation de Joern"
        exit 1
    fi
else
    echo "✓ Joern déjà installé"
fi

# Créer le fichier .env
echo "Création du fichier .env..."
cat > .env << EOF
# Chemins des bases de connaissances
KB1_INDEX_PATH=$PROJECT_DIR/kb1_index
KB2_INDEX_PATH=$PROJECT_DIR/kb2_index
KB2_METADATA_PATH=$PROJECT_DIR/kb2_metadata.json
KB3_INDEX_PATH=$PROJECT_DIR/kb3_index
KB3_METADATA_PATH=$PROJECT_DIR/kb3_metadata.json

# Configuration Hugging Face
HF_HOME=$CACHE_DIR
TRANSFORMERS_CACHE=$CACHE_DIR/transformers
HF_DATASETS_CACHE=$CACHE_DIR/datasets

# Configuration pour le cluster
SLURM_JOB_ID=\$SLURM_JOB_ID
SLURM_TMPDIR=\$SLURM_TMPDIR

# Modèles Hugging Face
QWEN2_5_MODEL=Qwen/Qwen2.5-7B-Instruct
KIRITO_MODEL=Qwen/Qwen2.5-14B-Instruct

# Configuration Joern
JOERN_HOME=$PROJECT_DIR/joern
JAVA_HOME=\$JAVA_HOME
EOF

# Script de vérification des outils
echo "Création du script de vérification..."
cat > check_tools.sh << 'EOF'
#!/bin/bash
echo "=== Vérification des outils VulnRAG ==="

# Outils statiques
echo "1. Outils statiques:"
tools=("cppcheck" "clang-tidy" "flawfinder" "semgrep")
for tool in "${tools[@]}"; do
    if command -v $tool &> /dev/null; then
        echo "  ✓ $tool: $(which $tool)"
    else
        echo "  ✗ $tool: NON TROUVÉ"
    fi
done

# Joern (CRITIQUE pour CPG extraction):
echo "2. Joern (CPG extraction):"
if command -v joern-parse &> /dev/null && command -v joern-export &> /dev/null; then
    echo "  ✓ joern-parse: $(which joern-parse)"
    echo "  ✓ joern-export: $(which joern-export)"
else
    echo "  ✗ Joern: NON TROUVÉ - CRITIQUE pour CPG extraction"
fi

# Java
echo "3. Java (requis pour Joern):"
if command -v java &> /dev/null; then
    echo "  ✓ Java: $(java -version 2>&1 | head -1)"
else
    echo "  ✗ Java: NON TROUVÉ"
fi

# Python packages
echo "4. Python packages:"
python -c "import transformers; print('  ✓ transformers')" 2>/dev/null || echo "  ✗ transformers"
python -c "import torch; print('  ✓ torch')" 2>/dev/null || echo "  ✗ torch"
python -c "import sentence_transformers; print('  ✓ sentence_transformers')" 2>/dev/null || echo "  ✗ sentence_transformers"
python -c "import faiss; print('  ✓ faiss')" 2>/dev/null || echo "  ✗ faiss"
python -c "import whoosh; print('  ✓ whoosh')" 2>/dev/null || echo "  ✗ whoosh"

echo "=== Fin de vérification ==="
EOF

chmod +x check_tools.sh

# Tester l'accès aux modèles Hugging Face
echo "Test de l'accès aux modèles Hugging Face..."
python -c "
from transformers import AutoTokenizer
import os

# Test du tokenizer Qwen2.5
print('Test du tokenizer Qwen2.5...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
print('✓ Tokenizer Qwen2.5 chargé avec succès')

# Test du tokenizer Kirito
print('Test du tokenizer Kirito...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-14B-Instruct', trust_remote_code=True)
print('✓ Tokenizer Kirito chargé avec succès')

print('Tous les tests sont passés avec succès!')
"

# Test spécifique de Joern
echo "Test de Joern..."
echo "int main() { return 0; }" > test.c
if command -v joern-parse &> /dev/null; then
    joern-parse test.c -o test.cpg
    if [ -f test.cpg ]; then
        echo "✓ Joern fonctionne correctement"
        rm test.c test.cpg
    else
        echo "✗ Problème avec Joern"
    fi
else
    echo "✗ Joern non disponible"
fi

# Rendre les scripts exécutables
echo "Configuration des permissions des scripts..."
chmod +x scripts/cedar/*.sh

# Créer un script de test rapide
echo "Création d'un script de test rapide..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Script de test rapide pour vérifier la configuration"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test des imports principaux"""
    print("Test des imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    try:
        import faiss
        print(f"✓ FAISS")
    except ImportError as e:
        print(f"✗ FAISS: {e}")
        return False
    
    try:
        import whoosh
        print(f"✓ Whoosh")
    except ImportError as e:
        print(f"✗ Whoosh: {e}")
        return False
    
    return True

def test_environment():
    """Test de l'environnement"""
    print("\nTest de l'environnement...")
    
    # Variables d'environnement
    env_vars = [
        'HF_HOME',
        'TRANSFORMERS_CACHE',
        'HF_DATASETS_CACHE',
        'SLURM_JOB_ID',
        'SLURM_TMPDIR'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'Non défini')
        print(f"  {var}: {value}")
    
    # Répertoires
    dirs = [
        '/scratch',
        os.getenv('HF_HOME', ''),
        os.getenv('TRANSFORMERS_CACHE', '')
    ]
    
    for dir_path in dirs:
        if dir_path and Path(dir_path).exists():
            print(f"✓ Répertoire {dir_path} existe")
        else:
            print(f"✗ Répertoire {dir_path} n'existe pas")

def test_tools():
    """Test des outils externes"""
    print("\nTest des outils externes...")
    
    import subprocess
    
    tools = ['cppcheck', 'clang-tidy', 'flawfinder', 'semgrep']
    
    for tool in tools:
        try:
            result = subprocess.run([tool, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ {tool} disponible")
            else:
                print(f"✗ {tool} non disponible")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"✗ {tool} non trouvé")
    
    # Test spécifique de Joern
    try:
        result = subprocess.run(['joern-parse', '--help'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ joern-parse disponible")
        else:
            print("✗ joern-parse non disponible")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("✗ joern-parse non trouvé")

if __name__ == "__main__":
    print("=== Test de configuration VulnRAG ===\n")
    
    success = True
    success &= test_imports()
    test_environment()
    test_tools()
    
    if success:
        print("\n✓ Configuration réussie!")
        sys.exit(0)
    else:
        print("\n✗ Configuration échouée!")
        sys.exit(1)
EOF

chmod +x test_setup.py

echo ""
echo "=== Configuration terminée ==="
echo ""
echo "Prochaines étapes:"
echo "1. Vérifier les outils: ./check_tools.sh"
echo "2. Tester la configuration: python test_setup.py"
echo "3. Générer les index: python rag/scripts/migration/migrate_*.py"
echo "4. Lancer un test rapide: sbatch scripts/cedar/quick_test.sh"
echo "5. Lancer l'évaluation complète: sbatch scripts/cedar/evaluation_job.sh"
echo ""
echo "Répertoires créés:"
echo "  - Projet: $PROJECT_DIR"
echo "  - Cache HF: $CACHE_DIR"
echo "  - Logs: $PROJECT_DIR/logs"
echo ""
echo "N'oubliez pas de modifier les scripts SLURM avec votre nom d'utilisateur!" 