# 🖥️ Configuration VulnRAG pour ComputeCanada Cedar (Hugging Face)

## Prérequis

### 1. Accès au cluster
```bash
# Se connecter au cluster Cedar
ssh username@cedar.computecanada.ca

# Allouer des ressources
salloc --account=def-username --time=2:00:00 --mem=32G --cpus-per-task=8
```

### 2. Modules nécessaires
```bash
# Charger les modules requis
module load python/3.9
module load gcc/9.3.0
module load llvm/12.0.0  # Pour clang-tidy
module load cuda/11.4    # Si utilisation GPU
module load java/11      # Pour Joern
```

### 3. Vérification et installation des outils statiques

#### **Outils disponibles via modules**
```bash
# Vérifier les modules disponibles
module avail cppcheck
module avail flawfinder
module avail semgrep

# Charger si disponibles
module load cppcheck
module load flawfinder
module load semgrep
```

#### **Installation via conda/pip (sans droits admin)**
```bash
# Cppcheck (via conda)
conda install -c conda-forge cppcheck

# Flawfinder et Semgrep (via pip dans l'environnement virtuel)
pip install flawfinder semgrep
```

#### **Installation de Joern (CRITIQUE) - Sans droits admin**
```bash
# Joern nécessite Java 11+
module load java/11

# Vérifier Java
java -version

# Installation de Joern via coursier (dans le répertoire utilisateur)
cd ~
curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-apple-darwin.gz | gzip -d > cs
chmod +x cs
./cs setup

# Installer Joern dans le répertoire utilisateur
./cs install joern

# Vérifier l'installation
~/.local/share/coursier/bin/joern-parse --help
~/.local/share/coursier/bin/joern-export --help

# Ajouter au PATH pour cette session
export PATH="$HOME/.local/share/coursier/bin:$PATH"
```

#### **Vérification complète des outils**
```bash
# Script de vérification
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

# Joern (CRITIQUE pour CPG)
echo "2. Joern (CPG extraction):"
if command -v joern-parse &> /dev/null && command -v joern-export &> /dev/null; then
    echo "  ✓ joern-parse: $(which joern-parse)"
    echo "  ✓ joern-export: $(which joern-export)"
elif [ -f "$HOME/.local/share/coursier/bin/joern-parse" ]; then
    echo "  ✓ joern-parse: $HOME/.local/share/coursier/bin/joern-parse"
    echo "  ✓ joern-export: $HOME/.local/share/coursier/bin/joern-export"
    echo "  ⚠️  Joern installé mais pas dans PATH - ajouter: export PATH=\"\$HOME/.local/share/coursier/bin:\$PATH\""
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
./check_tools.sh
```

## 🖥️ Gestion des GPUs sur Cedar

### **Vérifier la disponibilité des GPUs**
```bash
# Voir les partitions GPU disponibles
sinfo -p gpu

# Voir les GPUs disponibles
squeue -p gpu

# Voir les spécifications des GPUs
nvidia-smi  # Si tu as accès à un nœud GPU
```

### **Demander des ressources GPU**
```bash
# Demander un nœud avec GPU
salloc --account=def-username --time=2:00:00 --mem=32G --cpus-per-task=8 --gres=gpu:1

# Ou spécifier un type de GPU particulier
salloc --account=def-username --time=2:00:00 --mem=32G --cpus-per-task=8 --gres=gpu:v100:1
salloc --account=def-username --time=2:00:00 --mem=32G --cpus-per-task=8 --gres=gpu:a100:1
```

### **Types de GPUs disponibles sur Cedar**
```bash
# V100 (16GB VRAM)
salloc --gres=gpu:v100:1

# A100 (40GB VRAM) - Plus puissant
salloc --gres=gpu:a100:1

# RTX 6000 (24GB VRAM)
salloc --gres=gpu:rtx6000:1
```

### **Vérifier l'accès GPU dans le job**
```bash
# Dans ton script SLURM ou session interactive
echo "GPUs disponibles: $CUDA_VISIBLE_DEVICES"
nvidia-smi
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Nombre de GPUs: {torch.cuda.device_count()}')"
```

## Configuration de l'environnement

### 1. Variables d'environnement
```bash
# Créer le fichier .env sur Cedar
cat > .env << EOF
# Chemins des bases de connaissances
KB1_INDEX_PATH=/scratch/username/vulnrag/kb1_index
KB2_INDEX_PATH=/scratch/username/vulnrag/kb2_index
KB2_METADATA_PATH=/scratch/username/vulnrag/kb2_metadata.json
KB3_INDEX_PATH=/scratch/username/vulnrag/kb3_index
KB3_METADATA_PATH=/scratch/username/vulnrag/kb3_metadata.json

# Configuration Hugging Face
HF_HOME=/scratch/username/vulnrag/huggingface
TRANSFORMERS_CACHE=/scratch/username/vulnrag/huggingface/transformers
HF_DATASETS_CACHE=/scratch/username/vulnrag/huggingface/datasets

# Configuration pour le cluster
SLURM_JOB_ID=\$SLURM_JOB_ID
SLURM_TMPDIR=\$SLURM_TMPDIR

# Modèles Hugging Face
QWEN2_5_MODEL=Qwen/Qwen2.5-7B-Instruct
KIRITO_MODEL=Qwen/Qwen2.5-14B-Instruct

# Configuration Joern
JOERN_HOME=/scratch/username/vulnrag/joern
JAVA_HOME=\$JAVA_HOME
EOF
```

### 2. Structure des répertoires
```bash
# Créer la structure sur Cedar
mkdir -p /scratch/username/vulnrag/{data,models,logs,results}
mkdir -p /scratch/username/vulnrag/kb1_index
mkdir -p /scratch/username/vulnrag/kb2_index
mkdir -p /scratch/username/vulnrag/kb3_index
mkdir -p /scratch/username/vulnrag/huggingface/{transformers,datasets}
mkdir -p /scratch/username/vulnrag/joern
```

## Installation et déploiement

### 1. Cloner le projet
```bash
cd /scratch/username/
git clone <repository-url> vulnrag
cd vulnrag
git checkout cedar-migration
```

### 2. Environnement virtuel
```bash
# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Installer les dépendances Hugging Face
pip install transformers torch accelerate bitsandbytes
```

### 3. Configuration et test des outils
```bash
# Vérifier tous les outils
./check_tools.sh

# Test spécifique de Joern
echo "int main() { return 0; }" > test.c
if [ -f "$HOME/.local/share/coursier/bin/joern-parse" ]; then
    export PATH="$HOME/.local/share/coursier/bin:$PATH"
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
```

### 4. Test de l'interface Hugging Face
```bash
# Test des modèles Hugging Face
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
```

### 5. Test GPU (si disponible)
```bash
# Vérifier l'accès GPU
python -c "
import torch
print(f'CUDA disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Nombre de GPUs: {torch.cuda.device_count()}')
    print(f'GPU actuel: {torch.cuda.get_device_name()}')
    print(f'VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('Aucun GPU disponible')
"
```

### 6. Génération des index
```bash
# Générer les bases de connaissances
python rag/scripts/migration/migrate_kb1_to_whoosh.py
python rag/scripts/migration/migrate_kb2_to_faiss.py
python rag/scripts/migration/migrate_kb3_code_faiss.py
```

## Scripts de soumission SLURM

### 1. Script d'évaluation (CPU)
```bash
# evaluation_job.sh
#!/bin/bash
#SBATCH --account=def-username
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/evaluation_%j.out
#SBATCH --error=logs/evaluation_%j.err

# Charger les modules
module load python/3.9
module load gcc/9.3.0
module load llvm/12.0.0
module load java/11

# Activer l'environnement
source venv/bin/activate

# Configuration pour le cluster
export TMPDIR=$SLURM_TMPDIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAISS_NUM_THREADS=4

# Configuration Hugging Face
export HF_HOME=/scratch/$USER/vulnrag/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/vulnrag/huggingface/transformers
export HF_DATASETS_CACHE=/scratch/$USER/vulnrag/huggingface/datasets

# Configuration Joern
export PATH="$HOME/.local/share/coursier/bin:$PATH"

# Créer les répertoires de logs si nécessaire
mkdir -p logs

# Lancer l'évaluation
python evaluation/detection/evaluation_runner.py \
  --detectors vulnrag-qwen2.5 vulnrag-kirito qwen2.5 kirito static \
  --max-samples 100

echo "Evaluation completed successfully"
```

### 2. Script d'évaluation avec GPU
```bash
# evaluation_gpu_job.sh
#!/bin/bash
#SBATCH --account=def-username
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/evaluation_gpu_%j.out
#SBATCH --error=logs/evaluation_gpu_%j.err

module load python/3.9
module load cuda/11.4
module load java/11
source venv/bin/activate

# Configuration pour le cluster
export TMPDIR=$SLURM_TMPDIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAISS_NUM_THREADS=4

# Configuration Hugging Face
export HF_HOME=/scratch/$USER/vulnrag/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/vulnrag/huggingface/transformers
export HF_DATASETS_CACHE=/scratch/$USER/vulnrag/huggingface/datasets

# Configuration GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Configuration Joern
export PATH="$HOME/.local/share/coursier/bin:$PATH"

# Créer les répertoires de logs si nécessaire
mkdir -p logs

# Vérifier GPU
echo "GPU disponible: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Lancer l'évaluation avec GPU
python evaluation/detection/evaluation_runner.py \
  --detectors vulnrag-qwen2.5 vulnrag-kirito \
  --max-samples 50 \
  --use-gpu

echo "GPU evaluation completed successfully"
```

### 3. Script de test rapide
```bash
# quick_test.sh
#!/bin/bash
#SBATCH --account=def-username
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err

module load python/3.9
module load java/11
source venv/bin/activate

# Configuration pour le cluster
export TMPDIR=$SLURM_TMPDIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FAISS_NUM_THREADS=4

# Configuration Hugging Face
export HF_HOME=/scratch/$USER/vulnrag/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/vulnrag/huggingface/transformers
export HF_DATASETS_CACHE=/scratch/$USER/vulnrag/huggingface/datasets

# Configuration Joern
export PATH="$HOME/.local/share/coursier/bin:$PATH"

# Créer les répertoires de logs si nécessaire
mkdir -p logs

python evaluation/detection/quick_test.py

echo "Quick test completed successfully"
```

## Utilisation

### 1. Soumettre un job
```bash
# Évaluation complète (CPU)
sbatch evaluation_job.sh

# Évaluation avec GPU
sbatch evaluation_gpu_job.sh

# Test rapide
sbatch quick_test.sh
```

### 2. Surveiller les jobs
```bash
# Voir les jobs en cours
squeue -u username

# Voir les logs
tail -f logs/evaluation_*.out
```

### 3. Interface web (optionnel)
```bash
# Pour l'interface Streamlit (nécessite un tunnel SSH)
salloc --account=def-username --time=2:00:00 --mem=16G --cpus-per-task=4

# Démarrer l'interface
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Depuis votre machine locale
ssh -L 8501:cedar-node:8501 username@cedar.computecanada.ca
```

## Optimisations pour Cedar

### 1. Utilisation du stockage temporaire
```bash
# Utiliser SLURM_TMPDIR pour les fichiers temporaires
export TMPDIR=$SLURM_TMPDIR
```

### 2. Parallélisation
```bash
# Utiliser plusieurs cœurs pour l'évaluation
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

### 3. Gestion mémoire
```bash
# Limiter l'utilisation mémoire pour FAISS
export FAISS_NUM_THREADS=4

# Configuration pour Hugging Face
export HF_HUB_OFFLINE=1  # Utiliser le cache local
```

### 4. Optimisations GPU
```bash
# Utiliser la précision mixte
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Limiter l'utilisation GPU
export CUDA_VISIBLE_DEVICES=0
```

## Dépannage

### Problèmes courants
1. **Joern non trouvé** : Vérifier l'installation via coursier et ajouter au PATH
2. **Semgrep non trouvé** : Installer via `pip install semgrep` dans l'environnement virtuel
3. **Modules non trouvés** : Vérifier la disponibilité sur Cedar avec `module avail`
4. **Mémoire insuffisante** : Augmenter --mem dans le script SLURM
5. **Timeout** : Augmenter --time dans le script SLURM
6. **GPU non disponible** : Vérifier avec `sinfo -p gpu` et `squeue -p gpu`

### Logs utiles
```bash
# Logs SLURM
tail -f logs/*.err

# Logs Python
tail -f evaluation_log.txt

# Cache Hugging Face
ls -la /scratch/username/vulnrag/huggingface/

# Test Joern
~/.local/share/coursier/bin/joern-parse --help

# Test GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Avantages de Hugging Face sur Cedar

1. **Pas de serveur local** : Pas besoin de démarrer/arrêter Ollama
2. **Cache intelligent** : Les modèles sont mis en cache automatiquement
3. **Gestion mémoire** : Meilleure gestion avec PyTorch
4. **Support GPU** : Utilisation native des GPUs avec CUDA
5. **Flexibilité** : Facile de changer de modèle ou de version
6. **Stabilité** : Plus stable dans un environnement de cluster 

## ✅ **Parfait ! Maintenant sur Cedar :**

### **1. Mettre à jour le code :**
```bash
git pull
```

### **2. Lancer le script de réparation :**
```bash
<code_block_to_apply_changes_from>
```

## 🔧 **Ce que fait le script de réparation :**

1. **Nettoie l'environnement virtuel** (supprime et recrée)
2. **Installe NumPy compatible** Cedar (>=1.24.4)
3. **Installe PyTorch avec CUDA** (version 2.1.0)
4. **Installe toutes les dépendances** avec les bonnes versions
5. **Installe Joern** via coursier
6. **Vérifie l'installation** de tous les packages

## 🎯 **Après la réparation :**

Tu pourras lancer :
- **Test GPU :** `sbatch scripts/cedar/gpu_test.sh`
- **Évaluation :** `sbatch scripts/cedar/evaluation_gpu_job.sh`

**Lance le script de réparation sur Cedar !** 🚀 