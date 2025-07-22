#!/bin/bash
# Script de configuration automatique pour ComputeCanada Cedar

echo "=== Configuration VulnRAG pour ComputeCanada Cedar ==="
echo "Utilisateur: $USER"
echo "Répertoire projet: $(pwd)"
echo "Répertoire cache: /scratch/$USER/vulnrag/huggingface"

# Créer la structure des répertoires
echo "Création de la structure des répertoires..."
mkdir -p /scratch/$USER/vulnrag/huggingface/transformers
mkdir -p /scratch/$USER/vulnrag/huggingface/datasets
mkdir -p logs

# Charger les modules disponibles
echo "Chargement des modules..."
module load python/3.11.5
module load cuda/12.2
module load java/11.0.22

# Vérifier les modules chargés
echo "Modules chargés:"
module list

# Créer l'environnement virtuel
echo "Création de l'environnement virtuel..."
python -m venv venv
source venv/bin/activate

# Mettre à jour pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances
echo "Installation des dépendances..."
pip install -r requirements.txt

# Installation de Joern via coursier (non-admin)
echo "Installation de Joern..."
if ! command -v coursier &> /dev/null; then
    echo "Installation de coursier..."
    curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-apple-darwin.gz | gzip -d > cs
    chmod +x cs
    ./cs setup
    export PATH="$HOME/.local/share/coursier/bin:$PATH"
fi

# Installer Joern
export PATH="$HOME/.local/share/coursier/bin:$PATH"
coursier launch --fork io.shiftleft:joern:2.0.0 --main io.shiftleft.joern.JoernParse -- joern-parse --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Joern installé avec succès"
else
    echo "⚠️ Joern installation échouée, mais on continue..."
fi

# Installation des outils de sécurité
echo "Installation des outils de sécurité..."
pip install semgrep
pip install flawfinder

# Vérification des outils
echo "Vérification des outils..."

# Vérifier Python
if command -v python &> /dev/null; then
    echo "✓ Python: $(python --version)"
else
    echo "❌ Python non trouvé"
fi

# Vérifier les outils de compilation
if command -v gcc &> /dev/null; then
    echo "✓ GCC: $(gcc --version | head -n1)"
else
    echo "❌ GCC non trouvé"
fi

if command -v clang &> /dev/null; then
    echo "✓ Clang: $(clang --version | head -n1)"
else
    echo "❌ Clang non trouvé"
fi

# Vérifier les outils de sécurité
if command -v cppcheck &> /dev/null; then
    echo "✓ Cppcheck: $(cppcheck --version | head -n1)"
else
    echo "⚠️ Cppcheck non trouvé (peut être installé via module)"
fi

if command -v clang-tidy &> /dev/null; then
    echo "✓ Clang-tidy: $(clang-tidy --version | head -n1)"
else
    echo "⚠️ Clang-tidy non trouvé (peut être installé via module)"
fi

if command -v flawfinder &> /dev/null; then
    echo "✓ Flawfinder: $(flawfinder --version)"
else
    echo "❌ Flawfinder non trouvé"
fi

if command -v semgrep &> /dev/null; then
    echo "✓ Semgrep: $(semgrep --version)"
else
    echo "❌ Semgrep non trouvé"
fi

# Vérifier Joern
export PATH="$HOME/.local/share/coursier/bin:$PATH"
if command -v joern-parse &> /dev/null; then
    echo "✓ Joern: disponible"
else
    echo "⚠️ Joern non trouvé (installation manuelle requise)"
fi

# Vérifier Java
if command -v java &> /dev/null; then
    echo "✓ Java: $(java -version 2>&1 | head -n1)"
else
    echo "❌ Java non trouvé"
fi

# Vérifier CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA: $(nvidia-smi --version | head -n1)"
else
    echo "⚠️ CUDA non trouvé (peut être disponible dans un job GPU)"
fi

# Vérifier les packages Python
echo "Vérification des packages Python..."
python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA disponible: {torch.cuda.is_available()}')

import transformers
print(f'✓ Transformers: {transformers.__version__}')

import faiss
print(f'✓ FAISS: {faiss.__version__}')

import whoosh
print(f'✓ Whoosh: {whoosh.__version__}')
"

# Configuration des variables d'environnement
echo "Configuration des variables d'environnement..."
cat > .env << EOF
# Configuration Cedar
HF_HOME=/scratch/$USER/vulnrag/huggingface
TRANSFORMERS_CACHE=/scratch/$USER/vulnrag/huggingface/transformers
HF_DATASETS_CACHE=/scratch/$USER/vulnrag/huggingface/datasets

# Configuration Joern
PATH=\$PATH:\$HOME/.local/share/coursier/bin

# Configuration GPU
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Configuration cluster
TMPDIR=\$SLURM_TMPDIR
OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
FAISS_NUM_THREADS=4
EOF

echo "✓ Configuration terminée !"
echo ""
echo "=== RÉSUMÉ ==="
echo "Compte: def-foutsekh"
echo "Partition GPU: gpubase_bygpu_b6"
echo "Type GPU: v100l"
echo "Modules: python/3.11.5, cuda/12.2, java/11.0.22"
echo "Répertoire: /scratch/$USER/vulnrag"
echo ""
echo "Pour lancer un test GPU:"
echo "sbatch scripts/cedar/gpu_test.sh"
echo ""
echo "Pour lancer une évaluation:"
echo "sbatch scripts/cedar/evaluation_gpu_job.sh" 