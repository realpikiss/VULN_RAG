#!/bin/bash

echo "=== Réparation de l'installation VulnRAG ==="

# Charger les modules
module load python/3.11.5
module load cuda/12.2
module load java/11.0.22
module load arrow/12.0.1

# Activer l'environnement virtuel
source venv/bin/activate

echo "1. Nettoyage de l'environnement virtuel..."
rm -rf venv
python -m venv venv
source venv/bin/activate

echo "2. Mise à jour de pip..."
pip install --upgrade pip

echo "3. Installation des dépendances de base..."
pip install wheel setuptools

echo "4. Installation de NumPy 1.x compatible PyTorch..."
pip install "numpy<2.0.0"

echo "5. Installation de PyTorch avec CUDA..."
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "6. Installation des autres dépendances..."
pip install -r requirements.txt

echo "7. Installation de Joern via coursier..."
# Installer coursier si pas déjà fait
if ! command -v coursier &> /dev/null; then
    echo "Installation de coursier..."
    curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs
    chmod +x cs
    ./cs setup
    export PATH="$HOME/.local/share/coursier/bin:$PATH"
fi

# Installer Joern
export PATH="$HOME/.local/share/coursier/bin:$PATH"
coursier launch --fork io.shiftleft:joern:2.0.0 --main io.shiftleft.joern.JoernParse -- joern-parse --help > /dev/null 2>&1

echo "8. Vérification de l'installation..."
python -c "
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA disponible: {torch.cuda.is_available()}')

import transformers
print(f'✓ Transformers: {transformers.__version__}')

import numpy
print(f'✓ NumPy: {numpy.__version__}')

import faiss
print(f'✓ FAISS: {faiss.__version__}')

import whoosh
print(f'✓ Whoosh: {whoosh.__version__}')
"

echo "9. Test de Joern..."
export PATH="$HOME/.local/share/coursier/bin:$PATH"
if command -v joern-parse &> /dev/null; then
    echo "✓ Joern installé avec succès"
else
    echo "⚠️ Joern non trouvé, installation manuelle requise"
fi

echo "✓ Réparation terminée !" 