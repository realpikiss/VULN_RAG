#!/bin/bash

echo "=== Test simple VulnRAG sur Cedar ==="

# Charger les modules de base
module load python/3.11.5
module load cuda/12.2

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances de base
pip install --upgrade pip
pip install "numpy<2.0.0"
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.35.0

# Test simple
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

import transformers
print(f'Transformers: {transformers.__version__}')

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
print('✓ Tokenizer chargé avec succès')
"

echo "✓ Test terminé !" 