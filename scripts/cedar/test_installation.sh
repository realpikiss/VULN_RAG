#!/bin/bash

echo "=== Test rapide de l'installation VulnRAG ==="

# Charger les modules
module load python/3.11.5
module load cuda/12.2
module load java/11.0.22

# Activer l'environnement virtuel
source venv/bin/activate

echo "1. Test des imports Python..."
python -c "
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'✓ CUDA disponible: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import transformers
    print(f'✓ Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Transformers: {e}')

try:
    import numpy
    print(f'✓ NumPy: {numpy.__version__}')
except ImportError as e:
    print(f'❌ NumPy: {e}')

try:
    import faiss
    print(f'✓ FAISS: {faiss.__version__}')
except ImportError as e:
    print(f'❌ FAISS: {e}')

try:
    import whoosh
    print(f'✓ Whoosh: {whoosh.__version__}')
except ImportError as e:
    print(f'❌ Whoosh: {e}')

try:
    import semgrep
    print(f'✓ Semgrep: {semgrep.__version__}')
except ImportError as e:
    print(f'❌ Semgrep: {e}')

try:
    import flawfinder
    print(f'✓ Flawfinder: {flawfinder.__version__}')
except ImportError as e:
    print(f'❌ Flawfinder: {e}')
"

echo ""
echo "2. Test des outils externes..."
export PATH="$HOME/.local/share/coursier/bin:$PATH"

if command -v joern-parse &> /dev/null; then
    echo "✓ Joern: disponible"
else
    echo "❌ Joern: non trouvé"
fi

if command -v java &> /dev/null; then
    echo "✓ Java: $(java -version 2>&1 | head -n1)"
else
    echo "❌ Java: non trouvé"
fi

echo ""
echo "3. Test de chargement de modèle Hugging Face..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
    print('✓ Tokenizer Qwen2.5 chargé avec succès')
except Exception as e:
    print(f'❌ Erreur chargement tokenizer: {e}')
"

echo ""
echo "✓ Test terminé !" 