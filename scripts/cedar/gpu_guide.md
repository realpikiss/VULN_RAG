# 🖥️ Guide GPU pour ComputeCanada Cedar

## Vérification de la disponibilité des GPUs

### 1. Voir les partitions GPU
```bash
# Voir toutes les partitions disponibles
sinfo

# Voir spécifiquement les partitions GPU
sinfo -p gpu
```

### 2. Voir les GPUs disponibles
```bash
# Voir les jobs GPU en cours
squeue -p gpu

# Voir les nœuds GPU
sinfo -N -l | grep gpu
```

### 3. Types de GPUs sur Cedar
```bash
# V100 (16GB VRAM) - Plus commun
salloc --gres=gpu:v100:1

# A100 (40GB VRAM) - Plus puissant, plus rare
salloc --gres=gpu:a100:1

# RTX 6000 (24GB VRAM)
salloc --gres=gpu:rtx6000:1

# GPU générique (n'importe quel type)
salloc --gres=gpu:1
```

## Demander des ressources GPU

### 1. Session interactive avec GPU
```bash
# Demander un nœud avec GPU V100
salloc --account=def-username --time=2:00:00 --mem=32G --cpus-per-task=8 --gres=gpu:v100:1

# Demander un nœud avec GPU A100 (plus puissant)
salloc --account=def-username --time=2:00:00 --mem=64G --cpus-per-task=16 --gres=gpu:a100:1

# Demander plusieurs GPUs (si nécessaire)
salloc --account=def-username --time=4:00:00 --mem=64G --cpus-per-task=16 --gres=gpu:v100:2
```

### 2. Job batch avec GPU
```bash
# Soumettre un job avec GPU
sbatch --gres=gpu:v100:1 scripts/cedar/evaluation_gpu_job.sh

# Ou modifier le script pour inclure GPU
#SBATCH --gres=gpu:v100:1
```

## Vérifier l'accès GPU

### 1. Dans une session interactive
```bash
# Vérifier les GPUs disponibles
nvidia-smi

# Vérifier avec PyTorch
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

### 2. Variables d'environnement GPU
```bash
# Voir les variables GPU
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_CUDA_DEVICES: $SLURM_CUDA_DEVICES"

# Configurer pour utiliser le premier GPU
export CUDA_VISIBLE_DEVICES=0
```

## Optimisations GPU pour VulnRAG

### 1. Configuration PyTorch
```bash
# Optimisations mémoire GPU
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Utiliser la précision mixte
export CUDA_LAUNCH_BLOCKING=1
```

### 2. Configuration Hugging Face
```python
# Dans votre code Python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configuration 4-bit pour économiser la VRAM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Charger le modèle avec quantification
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
```

### 3. Gestion de la mémoire GPU
```python
# Vérifier l'utilisation mémoire
import torch
print(f"VRAM utilisée: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM réservée: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Nettoyer la mémoire
torch.cuda.empty_cache()
```

## Scripts SLURM avec GPU

### 1. Script d'évaluation GPU
```bash
#!/bin/bash
#SBATCH --account=def-username
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/evaluation_gpu_%j.out
#SBATCH --error=logs/evaluation_gpu_%j.err

module load python/3.9
module load cuda/11.4
module load java/11
source venv/bin/activate

# Configuration GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Configuration Hugging Face
export HF_HOME=/scratch/$USER/vulnrag/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/vulnrag/huggingface/transformers

# Configuration Joern
export PATH="$HOME/.local/share/coursier/bin:$PATH"

# Vérifier GPU
echo "GPU disponible: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Lancer l'évaluation
python evaluation/detection/evaluation_runner.py \
  --detectors vulnrag-qwen2.5 vulnrag-kirito \
  --max-samples 50 \
  --use-gpu
```

### 2. Script de test GPU rapide
```bash
#!/bin/bash
#SBATCH --account=def-username
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err

module load python/3.9
module load cuda/11.4
source venv/bin/activate

# Test GPU
echo "=== Test GPU ==="
nvidia-smi
python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Test modèle Hugging Face
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Test chargement modèle...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
print('✓ Tokenizer chargé')

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)
print('✓ Modèle chargé sur GPU')
"
```

## Dépannage GPU

### 1. Problèmes courants

#### GPU non disponible
```bash
# Vérifier si vous êtes dans un job GPU
echo $SLURM_JOB_PARTITION
echo $CUDA_VISIBLE_DEVICES

# Si pas de GPU, demander des ressources
salloc --gres=gpu:v100:1
```

#### Mémoire GPU insuffisante
```bash
# Vérifier l'utilisation mémoire
nvidia-smi

# Utiliser la quantification 4-bit
# Voir configuration Hugging Face ci-dessus
```

#### Module CUDA non trouvé
```bash
# Charger le module CUDA
module load cuda/11.4

# Vérifier l'installation
nvcc --version
```

### 2. Logs utiles
```bash
# Logs SLURM
tail -f logs/evaluation_gpu_*.err

# Informations GPU
nvidia-smi -l 1  # Mise à jour toutes les secondes

# Test PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

## Conseils pour l'utilisation des GPUs

### 1. Choisir le bon GPU
- **V100 (16GB)** : Bon pour la plupart des modèles 7B
- **A100 (40GB)** : Idéal pour les modèles 14B+ et batch processing
- **RTX 6000 (24GB)** : Bon compromis

### 2. Optimiser l'utilisation
- Utiliser la quantification 4-bit pour économiser la VRAM
- Nettoyer la mémoire entre les jobs
- Utiliser `device_map="auto"` pour la gestion automatique

### 3. Monitoring
- Surveiller l'utilisation avec `nvidia-smi`
- Vérifier les logs SLURM pour les erreurs
- Utiliser `torch.cuda.memory_summary()` pour le debugging

## Exemple complet d'utilisation

```bash
# 1. Demander des ressources GPU
salloc --account=def-username --time=2:00:00 --mem=32G --cpus-per-task=8 --gres=gpu:v100:1

# 2. Vérifier l'accès GPU
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Lancer l'évaluation
sbatch scripts/cedar/evaluation_gpu_job.sh

# 4. Surveiller
squeue -u username
tail -f logs/evaluation_gpu_*.out
``` 