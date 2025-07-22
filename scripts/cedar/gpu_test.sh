#!/bin/bash
#SBATCH --account=def-vernet
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpubase_bygpu_b6
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gpu_test_%j.out
#SBATCH --error=logs/gpu_test_%j.err

module load python/3.9
module load cuda/11.4
source venv/bin/activate

# Configuration GPU
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Configuration Hugging Face
export HF_HOME=/scratch/$USER/vulnrag/huggingface
export TRANSFORMERS_CACHE=/scratch/$USER/vulnrag/huggingface/transformers

# Configuration Joern
export PATH="$HOME/.local/share/coursier/bin:$PATH"

# Créer les répertoires de logs si nécessaire
mkdir -p logs

# Test GPU
echo "=== Test GPU ==="
echo "GPU disponible: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Test PyTorch CUDA
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

# Test modèle Hugging Face
echo "=== Test modèle Hugging Face ==="
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

# Test de génération
inputs = tokenizer('Hello, how are you?', return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'✓ Génération testée: {response}')
"

echo "GPU test completed successfully" 