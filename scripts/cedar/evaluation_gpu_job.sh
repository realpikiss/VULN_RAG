#!/bin/bash
#SBATCH --account=def-vernet
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpubase_bygpu_b6
#SBATCH --gres=gpu:1
#SBATCH --output=logs/evaluation_gpu_%j.out
#SBATCH --error=logs/evaluation_gpu_%j.err

# Charger les modules
module load python/3.9
module load gcc/9.3.0
module load llvm/12.0.0
module load cuda/11.4
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