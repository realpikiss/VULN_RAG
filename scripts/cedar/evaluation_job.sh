#!/bin/bash
#SBATCH --account=def-fouts+
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/evaluation_%j.out
#SBATCH --error=logs/evaluation_%j.err

# Charger les modules
module load python/3.11.5
module load gcc/9.3.0
module load llvm/12.0.0
module load java/11.0.22

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