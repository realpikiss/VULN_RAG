#!/bin/bash
#SBATCH --account=def-username
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err

# Charger les modules
module load python/3.9
module load gcc/9.3.0
module load llvm/12.0.0

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

# Créer les répertoires de logs si nécessaire
mkdir -p logs

# Lancer le test rapide
python evaluation/detection/quick_test.py

echo "Quick test completed successfully" 