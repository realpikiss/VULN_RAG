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
```

### 3. Installation des outils statiques
```bash
# Cppcheck (via conda ou compilation)
conda install -c conda-forge cppcheck

# Flawfinder (via pip)
pip install flawfinder

# Semgrep
pip install semgrep
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

### 3. Configuration Hugging Face
```bash
# Télécharger les modèles (optionnel - se fait automatiquement)
# python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"

# Vérifier l'accès aux modèles
python -c "from transformers import AutoTokenizer; print('Qwen2.5-7B tokenizer loaded successfully')"
```

### 4. Génération des index
```bash
# Générer les bases de connaissances
python rag/scripts/migration/migrate_kb1_to_whoosh.py
python rag/scripts/migration/migrate_kb2_to_faiss.py
python rag/scripts/migration/migrate_kb3_code_faiss.py
```

## Scripts de soumission SLURM

### 1. Script d'évaluation
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

# Activer l'environnement
source venv/bin/activate

# Lancer l'évaluation
python evaluation/detection/evaluation_runner.py \
  --detectors vulnrag-qwen2.5 vulnrag-kirito qwen2.5 kirito static \
  --max-samples 100
```

### 2. Script de test rapide
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
source venv/bin/activate

python evaluation/detection/quick_test.py
```

### 3. Script avec GPU (optionnel)
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
source venv/bin/activate

python evaluation/detection/evaluation_runner.py \
  --detectors vulnrag-qwen2.5 vulnrag-kirito \
  --max-samples 50 \
  --use-gpu
```

## Utilisation

### 1. Soumettre un job
```bash
# Évaluation complète
sbatch evaluation_job.sh

# Test rapide
sbatch quick_test.sh

# Avec GPU
sbatch evaluation_gpu_job.sh
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
1. **Modèles non trouvés** : Vérifier l'accès internet et les permissions
2. **Modules non trouvés** : Vérifier la disponibilité sur Cedar
3. **Mémoire insuffisante** : Augmenter --mem dans le script SLURM
4. **Timeout** : Augmenter --time dans le script SLURM
5. **GPU non disponible** : Vérifier la disponibilité des GPUs sur Cedar

### Logs utiles
```bash
# Logs SLURM
tail -f logs/*.err

# Logs Python
tail -f evaluation_log.txt

# Cache Hugging Face
ls -la /scratch/username/vulnrag/huggingface/
```

## Avantages de Hugging Face sur Cedar

1. **Pas de serveur local** : Pas besoin de démarrer/arrêter Ollama
2. **Cache intelligent** : Les modèles sont mis en cache automatiquement
3. **Gestion mémoire** : Meilleure gestion de la mémoire avec torch
4. **Support GPU** : Utilisation native des GPUs avec CUDA
5. **Flexibilité** : Facile de changer de modèle ou de version
6. **Stabilité** : Plus stable dans un environnement de cluster 