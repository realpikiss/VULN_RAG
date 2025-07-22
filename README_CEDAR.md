# 🖥️ Migration VulnRAG vers ComputeCanada Cedar

## Vue d'ensemble

Cette branche `cedar-migration` contient la configuration complète pour déployer VulnRAG sur le cluster ComputeCanada Cedar en utilisant **Hugging Face** au lieu d'Ollama pour les modèles LLM.

## 🚀 Démarrage rapide

### 1. Connexion à Cedar
```bash
ssh username@cedar.computecanada.ca
```

### 2. Configuration automatique
```bash
# Allouer des ressources
salloc --account=def-username --time=2:00:00 --mem=32G --cpus-per-task=8

# Cloner le projet
cd /scratch/username/
git clone <repository-url> vulnrag
cd vulnrag
git checkout cedar-migration

# Configuration automatique
bash scripts/cedar/setup_cedar.sh
```

### 3. Test de la configuration
```bash
python test_setup.py
```

### 4. Génération des index
```bash
python rag/scripts/migration/migrate_kb1_to_whoosh.py
python rag/scripts/migration/migrate_kb2_to_faiss.py
python rag/scripts/migration/migrate_kb3_code_faiss.py
```

### 5. Lancement des tests
```bash
# Test rapide
sbatch scripts/cedar/quick_test.sh

# Évaluation complète
sbatch scripts/cedar/evaluation_job.sh

# Avec GPU (si disponible)
sbatch scripts/cedar/evaluation_gpu_job.sh
```

## 🔧 Configuration détaillée

### Prérequis sur Cedar

#### Modules nécessaires
```bash
module load python/3.9
module load gcc/9.3.0
module load llvm/12.0.0  # Pour clang-tidy
module load cuda/11.4    # Si utilisation GPU
```

#### Outils statiques
```bash
# Cppcheck
conda install -c conda-forge cppcheck

# Flawfinder et Semgrep
pip install flawfinder semgrep
```

### Structure des répertoires

```
/scratch/username/vulnrag/
├── venv/                    # Environnement virtuel Python
├── huggingface/            # Cache Hugging Face
│   ├── transformers/       # Modèles et tokenizers
│   └── datasets/          # Datasets
├── kb1_index/             # Index Whoosh
├── kb2_index/             # Index FAISS CPG
├── kb3_index/             # Index FAISS code
├── logs/                  # Logs SLURM
├── results/               # Résultats d'évaluation
└── data/                  # Données du projet
```

### Variables d'environnement

Le fichier `.env` est créé automatiquement avec :

```bash
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

# Modèles Hugging Face
QWEN2_5_MODEL=Qwen/Qwen2.5-7B-Instruct
KIRITO_MODEL=Qwen/Qwen2.5-14B-Instruct
```

## 📋 Scripts SLURM

### Scripts disponibles

| Script | Description | Ressources | Durée |
|--------|-------------|------------|-------|
| `quick_test.sh` | Test rapide | 32G RAM, 8 CPU | 1h |
| `evaluation_job.sh` | Évaluation complète | 64G RAM, 16 CPU | 4h |
| `evaluation_gpu_job.sh` | Évaluation avec GPU | 32G RAM, 8 CPU, 1 GPU | 2h |

### Utilisation

```bash
# Soumettre un job
sbatch scripts/cedar/evaluation_job.sh

# Surveiller les jobs
squeue -u username

# Voir les logs
tail -f logs/evaluation_*.out
```

### Personnalisation

**IMPORTANT** : Modifiez les scripts avec votre nom d'utilisateur :

```bash
# Remplacer 'username' par votre nom d'utilisateur
sed -i 's/username/VOTRE_NOM_UTILISATEUR/g' scripts/cedar/*.sh
```

## 🤖 Modèles Hugging Face

### Modèles supportés

| Modèle | Taille | Usage | Performance |
|--------|--------|-------|-------------|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Détection par défaut | Rapide, bonne qualité |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Détection améliorée | Plus lent, meilleure qualité |

### Configuration

Les modèles sont automatiquement téléchargés et mis en cache dans `/scratch/username/vulnrag/huggingface/`.

### Optimisations

- **Quantification 4-bit** : Réduction de l'utilisation mémoire
- **Cache intelligent** : Téléchargement automatique et mise en cache
- **Support GPU** : Utilisation native de CUDA

## 🔍 Interface Hugging Face

### Remplacement d'Ollama

L'interface `HuggingFaceInterface` remplace complètement Ollama :

```python
from rag.core.generation.huggingface_interface import HuggingFaceInterface

# Interface compatible avec l'ancienne API
interface = HuggingFaceInterface(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    use_4bit=True,  # Quantification pour économiser la mémoire
    temperature=0.1
)

# Génération de texte
response = interface.generate(
    prompt="Analyze this C++ code for vulnerabilities...",
    max_new_tokens=512
)
```

### Mapping des modèles

```python
# Anciens noms Ollama → Nouveaux noms Hugging Face
"qwen2.5-coder:latest" → "Qwen/Qwen2.5-7B-Instruct"
"kirito1/qwen3-coder:latest" → "Qwen/Qwen2.5-14B-Instruct"
```

## 📊 Évaluation sur Cedar

### Détecteurs disponibles

| Détecteur | Description | Modèle |
|-----------|-------------|--------|
| `vulnrag-qwen2.5` | VulnRAG avec Qwen2.5 | Qwen/Qwen2.5-7B-Instruct |
| `vulnrag-kirito` | VulnRAG avec Kirito | Qwen/Qwen2.5-14B-Instruct |
| `qwen2.5` | Qwen2.5 solo | Qwen/Qwen2.5-7B-Instruct |
| `kirito` | Kirito solo | Qwen/Qwen2.5-14B-Instruct |
| `static` | Outils statiques uniquement | Aucun |

### Lancement d'évaluation

```bash
# Évaluation complète
python evaluation/detection/evaluation_runner.py \
  --detectors vulnrag-qwen2.5 vulnrag-kirito qwen2.5 kirito static \
  --max-samples 100

# Test rapide
python evaluation/detection/quick_test.py
```

## 🛠️ Dépannage

### Problèmes courants

#### 1. Modules non trouvés
```bash
# Vérifier les modules disponibles
module avail python
module avail gcc
module avail llvm
```

#### 2. Mémoire insuffisante
```bash
# Augmenter la mémoire dans les scripts SLURM
#SBATCH --mem=128G  # Au lieu de 64G
```

#### 3. Modèles Hugging Face non téléchargés
```bash
# Vérifier l'accès internet
curl -I https://huggingface.co

# Télécharger manuellement
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

#### 4. GPU non disponible
```bash
# Vérifier la disponibilité des GPUs
sinfo -p gpu

# Utiliser CPU uniquement
export CUDA_VISIBLE_DEVICES=""
```

### Logs utiles

```bash
# Logs SLURM
tail -f logs/*.err

# Logs Python
tail -f evaluation_log.txt

# Cache Hugging Face
ls -la /scratch/username/vulnrag/huggingface/
```

## 🔄 Migration depuis Ollama

### Changements principaux

1. **Remplacement d'Ollama** : Interface Hugging Face native
2. **Cache intelligent** : Téléchargement automatique des modèles
3. **Support GPU** : Utilisation native de CUDA
4. **Quantification** : Réduction de l'utilisation mémoire
5. **Stabilité** : Plus de serveur local à gérer

### Compatibilité

L'interface Hugging Face maintient la compatibilité avec l'ancienne API Ollama :

```python
# Ancien code (fonctionne toujours)
from rag.core.generation.ollama_qwen import generate
result = generate(prompt, model="qwen2.5-coder:latest")

# Nouveau code (recommandé)
from rag.core.generation.huggingface_interface import HuggingFaceInterface
interface = HuggingFaceInterface("Qwen/Qwen2.5-7B-Instruct")
result = interface.generate(prompt)
```

## 📈 Avantages de Hugging Face sur Cedar

1. **Pas de serveur local** : Pas besoin de démarrer/arrêter Ollama
2. **Cache intelligent** : Les modèles sont mis en cache automatiquement
3. **Gestion mémoire** : Meilleure gestion avec PyTorch
4. **Support GPU** : Utilisation native des GPUs avec CUDA
5. **Flexibilité** : Facile de changer de modèle ou de version
6. **Stabilité** : Plus stable dans un environnement de cluster
7. **Performance** : Optimisations spécifiques pour les clusters

## 🎯 Prochaines étapes

1. **Test de la configuration** : `python test_setup.py`
2. **Génération des index** : Scripts de migration
3. **Test rapide** : `sbatch scripts/cedar/quick_test.sh`
4. **Évaluation complète** : `sbatch scripts/cedar/evaluation_job.sh`
5. **Optimisation** : Ajuster les paramètres selon les résultats

## 📞 Support

- **Documentation** : Voir `cedar_setup.md` pour plus de détails
- **Scripts** : Tous les scripts sont dans `scripts/cedar/`
- **Logs** : Vérifiez les logs dans `logs/`
- **Configuration** : Le fichier `.env` contient toutes les variables 