# VulnRAG sur ComputeCanada Cedar

## 🚀 Installation simple

### 1. Connexion et allocation
```bash
# Se connecter à Cedar
ssh vernet@cedar.computecanada.ca

# Aller dans scratch
cd /scratch/vernet

# Allouer des ressources GPU
salloc --account=def-foutsekh --time=2:00:00 --mem=32G --cpus-per-task=8 --partition=gpubase_bygpu_b6 --gres=gpu:v100l:1
```

### 2. Cloner le projet
```bash
git clone https://github.com/realpikiss/VULN_RAG.git vulnrag
cd vulnrag
git checkout cedar-migration
```

### 3. Configuration de base
```bash
# Charger les modules
module load python/3.11.5
module load cuda/12.2
module load java/11.0.22

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Installation de Joern
```bash
# Installer coursier
curl -fL https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz | gzip -d > cs
chmod +x cs
./cs setup
export PATH="$HOME/.local/share/coursier/bin:$PATH"

# Installer Joern
coursier launch --fork io.shiftleft:joern:2.0.0 --main io.shiftleft.joern.JoernParse -- joern-parse --help
```

## 🎯 Utilisation

### Test rapide
```bash
python evaluation/detection/quick_test.py
```

### Évaluation complète
```bash
python evaluation/detection/evaluation_runner.py --detectors vulnrag-qwen2.5 --max-samples 10
```

### Interface web
```bash
streamlit run app.py
```

## 📋 Configuration

- **Compte :** def-foutsekh
- **Partition GPU :** gpubase_bygpu_b6
- **GPU :** v100l
- **Python :** 3.11.5
- **CUDA :** 12.2
- **Java :** 11.0.22

## 🔧 Dépannage

### Problèmes courants
1. **NumPy 2.x incompatible** → Installer `numpy<2.0.0`
2. **Joern non trouvé** → Vérifier le PATH avec `export PATH="$HOME/.local/share/coursier/bin:$PATH"`
3. **CUDA non disponible** → Vérifier qu'on est dans un job GPU avec `nvidia-smi` 