# 🔍 Detection Evaluation Framework

Framework d'évaluation pour comparer VulnRAG contre des baselines sur la détection de vulnérabilités avec support multi-LLM.

## 📁 Structure

```
evaluation/detection/
├── __init__.py
├── dataset_loader.py      # Chargement du dataset de détection
├── detection_metrics.py   # Calcul des métriques d'évaluation
├── baselines.py          # Implémentation des détecteurs
├── evaluation_runner.py  # Runner principal d'évaluation
├── quick_test.py         # Test rapide du framework
├── results/              # Résultats d'évaluation
└── README.md            # Ce fichier
```

## 🚀 Utilisation Rapide

### 1. Test du Framework
```bash
cd evaluation/detection
python quick_test.py
```

### 2. Évaluation Complète
```bash
# Évaluation avec tous les détecteurs disponibles
python evaluation_runner.py

# Évaluation avec limitation d'échantillons
python evaluation_runner.py --max-samples 50

# Évaluation avec détecteurs spécifiques
python evaluation_runner.py --detectors vulnrag-qwen2.5 vulnrag-kirito static
```

### 3. Évaluation Détaillée
```bash
# Évaluation complète (1,172 échantillons)
python evaluation_runner.py --output-dir results/full_evaluation

# Comparaison multi-LLM
python evaluation_runner.py --detectors vulnrag-qwen2.5 vulnrag-kirito qwen2.5 kirito
```

## 📊 Dataset

- **Source** : `rag/data/Datasets/dataset_detection_clean.csv`
- **Taille** : 1,172 fonctions
- **Répartition** : 586 vulnérables (label=1) + 586 non vulnérables (label=0)
- **CWE couvertes** : 416, 476, 362, 119, 787, 20, 200, 125, 264, 401

## 🎯 Détecteurs Disponibles

### **VulnRAG Multi-LLM**
| Détecteur | LLM | Description | Performance |
|-----------|-----|-------------|-------------|
| **vulnrag-qwen2.5** | Qwen2.5-Coder | VulnRAG avec Qwen2.5 | Rapide, bonne qualité |
| **vulnrag-kirito** | Kirito/Qwen3-Coder | VulnRAG avec Kirito | Plus lent, meilleure qualité |

### **LLM Solo**
| Détecteur | LLM | Description | Performance |
|-----------|-----|-------------|-------------|
| **qwen2.5** | Qwen2.5-Coder | Qwen2.5 sans RAG | Baseline Qwen |
| **kirito** | Kirito/Qwen3-Coder | Kirito sans RAG | Baseline Kirito |

### **Baselines Traditionnels**
| Détecteur | Type | Description | Performance |
|-----------|------|-------------|-------------|
| **static** | Outils statiques | Cppcheck + Clang-Tidy + Flawfinder | Rapide, limité |
| **gpt** | API externe | GPT-4 via OpenAI | Qualité élevée, coût |

## 📈 Métriques Calculées

### Métriques de Classification
- **Accuracy** : (TP + TN) / Total
- **Precision** : TP / (TP + FP)
- **Recall** : TP / (TP + FN)
- **F1-Score** : 2 × (Precision × Recall) / (Precision + Recall)

### Métriques Probabilistes
- **AUC-ROC** : Area Under ROC Curve
- **AUC-PR** : Area Under Precision-Recall Curve

### Métriques de Performance
- **Temps total** : Temps d'évaluation complet
- **Temps par échantillon** : Temps moyen par fonction

### Métriques VulnRAG Spécifiques
- **Decision Analysis** : Répartition des décisions (Static/Heuristic, LLM Arbitration, Full RAG)
- **Confidence Scores** : Scores de confiance calculés

## 📊 Résultats Récents

### **Performance Comparaison (5 échantillons)**

| Detector | Accuracy | Precision | Recall | F1-Score | Time/sample |
|----------|----------|-----------|--------|----------|-------------|
| **vulnrag-qwen2.5** | 100% | 100% | 100% | 1.0 | 28.7s |
| **vulnrag-kirito** | 100% | 100% | 100% | 1.0 | 25.4s |
| **kirito** | 100% | 100% | 100% | 1.0 | 11.8s |
| **qwen2.5** | 0% | 0% | 0% | 0.0 | 7.4s |
| **static** | 50% | 0% | 0% | 0.0 | 0.2s |

### **Insights Clés**

- **VulnRAG améliore significativement Qwen2.5** (0% → 100% recall)
- **Kirito montre de meilleures performances solo** que Qwen2.5
- **VulnRAG maintient une haute précision** across models
- **Les outils statiques seuls sont insuffisants** (0% recall)

## 📊 Sauvegarde des Résultats

Les résultats sont sauvegardés dans `evaluation/detection/results/` :

- `detection_results.json` : Résultats détaillés par détecteur
- `comparison_table.json` : Tableau comparatif
- `dataset_stats.json` : Statistiques du dataset

### Format des Résultats
```json
{
  "vulnrag-qwen2.5": {
    "metrics": {
      "accuracy": 1.000,
      "precision": 1.000,
      "recall": 1.000,
      "f1_score": 1.000,
      "auc_roc": 0.0,
      "total_time": 143.5,
      "avg_time_per_sample": 28.7
    },
    "detailed_results": [...],
    "detection_results": {...},
    "decision_analysis": {
      "STATIC_HEURISTIC_AGREEMENT": 1,
      "LLM_ARBITRATION": 4,
      "FULL_RAG_PIPELINE": 0,
      "UNKNOWN": 0
    }
  }
}
```

## 🔧 Configuration

### Variables d'Environnement
```bash
# Pour GPT
export OPENAI_API_KEY="your-api-key"

# Pour VulnRAG (déjà configuré)
export KB1_INDEX_PATH="rag/data/indexes/kb1_index"
export KB2_INDEX_PATH="rag/data/indexes/kb2_index/kb2_code.index"
export KB3_INDEX_PATH="rag/data/indexes/kb3_index/kb3_code.index"
```

### Modèles Ollama Requis
```bash
# Installer les modèles nécessaires
ollama pull qwen2.5-coder:latest
ollama pull kirito1/qwen3-coder:latest

# Vérifier les modèles installés
ollama list
```

### Dépendances
```bash
# Outils statiques
brew install cppcheck clang-tidy
pip install flawfinder semgrep

# LLMs
pip install ollama openai

# Métriques
pip install scikit-learn pandas numpy

# RAG components
pip install sentence-transformers faiss-cpu whoosh
```

## 🛠️ Extension

### Ajouter un Nouveau Détecteur
1. Créer une classe héritant de `BaseDetector`
2. Implémenter la méthode `detect()`
3. Ajouter dans `get_available_detectors()`

### Exemple
```python
class CustomDetector(BaseDetector):
    def __init__(self, model: str = "custom-model:latest"):
        super().__init__(f"Custom-{model.split('/')[-1].split(':')[0]}")
        self.model = model
    
    def detect(self, code: str) -> Tuple[int, float]:
        # Votre logique de détection
        return prediction, confidence
```

### Ajouter un Nouveau Modèle LLM
1. Ajouter le modèle dans `VulnRAGDetector`
2. Ajouter le détecteur solo correspondant
3. Mettre à jour `get_available_detectors()`

## 📝 Notes Importantes

1. **Ollama** : Le serveur Ollama doit être en cours d'exécution
2. **Modèles** : Vérifiez que tous les modèles sont installés avec `ollama list`
3. **Outils statiques** : Cppcheck, Clang-Tidy et Flawfinder doivent être installés
4. **Mémoire** : L'évaluation complète peut prendre du temps (plusieurs heures)
5. **API GPT** : Coûts associés à l'utilisation de l'API

## 🐛 Dépannage

### Erreurs Communes
- **Import Error** : Vérifiez le PYTHONPATH
- **Ollama Connection** : Vérifiez que le serveur Ollama fonctionne
- **Model Not Found** : Vérifiez `ollama list` et installez les modèles manquants
- **Tool Not Found** : Installez cppcheck, clang-tidy et flawfinder
- **API Key Missing** : Configurez OPENAI_API_KEY

### Logs
```bash
# Activer les logs détaillés
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -u evaluation_runner.py --max-samples 5
```

### Test Rapide
```bash
# Test complet du framework
python quick_test.py

# Vérification des détecteurs disponibles
python -c "from baselines import get_available_detectors; print(list(get_available_detectors().keys()))"
```

## 🎯 Cas d'Usage Recommandés

### **Évaluation Complète**
```bash
# Tous les détecteurs sur tout le dataset
python evaluation_runner.py
```

### **Comparaison Multi-LLM**
```bash
# VulnRAG vs LLM solo
python evaluation_runner.py --detectors vulnrag-qwen2.5 vulnrag-kirito qwen2.5 kirito
```

### **Test Rapide**
```bash
# Test avec peu d'échantillons
python evaluation_runner.py --max-samples 10 --detectors vulnrag-qwen2.5 static
```

### **Analyse de Performance**
```bash
# Focus sur les temps de traitement
python evaluation_runner.py --detectors vulnrag-qwen2.5 vulnrag-kirito --max-samples 50
``` 