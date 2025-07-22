#!/bin/bash

echo "=== Vérification des modules disponibles sur Cedar ==="

echo "1. Modules Python disponibles:"
module avail python 2>&1 | head -20

echo ""
echo "2. Modules CUDA disponibles:"
module avail cuda 2>&1 | head -10

echo ""
echo "3. Modules Java disponibles:"
module avail java 2>&1 | head -10

echo ""
echo "4. Modules Arrow disponibles:"
module avail arrow 2>&1 | head -10

echo ""
echo "5. Modules GCC disponibles:"
module avail gcc 2>&1 | head -10

echo ""
echo "6. Modules LLVM disponibles:"
module avail llvm 2>&1 | head -10

echo ""
echo "7. Modules actuellement chargés:"
module list

echo ""
echo "8. Test de chargement des modules:"
echo "Test Python..."
module load python/3.11.5 2>&1 && echo "✓ Python chargé" || echo "❌ Python non disponible"

echo "Test CUDA..."
module load cuda/12.2 2>&1 && echo "✓ CUDA chargé" || echo "❌ CUDA non disponible"

echo "Test Java..."
module load java/11.0.22 2>&1 && echo "✓ Java chargé" || echo "❌ Java non disponible"

echo "Test Arrow..."
module load arrow/12.0.1 2>&1 && echo "✓ Arrow chargé" || echo "❌ Arrow non disponible" 