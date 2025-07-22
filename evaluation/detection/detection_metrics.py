"""
Detection evaluation metrics calculator
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DetectionMetricsCalculator:
    """Calculate detection evaluation metrics"""
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, y_true: List[int], y_pred: List[int], 
                         y_scores: List[float] = None) -> Dict:
        """Calculate all detection metrics"""
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic classification metrics
        self.metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "total_samples": len(y_true),
            "vulnerable_detected": np.sum(y_pred),
            "vulnerable_actual": np.sum(y_true)
        }
        
        # Confusion matrix with proper handling for single samples
        try:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            self.metrics.update({
                "true_positives": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
                "false_positives": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
                "true_negatives": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
                "false_negatives": int(cm[1, 0]) if cm.shape == (2, 2) else 0
            })
        except Exception as e:
            logger.warning(f"Error calculating confusion matrix: {e}")
            # Fallback for edge cases
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            self.metrics.update({
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            })
        
        # Probability-based metrics (if scores provided)
        if y_scores is not None:
            y_scores = np.array(y_scores)
            try:
                self.metrics["auc_roc"] = roc_auc_score(y_true, y_scores)
                self.metrics["auc_pr"] = average_precision_score(y_true, y_scores)
            except ValueError as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                self.metrics["auc_roc"] = None
                self.metrics["auc_pr"] = None
        
        # Additional derived metrics
        self.metrics.update({
            "false_positive_rate": self.metrics["false_positives"] / (self.metrics["false_positives"] + self.metrics["true_negatives"]) if (self.metrics["false_positives"] + self.metrics["true_negatives"]) > 0 else 0,
            "false_negative_rate": self.metrics["false_negatives"] / (self.metrics["false_negatives"] + self.metrics["true_positives"]) if (self.metrics["false_negatives"] + self.metrics["true_positives"]) > 0 else 0
        })
        
        logger.info(f"Metrics calculated: {self.metrics}")
        return self.metrics
    
    def get_summary(self) -> str:
        """Get metrics summary as string"""
        if not self.metrics:
            return "No metrics calculated yet"
            
        summary = f"""
Detection Metrics Summary:
========================
Accuracy:  {self.metrics.get('accuracy', 0):.4f}
Precision: {self.metrics.get('precision', 0):.4f}
Recall:    {self.metrics.get('recall', 0):.4f}
F1-Score:  {self.metrics.get('f1_score', 0):.4f}
AUC-ROC:   {self.metrics.get('auc_roc', 'N/A')}
AUC-PR:    {self.metrics.get('auc_pr', 'N/A')}

Confusion Matrix:
TP: {self.metrics.get('true_positives', 0)} | FP: {self.metrics.get('false_positives', 0)}
FN: {self.metrics.get('false_negatives', 0)} | TN: {self.metrics.get('true_negatives', 0)}

Samples: {self.metrics.get('total_samples', 0)} total
         {self.metrics.get('vulnerable_actual', 0)} vulnerable
         {self.metrics.get('vulnerable_detected', 0)} detected as vulnerable
        """
        return summary
    
    def compare_baselines(self, baseline_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple baselines"""
        comparison_data = []
        
        for baseline_name, results in baseline_results.items():
            row = {
                "Baseline": baseline_name,
                "Accuracy": results.get("accuracy", 0),
                "Precision": results.get("precision", 0),
                "Recall": results.get("recall", 0),
                "F1-Score": results.get("f1_score", 0),
                "AUC-ROC": results.get("auc_roc", None),
                "AUC-PR": results.get("auc_pr", None),
                "TP": results.get("true_positives", 0),
                "FP": results.get("false_positives", 0),
                "FN": results.get("false_negatives", 0),
                "TN": results.get("true_negatives", 0)
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data) 