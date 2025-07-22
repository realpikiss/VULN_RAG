"""
Main evaluation runner for detection
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

from dataset_loader import DetectionDatasetLoader
from detection_metrics import DetectionMetricsCalculator
from baselines import get_available_detectors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DetectionEvaluationRunner:
    """Main runner for detection evaluation"""
    
    def __init__(self, output_dir: str = "evaluation/detection/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_loader = DetectionDatasetLoader()
        self.metrics_calculator = DetectionMetricsCalculator()
        self.detectors = get_available_detectors()
        
        logger.info(f"Available detectors: {list(self.detectors.keys())}")
    
    def run_evaluation(self, max_samples: int = None, 
                      selected_detectors: List[str] = None) -> Dict:
        """Run complete detection evaluation"""
        
        logger.info("Starting detection evaluation")
        
        # Load dataset
        samples = self.dataset_loader.get_samples(max_samples)
        dataset_stats = self.dataset_loader.get_statistics()
        
        logger.info(f"Loaded {len(samples)} samples for evaluation")
        logger.info(f"Dataset stats: {dataset_stats}")
        
        # Filter detectors if specified
        if selected_detectors:
            available_detectors = {k: v for k, v in self.detectors.items() 
                                 if k in selected_detectors}
        else:
            available_detectors = self.detectors
        
        # Run evaluation for each detector
        results = {}
        comparison_data = {}
        
        for detector_name, detector in available_detectors.items():
            logger.info(f"Evaluating {detector_name}")
            
            try:
                # Run detection
                detection_results = detector.evaluate_batch(samples)
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_metrics(
                    detection_results["true_labels"],
                    detection_results["predictions"],
                    detection_results["confidences"]
                )
                
                # Add timing information
                metrics.update({
                    "total_time": detection_results["total_time"],
                    "avg_time_per_sample": detection_results["avg_time_per_sample"]
                })
                
                results[detector_name] = {
                    "metrics": metrics,
                    "detailed_results": detector.results,
                    "detection_results": detection_results
                }
                
                # Add decision maker analysis for VulnRAG
                if detector_name.startswith("vulnrag-"):
                    decision_makers = self._analyze_vulnrag_decisions(detector.results)
                    results[detector_name]["decision_analysis"] = decision_makers
                
                comparison_data[detector_name] = metrics
                
                logger.info(f"{detector_name} evaluation completed")
                logger.info(self.metrics_calculator.get_summary())
                
            except Exception as e:
                logger.error(f"Error evaluating {detector_name}: {e}")
                continue
        
        # Generate comparison
        if len(comparison_data) > 1:
            comparison_df = self.metrics_calculator.compare_baselines(comparison_data)
            results["comparison"] = comparison_df.to_dict("records")
        
        # Save results
        self._save_results(results, dataset_stats)
        
        return results
    
    def _save_results(self, results: Dict, dataset_stats: Dict):
        """Save evaluation results"""
        
        # Save detailed results
        results_file = self.output_dir / "detection_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save comparison table
        if "comparison" in results:
            comparison_file = self.output_dir / "comparison_table.json"
            with open(comparison_file, 'w') as f:
                json.dump(results["comparison"], f, indent=2)
        
        # Save dataset stats
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _analyze_vulnrag_decisions(self, results: List[Dict]) -> Dict[str, int]:
        """Analyze who made decisions in VulnRAG pipeline"""
        decision_makers = {
            "STATIC_HEURISTIC_AGREEMENT": 0,
            "LLM_ARBITRATION": 0,
            "FULL_RAG_PIPELINE": 0,
            "UNKNOWN": 0
        }
        
        for result in results:
            # Use decision_analysis field if available
            if "decision_analysis" in result:
                decision_maker = result["decision_analysis"]
                if decision_maker in decision_makers:
                    decision_makers[decision_maker] += 1
                else:
                    decision_makers["UNKNOWN"] += 1
            else:
                # Fallback to timings analysis
                timings = result.get("timings_s", {})
                
                if "preprocessing" in timings and "retrieval" in timings and "assembly" in timings:
                    decision_makers["FULL_RAG_PIPELINE"] += 1
                elif "llm_arbitration" in timings:
                    decision_makers["LLM_ARBITRATION"] += 1
                elif "static" in timings and len(timings) <= 2:
                    decision_makers["STATIC_HEURISTIC_AGREEMENT"] += 1
                else:
                    decision_makers["UNKNOWN"] += 1
        
        return decision_makers
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        
        print("\n" + "="*60)
        print("DETECTION EVALUATION SUMMARY")
        print("="*60)
        
        if "comparison" in results:
            print("\nComparison Table:")
            print("-" * 60)
            print(f"{'Detector':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 60)
            
            for row in results["comparison"]:
                print(f"{row['Baseline']:<15} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} "
                      f"{row['Recall']:<10.4f} {row['F1-Score']:<10.4f}")
        
        print("\nDetailed Results:")
        print("-" * 60)
        
        for detector_name, result in results.items():
            if detector_name == "comparison":
                continue
                
            metrics = result["metrics"]
            print(f"\n{detector_name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  AUC-ROC:   {metrics.get('auc_roc', 'N/A')}")
            print(f"  Time:      {metrics['total_time']:.2f}s ({metrics['avg_time_per_sample']:.3f}s/sample)")
            
            # Show VulnRAG decision analysis
            if detector_name.startswith("vulnrag-") and "decision_analysis" in result:
                print(f"  Decision Analysis:")
                for decision_maker, count in result["decision_analysis"].items():
                    if count > 0:
                        print(f"    {decision_maker}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Run detection evaluation")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--detectors", nargs="+", 
                       choices=["vulnrag-qwen2.5", "vulnrag-kirito", "qwen2.5", "kirito", "gpt", "static"],
                       help="Specific detectors to evaluate")
    parser.add_argument("--output-dir", type=str, 
                       default="evaluation/detection/results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run evaluation
    runner = DetectionEvaluationRunner(args.output_dir)
    results = runner.run_evaluation(
        max_samples=args.max_samples,
        selected_detectors=args.detectors
    )
    
    # Print summary
    runner.print_summary(results)
    
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 