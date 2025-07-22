"""
Dataset loader for vulnerability detection evaluation
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class DetectionDatasetLoader:
    """Load and manage detection dataset"""
    
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            # Try to find the dataset in the project structure
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir / "../../rag/data/Datasets/dataset_detection_clean.csv",
                current_dir / "../../rag/data/Datasets/dataset_detection.csv",
                Path("rag/data/Datasets/dataset_detection_clean.csv"),
                Path("rag/data/Datasets/dataset_detection.csv")
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.dataset_path = path
                    break
            else:
                raise FileNotFoundError(f"Dataset not found in any of the expected locations: {possible_paths}")
        else:
            self.dataset_path = Path(dataset_path)
        self.data = None
        self.stats = {}
        
    def load_dataset(self) -> pd.DataFrame:
        """Load detection dataset"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
        logger.info(f"Loading detection dataset from {self.dataset_path}")
        self.data = pd.read_csv(self.dataset_path)
        
        # Calculate statistics
        self._calculate_stats()
        
        logger.info(f"Dataset loaded: {len(self.data)} samples")
        logger.info(f"Statistics: {self.stats}")
        
        return self.data
    
    def _calculate_stats(self):
        """Calculate dataset statistics"""
        if self.data is None:
            return
            
        self.stats = {
            "total_samples": len(self.data),
            "vulnerable_samples": len(self.data[self.data['label'] == 1]),
            "non_vulnerable_samples": len(self.data[self.data['label'] == 0]),
            "vulnerability_rate": len(self.data[self.data['label'] == 1]) / len(self.data)
        }
    
    def get_samples(self, max_samples: int = None) -> List[Dict]:
        """Get samples for evaluation"""
        if self.data is None:
            self.load_dataset()
            
        samples = []
        data_subset = self.data
        
        if max_samples:
            data_subset = self.data.head(max_samples)
            
        for idx, row in data_subset.iterrows():
            sample = {
                "id": idx,
                "func": row["func"],
                "label": int(row["label"]),
                "is_vulnerable": bool(row["label"])
            }
            samples.append(sample)
            
        logger.info(f"Returning {len(samples)} samples for evaluation")
        return samples
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self.stats:
            self._calculate_stats()
        return self.stats 