"""
Quick test for detection evaluation framework
"""

import logging
from pathlib import Path

from dataset_loader import DetectionDatasetLoader
from detection_metrics import DetectionMetricsCalculator
from baselines import get_available_detectors

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_loader():
    """Test dataset loader"""
    print("🧪 Testing Dataset Loader...")
    
    try:
        loader = DetectionDatasetLoader()
        data = loader.load_dataset()
        stats = loader.get_statistics()
        
        print(f"✅ Dataset loaded successfully")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Vulnerable: {stats['vulnerable_samples']}")
        print(f"   Non-vulnerable: {stats['non_vulnerable_samples']}")
        
        # Test getting samples
        samples = loader.get_samples(max_samples=5)
        print(f"   Retrieved {len(samples)} test samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loader failed: {e}")
        return False

def test_metrics_calculator():
    """Test metrics calculator"""
    print("\n🧪 Testing Metrics Calculator...")
    
    try:
        calculator = DetectionMetricsCalculator()
        
        # Test with dummy data
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        y_scores = [0.9, 0.1, 0.3, 0.2, 0.8]
        
        metrics = calculator.calculate_metrics(y_true, y_pred, y_scores)
        
        print(f"✅ Metrics calculated successfully")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metrics calculator failed: {e}")
        return False

def test_detectors():
    """Test available detectors"""
    print("\n🧪 Testing Detectors...")
    
    try:
        detectors = get_available_detectors()
        
        print(f"✅ Found {len(detectors)} available detectors:")
        for name, detector in detectors.items():
            print(f"   - {name}: {detector.name}")
        
        return len(detectors) > 0
        
    except Exception as e:
        print(f"❌ Detectors test failed: {e}")
        return False

def test_static_tools():
    """Test static tools specifically"""
    print("\n🧪 Testing Static Tools...")
    
    try:
        from baselines import StaticToolsDetector
        
        detector = StaticToolsDetector()
        
        # Test with simple vulnerable code
        test_code = """
        void vulnerable_function(char* input) {
            char buffer[10];
            strcpy(buffer, input);  // Buffer overflow
        }
        """
        
        pred, conf = detector.detect(test_code)
        print(f"✅ Static tools test completed")
        print(f"   Prediction: {pred}, Confidence: {conf:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Static tools test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Quick Test for Detection Evaluation Framework")
    print("=" * 60)
    
    tests = [
        ("Dataset Loader", test_dataset_loader),
        ("Metrics Calculator", test_metrics_calculator),
        ("Detectors", test_detectors),
        ("Static Tools", test_static_tools)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main() 