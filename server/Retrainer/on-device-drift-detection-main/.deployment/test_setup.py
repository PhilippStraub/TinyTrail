#!/usr/bin/env python3
"""
Test script to verify TinyLCM setup on Raspberry Pi
"""

import sys
import json
import os
from pathlib import Path

def test_imports():
    """Test all required imports"""
    print("Testing Python imports...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not available")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV not available")
        return False
    
    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn not available")
        return False
    
    try:
        import psutil
        print(f"✓ psutil {psutil.__version__}")
    except ImportError:
        print("✗ psutil not available")
        return False
    
    try:
        import tflite_runtime.interpreter as tflite
        print("✓ TFLite runtime available")
    except ImportError:
        print("⚠ TFLite runtime not available")
        return False
    
    return True

def test_tinylcm_imports():
    """Test TinyLCM library imports"""
    print("\nTesting TinyLCM imports...")
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from tinylcm_python.feature_extractor import TFLiteFeatureExtractor
        print("✓ TFLiteFeatureExtractor")
    except ImportError as e:
        print(f"✗ TFLiteFeatureExtractor: {e}")
        return False
    
    try:
        from tinylcm_python.knn import LightweightKNN
        print("✓ LightweightKNN")
    except ImportError as e:
        print(f"✗ LightweightKNN: {e}")
        return False
    
    try:
        from tinylcm_python.drift_monitor import KNNDistanceMonitor
        print("✓ KNNDistanceMonitor")
    except ImportError as e:
        print(f"✗ KNNDistanceMonitor: {e}")
        return False
    
    try:
        from tinylcm_python.scaler_pca import StandardScalerPCATransformer
        print("✓ StandardScalerPCATransformer")
    except ImportError as e:
        print(f"✗ StandardScalerPCATransformer: {e}")
        return False
    
    return True

def test_scenario_structure():
    """Test scenario directory structure"""
    print("\nTesting scenario structure...")
    
    project_root = Path(__file__).parent.parent
    
    # Test Scenario 1
    s1_path = project_root / "examples/python_examples/scenario_lego_drift"
    if s1_path.exists():
        print("✓ Scenario 1 directory exists")
        
        # Check for key files
        if (s1_path / "main.py").exists():
            print("  ✓ main.py")
        else:
            print("  ✗ main.py missing")
        
        if (s1_path / "config.json").exists():
            print("  ✓ config.json")
        else:
            print("  ✗ config.json missing")
        
        if (s1_path / "model").exists():
            print("  ✓ model directory")
            model_files = list((s1_path / "model").glob("*"))
            print(f"    Files: {[f.name for f in model_files]}")
        else:
            print("  ✗ model directory missing")
        
        # Check initial_state directory
        initial_state_dir = s1_path / "initial_state"
        if initial_state_dir.exists():
            print("  ✓ initial_state directory")
            state_files = list(initial_state_dir.glob("*"))
            if state_files:
                print(f"    Files: {[f.name for f in state_files]}")
            else:
                print("    (empty - run create_initial_state.py)")
        else:
            print("  ⚠ initial_state directory missing (will be created)")
    else:
        print("✗ Scenario 1 directory missing")
    
    # Test Scenario 2 
    s2_path = project_root / "examples/python_examples/scenario_ball_drift"
    if s2_path.exists():
        print("✓ Scenario 2 directory exists")
        
        if (s2_path / "main.py").exists():
            print("  ✓ main.py")
        else:
            print("  ✗ main.py missing")
        
        if (s2_path / "config.json").exists():
            print("  ✓ config.json")
        else:
            print("  ✗ config.json missing")
            
        if (s2_path / "model").exists():
            print("  ✓ model directory")
        else:
            print("  ✗ model directory missing")
    else:
        print("⚠ Scenario 2 directory not found (optional)")

def test_camera():
    """Test camera availability"""
    print("\nTesting camera...")
    
    try:
        import cv2
        # Try to open camera
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✓ Camera working - Resolution: {frame.shape}")
            else:
                print("⚠ Camera detected but no frames")
            cap.release()
        else:
            print("✗ No camera detected")
    except Exception as e:
        print(f"✗ Camera test failed: {e}")

def test_config_files():
    """Test configuration files"""
    print("\nTesting configuration...")
    
    project_root = Path(__file__).parent.parent
    
    # Test Scenario 1 config
    config_path = project_root / "examples/python_examples/scenario_lego_drift/config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print("✓ Scenario 1 config.json valid")
            
            # Check key paths
            model_path = config.get("model", {}).get("model_path", "")
            if model_path.startswith("./"):
                print("  ✓ Relative paths used")
            else:
                print("  ⚠ Absolute paths might cause issues")
                
        except json.JSONDecodeError:
            print("✗ Scenario 1 config.json invalid JSON")
    else:
        print("✗ Scenario 1 config.json missing")

def main():
    print("TinyLCM Pi Setup Test")
    print("=" * 40)
    
    all_good = True
    
    # Run all tests
    all_good &= test_imports()
    all_good &= test_tinylcm_imports()
    test_scenario_structure()
    test_camera()
    test_config_files()
    
    print("\n" + "=" * 40)
    if all_good:
        print("✓ All critical tests passed!")
        print("Ready to run scenarios.")
    else:
        print("✗ Some tests failed.")
        print("Check the output above and install missing dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()