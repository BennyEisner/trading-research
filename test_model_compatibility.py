#!/usr/bin/env python3

"""
Test script to verify TensorFlow/Keras model compatibility
Run this to ensure all model code works with updated versions
"""

import sys
import numpy as np

if sys.version_info < (3, 12):
    raise RuntimeError("This test requires Python 3.12 or higher")

def test_tensorflow_import():
    """Test TensorFlow can be imported and basic functionality works"""
    print("Testing TensorFlow import...")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test basic operations
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        z = tf.add(x, y)
        print(f"Basic TF operations work: {z.numpy()}")
        
        return True
    except Exception as e:
        print(f"TensorFlow import failed: {e}")
        return False

def test_model_imports():
    """Test that all model modules can be imported"""
    print("\nTesting model imports...")
    try:
        from src.models.multi_scale_lstm import MultiScaleLSTMBuilder
        print("MultiScaleLSTMBuilder imported")
        
        from src.models.directional_loss import DirectionalLoss
        print("DirectionalLoss imported")
        
        from src.models.model_builder import ModelBuilder
        print("ModelBuilder imported")
        
        return True
    except Exception as e:
        print(f"Model import failed: {e}")
        return False

def test_simple_model_creation():
    """Test creating a simple model with updated syntax"""
    print("\nTesting simple model creation...")
    try:
        import tensorflow as tf
        
        # Create simple sequential model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        # Compile with Adam optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.mean_squared_error,
            metrics=['mae']
        )
        
        print("Simple model created and compiled")
        
        # Test with dummy data
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 1)
        
        # Test fit (1 epoch)
        model.fit(X, y, epochs=1, verbose=0)
        print("Model training works")
        
        # Test prediction
        pred = model.predict(X[:5], verbose=0)
        print(f"Model prediction works: shape {pred.shape}")
        
        return True
    except Exception as e:
        print(f"Model creation/training failed: {e}")
        return False

def test_directional_loss():
    """Test directional loss functions"""
    print("\nTesting directional loss functions...")
    try:
        from src.models.directional_loss import DirectionalLoss
        import tensorflow as tf
        
        # Create dummy data
        y_true = tf.constant([0.1, -0.05, 0.02, -0.1, 0.03])
        y_pred = tf.constant([0.08, -0.02, 0.01, -0.12, 0.05])
        
        # Test directional MSE loss
        loss = DirectionalLoss.directional_mse_loss(y_true, y_pred, alpha=0.4)
        print(f"Directional MSE loss: {loss.numpy():.4f}")
        
        # Test asymmetric loss
        asym_loss = DirectionalLoss.asymmetric_directional_loss(y_true, y_pred)
        print(f"Asymmetric directional loss: {asym_loss.numpy():.4f}")
        
        return True
    except Exception as e:
        print(f"Directional loss test failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("TensorFlow/Keras Model Compatibility Test")
    print("=" * 50)
    
    tests = [
        test_tensorflow_import,
        test_model_imports,
        test_simple_model_creation,
        test_directional_loss
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"{test.__name__} failed")
        except Exception as e:
            print(f"{test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All compatibility tests passed! Models are ready for TensorFlow 2.18+")
        return True
    else:
        print("Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)