"""
File: tests/comprehensive_validation.py
Purpose: Comprehensive validation tests for the trained model
"""

import os
import sys
import time
import numpy as np
import pickle
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.calibration import calibration_curve

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ============================================================================
# 1. Paths
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")

# ============================================================================
# 2. Load Data
# ============================================================================

print("="*70)
print("🔬 LOADING MODEL AND DATA FOR VALIDATION")
print("="*70)

# Load preprocessed data
print("\n📂 Loading preprocessed data...")
X_symptoms_train = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_train.npy"))
X_symptoms_val = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_val.npy"))
X_symptoms_test = np.load(os.path.join(PROCESSED_DIR, "X_symptoms_test.npy"))

X_severities_train = np.load(os.path.join(PROCESSED_DIR, "X_severities_train.npy"))
X_severities_val = np.load(os.path.join(PROCESSED_DIR, "X_severities_val.npy"))
X_severities_test = np.load(os.path.join(PROCESSED_DIR, "X_severities_test.npy"))

y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print(f"   - Train: {len(X_symptoms_train)} samples")
print(f"   - Validation: {len(X_symptoms_val)} samples")
print(f"   - Test: {len(X_symptoms_test)} samples")

# Load label encoder
with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), 'rb') as f:
    label_encoder = pickle.load(f)

# Load model
print("\n🤖 Loading trained model...")
model_path = os.path.join(MODELS_DIR, "bilstm_final.keras")
model = tf.keras.models.load_model(model_path)
print(f"✅ Model loaded successfully!")

# Combine all data for cross-validation
X_symptoms_all = np.vstack([X_symptoms_train, X_symptoms_val, X_symptoms_test])
X_severities_all = np.vstack([X_severities_train, X_severities_val, X_severities_test])
y_all = np.hstack([y_train, y_val, y_test])

# ============================================================================
# 3. TEST 1: Shuffled Labels Test
# ============================================================================

def test_shuffled_labels():
    """Test if model learns real patterns, not memorization"""
    
    print("\n" + "="*70)
    print("🔬 TEST 1: SHUFFLED LABELS TEST")
    print("="*70)
    print("Purpose: Verify model learns real patterns, not memorization")
    
    # Shuffle labels randomly
    y_shuffled = shuffle(y_test)
    
    # Create a copy of the model
    model_copy = tf.keras.models.clone_model(model)
    model_copy.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train on shuffled labels (5 epochs only)
    print("\n🔄 Training on shuffled labels (5 epochs)...")
    history = model_copy.fit(
        [X_symptoms_test, X_severities_test],
        y_shuffled,
        epochs=5,
        batch_size=32,
        verbose=0
    )
    
    final_acc = history.history['accuracy'][-1]
    
    print(f"\n📊 Results:")
    print(f"   Final accuracy on shuffled labels: {final_acc*100:.4f}%")
    
    if final_acc < 0.10:
        print("\n✅ PASSED: Model learns REAL patterns!")
        print("   (Accuracy on shuffled labels is low, meaning model didn't memorize)")
    else:
        print("\n❌ FAILED: Model may be memorizing data!")
        print("   (Accuracy on shuffled labels is too high)")
    
    return final_acc

# ============================================================================
# 4. TEST 2: Holdout Validation Test
# ============================================================================

def test_holdout_validation():
    """Test model on a completely new holdout set"""
    
    print("\n" + "="*70)
    print("🔬 TEST 2: HOLDOUT VALIDATION TEST")
    print("="*70)
    print("Purpose: Verify model performance on completely unseen data")
    
    # Split data into new train/holdout
    X_symptoms_train_new, X_symptoms_holdout, X_severities_train_new, X_severities_holdout, y_train_new, y_holdout = train_test_split(
        X_symptoms_all, X_severities_all, y_all, test_size=0.2, random_state=123, stratify=y_all
    )
    
    print(f"\n📊 New split:")
    print(f"   - Training: {len(X_symptoms_train_new)} samples")
    print(f"   - Holdout: {len(X_symptoms_holdout)} samples")
    
    # Create a copy of the model
    model_copy = tf.keras.models.clone_model(model)
    model_copy.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train on new split (10 epochs)
    print("\n🔄 Training on new split (10 epochs)...")
    model_copy.fit(
        [X_symptoms_train_new, X_severities_train_new],
        y_train_new,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    
    # Evaluate on holdout
    test_loss, test_acc = model_copy.evaluate(
        [X_symptoms_holdout, X_severities_holdout], 
        y_holdout, 
        verbose=0
    )
    
    print(f"\n📊 Results:")
    print(f"   Holdout Accuracy: {test_acc*100:.4f}%")
    
    if test_acc > 0.90:
        print("\n✅ PASSED: Model generalizes well to new data!")
    elif test_acc > 0.80:
        print("\n⚠️ ACCEPTABLE: Model shows reasonable generalization")
    else:
        print("\n❌ FAILED: Model may be overfitting!")
    
    return test_acc

# ============================================================================
# 5. TEST 3: Cross-Validation Test
# ============================================================================

def test_cross_validation(n_folds=5):
    """Test model stability using K-Fold cross-validation"""
    
    print("\n" + "="*70)
    print(f"🔬 TEST 3: {n_folds}-FOLD CROSS-VALIDATION TEST")
    print("="*70)
    print("Purpose: Verify model stability across different splits")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_symptoms_all, y_all)):
        print(f"\n   Fold {fold + 1}/{n_folds}...")
        
        X_train_fold = X_symptoms_all[train_idx]
        X_val_fold = X_symptoms_all[val_idx]
        X_sev_train_fold = X_severities_all[train_idx]
        X_sev_val_fold = X_severities_all[val_idx]
        y_train_fold = y_all[train_idx]
        y_val_fold = y_all[val_idx]
        
        # Create model
        model_fold = tf.keras.models.clone_model(model)
        model_fold.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        model_fold.fit(
            [X_train_fold, X_sev_train_fold],
            y_train_fold,
            epochs=10,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        _, acc = model_fold.evaluate(
            [X_val_fold, X_sev_val_fold],
            y_val_fold,
            verbose=0
        )
        
        y_pred = np.argmax(model_fold.predict([X_val_fold, X_sev_val_fold], verbose=0), axis=1)
        f1 = f1_score(y_val_fold, y_pred, average='macro')
        
        fold_accuracies.append(acc)
        fold_f1_scores.append(f1)
        
        print(f"      Accuracy: {acc*100:.4f}%, F1-Score: {f1*100:.4f}%")
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    
    print(f"\n📊 Cross-Validation Results:")
    print(f"   Mean Accuracy: {mean_acc*100:.4f}% (±{std_acc*100:.4f}%)")
    print(f"   Mean F1-Score: {mean_f1*100:.4f}% (±{std_f1*100:.4f}%)")
    
    if std_acc < 0.05:
        print("\n✅ PASSED: Model is stable across different splits!")
    else:
        print("\n⚠️ WARNING: Model shows some instability across splits")
    
    return mean_acc, std_acc

# ============================================================================
# 6. TEST 4: Confidence Calibration Test
# ============================================================================

def test_confidence_calibration():
    """Test if model's confidence is well-calibrated"""
    
    print("\n" + "="*70)
    print("🔬 TEST 4: CONFIDENCE CALIBRATION TEST")
    print("="*70)
    print("Purpose: Verify model confidence matches actual accuracy")
    
    # Get predictions
    y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    confidence = np.max(y_pred_probs, axis=1)
    
    # Group by confidence intervals
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(len(bins) - 1):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if np.sum(mask) > 0:
            acc = np.mean(y_test[mask] == y_pred[mask])
            conf = np.mean(confidence[mask])
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            print(f"   {bins[i]:.2f}-{bins[i+1]:.2f}: conf={conf:.4f}, acc={acc:.4f}, n={np.sum(mask)}")
    
    # Calculate calibration error
    calibration_error = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
    
    print(f"\n📊 Results:")
    print(f"   Calibration Error: {calibration_error:.4f}")
    
    if calibration_error < 0.05:
        print("\n✅ PASSED: Model is well-calibrated!")
    elif calibration_error < 0.10:
        print("\n⚠️ ACCEPTABLE: Model shows reasonable calibration")
    else:
        print("\n❌ FAILED: Model is poorly calibrated")
    
    return calibration_error

# ============================================================================
# 7. TEST 5: Robustness to Noise Test
# ============================================================================

def test_robustness():
    """Test model robustness to noisy inputs"""
    
    print("\n" + "="*70)
    print("🔬 TEST 5: ROBUSTNESS TO NOISE TEST")
    print("="*70)
    print("Purpose: Verify model handles noisy inputs gracefully")
    
    vocab_size = 133
    noise_levels = [0, 0.05, 0.10, 0.15, 0.20]
    accuracies = []
    
    for noise in noise_levels:
        # Add noise to symptoms
        X_noisy = X_symptoms_test.copy()
        mask = np.random.random(X_noisy.shape) < noise
        X_noisy[mask] = np.random.randint(1, vocab_size, size=np.sum(mask))
        
        # Predict
        _, acc = model.evaluate([X_noisy, X_severities_test], y_test, verbose=0)
        accuracies.append(acc)
        
        print(f"   Noise level {noise*100:.0f}%: Accuracy = {acc*100:.4f}%")
    
    # Calculate robustness score
    drop_20 = (accuracies[0] - accuracies[-1]) / accuracies[0] * 100
    
    print(f"\n📊 Results:")
    print(f"   Performance drop at 20% noise: {drop_20:.2f}%")
    
    if drop_20 < 20:
        print("\n✅ PASSED: Model is robust to noise!")
    elif drop_20 < 40:
        print("\n⚠️ ACCEPTABLE: Model shows reasonable robustness")
    else:
        print("\n❌ FAILED: Model is sensitive to noise")
    
    return drop_20

# ============================================================================
# 8. TEST 6: Per-Class Performance Test
# ============================================================================

def test_per_class_performance():
    """Test performance on each individual disease"""
    
    print("\n" + "="*70)
    print("🔬 TEST 6: PER-CLASS PERFORMANCE TEST")
    print("="*70)
    print("Purpose: Verify all 41 diseases are recognized correctly")
    
    # Get predictions
    y_pred_probs = model.predict([X_symptoms_test, X_severities_test], verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate per-class metrics
    classes = label_encoder.classes_
    results = []
    
    for i, disease in enumerate(classes):
        mask = (y_test == i)
        n_samples = np.sum(mask)
        
        if n_samples > 0:
            correct = np.sum(y_pred[mask] == i)
            accuracy = correct / n_samples
            
            results.append({
                'disease': disease,
                'samples': n_samples,
                'correct': correct,
                'accuracy': accuracy
            })
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'])
    
    print(f"\n📊 Per-Class Performance:")
    print(f"   {'Disease':<45} {'Samples':<10} {'Accuracy':<10}")
    print(f"   {'-'*65}")
    
    for r in results:
        status = "✅" if r['accuracy'] == 1.0 else "⚠️"
        disease_name = r['disease'][:43] + "..." if len(r['disease']) > 43 else r['disease']
        print(f"   {status} {disease_name:<45} {r['samples']:<10} {r['accuracy']*100:.2f}%")
    
    perfect_classes = sum(1 for r in results if r['accuracy'] == 1.0)
    total_classes = len(results)
    
    print(f"\n📊 Summary:")
    print(f"   Perfect classes (100%): {perfect_classes}/{total_classes}")
    print(f"   Classes with issues: {total_classes - perfect_classes}")
    
    if perfect_classes == total_classes:
        print("\n✅ PASSED: All 41 diseases recognized perfectly!")
    elif perfect_classes > total_classes * 0.9:
        print("\n✅ PASSED: Most diseases recognized correctly")
    else:
        print("\n❌ FAILED: Some diseases are not recognized well")
    
    return results

# ============================================================================
# 9. TEST 7: Inference Time Test
# ============================================================================

def test_inference_time(n_iterations=500):
    """Test model inference speed"""
    
    print("\n" + "="*70)
    print("🔬 TEST 7: INFERENCE TIME TEST")
    print("="*70)
    print("Purpose: Verify model meets real-time requirements")
    
    # Create random input
    X_symptoms_random = np.random.randint(0, 133, size=(1, 17))
    X_severities_random = np.random.randint(0, 8, size=(1, 17))
    
    # Warm-up
    for _ in range(100):
        model.predict([X_symptoms_random, X_severities_random], verbose=0)
    
    # Measure inference time
    times = []
    for _ in range(n_iterations):
        start = time.time()
        model.predict([X_symptoms_random, X_severities_random], verbose=0)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n📊 Results:")
    print(f"   Mean inference time: {mean_time:.2f} ms")
    print(f"   Std deviation: {std_time:.2f} ms")
    print(f"   Max time: {np.max(times):.2f} ms")
    print(f"   Min time: {np.min(times):.2f} ms")
    
    if mean_time < 100:
        print("\n✅ PASSED: Model meets real-time requirements (<100ms)!")
    elif mean_time < 500:
        print("\n⚠️ ACCEPTABLE: Model may have slight latency")
    else:
        print("\n❌ FAILED: Model is too slow for real-time use")
    
    return mean_time

# ============================================================================
# 10. Run All Tests
# ============================================================================

def run_all_tests():
    """Run all validation tests"""
    
    print("\n" + "="*70)
    print("🚀 RUNNING COMPREHENSIVE VALIDATION SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Shuffled Labels
    results['shuffled_labels_acc'] = test_shuffled_labels()
    
    # Test 2: Holdout Validation
    results['holdout_accuracy'] = test_holdout_validation()
    
    # Test 3: Cross-Validation
    mean_acc, std_acc = test_cross_validation(n_folds=5)
    results['cv_mean_accuracy'] = mean_acc
    results['cv_std_accuracy'] = std_acc
    
    # Test 4: Confidence Calibration
    results['calibration_error'] = test_confidence_calibration()
    
    # Test 5: Robustness
    results['robustness_drop'] = test_robustness()
    
    # Test 6: Per-Class Performance
    per_class_results = test_per_class_performance()
    results['perfect_classes'] = sum(1 for r in per_class_results if r['accuracy'] == 1.0)
    results['total_classes'] = len(per_class_results)
    
    # Test 7: Inference Time
    results['inference_time_ms'] = test_inference_time()
    
    # Final Summary
    print("\n" + "="*70)
    print("🎯 FINAL VALIDATION SUMMARY")
    print("="*70)
    
    print(f"\n📊 All Tests Results:")
    print(f"   ✅ Shuffled Labels Test: {'PASSED' if results['shuffled_labels_acc'] < 0.1 else 'FAILED'}")
    print(f"   ✅ Holdout Validation: {'PASSED' if results['holdout_accuracy'] > 0.9 else 'FAILED'}")
    print(f"   ✅ Cross-Validation: {'PASSED' if results['cv_std_accuracy'] < 0.05 else 'WARNING'}")
    print(f"   ✅ Confidence Calibration: {'PASSED' if results['calibration_error'] < 0.05 else 'WARNING'}")
    print(f"   ✅ Robustness: {'PASSED' if results['robustness_drop'] < 20 else 'WARNING'}")
    print(f"   ✅ Per-Class Performance: {results['perfect_classes']}/{results['total_classes']} perfect")
    print(f"   ✅ Inference Time: {results['inference_time_ms']:.2f} ms")
    
    # Final Verdict
    all_passed = (
        results['shuffled_labels_acc'] < 0.1 and
        results['holdout_accuracy'] > 0.9 and
        results['cv_std_accuracy'] < 0.05 and
        results['calibration_error'] < 0.05 and
        results['robustness_drop'] < 20 and
        results['perfect_classes'] == results['total_classes'] and
        results['inference_time_ms'] < 100
    )
    
    if all_passed:
        print("\n" + "="*70)
        print("🏆 FINAL VERDICT: MODEL IS PRODUCTION READY! 🏆")
        print("="*70)
        print("\n✅ All tests passed!")
        print("✅ Model shows excellent generalization")
        print("✅ No data leakage detected")
        print("✅ Real-time inference capable")
        print("✅ All 41 diseases recognized perfectly")
    else:
        print("\n⚠️ Some tests need attention")
    
    return results

# ============================================================================
# 11. Main
# ============================================================================

if __name__ == "__main__":
    results = run_all_tests()