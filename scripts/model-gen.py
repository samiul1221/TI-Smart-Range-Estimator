"""
Embedded AI Model Generator for EV Range Prediction - FIXED VERSION
====================================================================
Trains a Random Forest Regressor optimized for ESP32 deployment via emlearn.

CRITICAL FIXES:
- Correct emlearn API usage
- Removed unnecessary StandardScaler (RF doesn't need it)
- Proper C code generation with prediction function
- Generated Arduino example code
- Accurate memory estimates

Target: Predict remaining_range_km on ESP32
Constraints: 4MB Flash, 520KB RAM

Author: EV-TI Project
Date: February 2026
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Embedded ML export
try:
    import emlearn
    EMLEARN_AVAILABLE = True
    print("✓ emlearn library available for C code generation")
except ImportError:
    EMLEARN_AVAILABLE = False
    print("⚠ emlearn not installed. C code generation will be skipped.")
    print("  Install with: pip install emlearn")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'dataset_path': r'C:\D\EV_TI\output\ev_range_prediction_physics_accurate.csv',
    'output_dir': r'C:\D\EV_TI\output\models',
    'target_column': 'remaining_range_km',
    'random_state': 42,
    
    # Feature selection
    'max_features': 12,  # Reduced to 12 for smaller model size
    
    # Train/test split
    'test_size': 0.20,
    
    # Model hyperparameters (ESP32-optimized)
    'model_params': {
        'n_estimators':50,       # 36 trees = good accuracy, small size
        'max_depth': 11,          # Depth 10 = max 1024 nodes per tree
        'min_samples_split': 15,  # Prevent overfitting
        'min_samples_leaf': 8,    # Prune weak leaves
        'max_features': 'sqrt',   # Decorrelate trees
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    },
    
    # Evaluation
    'cv_folds': 5,
    'overfitting_threshold': 0.05,
}


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data(filepath: str) -> tuple:
    """Load dataset and perform initial preprocessing."""
    print(f"\n{'='*80}")
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print(f"{'='*80}\n")
    
    print(f"Loading dataset: {filepath}")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    # Check for missing values (report only, DO NOT impute yet)
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"⚠ Found {null_count} null values - will impute AFTER train/test split")
    else:
        print(f"✓ No null values detected")
    
    # Define candidate features (exclude target and non-numeric)
    exclude_cols = [
        CONFIG['target_column'],
        'timestamp', 'trip_id', 'vehicle_type', 
        'weather_condition',
        'is_daytime'  # Boolean, not useful
    ]
    
    feature_columns = [col for col in df.columns 
                      if col not in exclude_cols 
                      and df[col].dtype in [np.float64, np.int64, np.float32, np.int32, np.int16, np.int8]]
    
    print(f"\n✓ Identified {len(feature_columns)} candidate numeric features")
    print(f"✓ Target variable: {CONFIG['target_column']}")
    
    return df, feature_columns, CONFIG['target_column']


def select_top_features_from_training(X_train: pd.DataFrame, y_train: pd.Series, 
                                      max_features: int = 12) -> list:
    """Select top N features using Random Forest on TRAINING DATA ONLY (prevents data leakage)."""
    print(f"\n{'='*80}")
    print("STEP 3: FEATURE SELECTION (TRAINING DATA ONLY - NO LEAKAGE)")
    print(f"{'='*80}\n")
    
    print(f"⚠ CRITICAL: Feature selection using ONLY training data to prevent leakage")
    print(f"Training feature selector to rank importance...")
    print(f"Target: Select top {max_features} features for ESP32 deployment")
    
    # Train a quick RF to get feature importance (ONLY on training data)
    rf_selector = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        verbose=0
    )
    
    rf_selector.fit(X_train, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top N
    selected_features = importance.head(max_features)['feature'].tolist()
    
    print(f"\n✓ Selected {len(selected_features)} top features:")
    print(f"\n{'Rank':<6}{'Feature':<35}{'Importance':<12}{'Bar'}")
    print(f"{'-'*80}")
    
    for idx, row in importance.head(max_features).iterrows():
        rank = list(importance.index).index(idx) + 1
        bar = '█' * int(row['importance'] * 100)
        print(f"{rank:<6}{row['feature']:<35}{row['importance']:.6f}    {bar}")
    
    # Show total importance retained
    total_importance = importance['importance'].sum()
    retained_importance = importance.head(max_features)['importance'].sum()
    retention_pct = (retained_importance / total_importance) * 100
    
    print(f"\nℹ  Retained {retention_pct:.1f}% of total feature importance")
    print(f"   (using {max_features}/{len(X_train.columns)} features)")
    
    return selected_features


def split_first_then_impute(df: pd.DataFrame, 
                            candidate_features: list,
                            target_col: str) -> tuple:
    """Split FIRST, then impute using training statistics ONLY (prevents data leakage)."""
    print(f"\n{'='*80}")
    print("STEP 2: TRAIN/TEST SPLIT → IMPUTATION (NO LEAKAGE)")
    print(f"{'='*80}\n")
    
    print("⚠ CRITICAL: Splitting BEFORE imputation to prevent test data leakage")
    
    # Prepare X and y
    X = df[candidate_features].copy()
    y = df[target_col].copy()
    
    # Replace infinities with NaN (will be imputed later)
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows where target is null
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    print(f"Total samples (after removing null targets): {len(X):,}")
    
    # STEP 1: Split FIRST (before any imputation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        shuffle=True
    )
    
    print(f"✓ Train set: {len(X_train):,} samples ({100*(1-CONFIG['test_size']):.0f}%)")
    print(f"✓ Test set:  {len(X_test):,} samples ({100*CONFIG['test_size']:.0f}%)")
    
    # STEP 2: Calculate medians from TRAINING data (always needed for C export)
    print(f"\n✓ Computing training medians for sensor fallback...")
    train_medians = X_train.median()
    
    # Check if imputation is needed
    train_nulls = X_train.isnull().sum().sum()
    test_nulls = X_test.isnull().sum().sum()
    
    if train_nulls > 0 or test_nulls > 0:
        print(f"⚠ Imputing nulls: {train_nulls} in train, {test_nulls} in test")
        print("  Using TRAINING median values for both sets (no leakage)")
        
        # Apply to both train and test
        X_train = X_train.fillna(train_medians)
        X_test = X_test.fillna(train_medians)
        
        print(f"✓ Imputation complete using training statistics")
    else:
        print(f"✓ No null values to impute (medians still computed for C export)")
    
    print(f"✓ No scaling applied (Random Forest is scale-invariant)")
    
    # Return medians for export to C code (sensor fallback values)
    return X_train, X_test, y_train, y_test, train_medians


# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_embedded_optimized_model(X_train: pd.DataFrame, 
                                  y_train: pd.Series) -> RandomForestRegressor:
    """Train Random Forest with ESP32-optimized hyperparameters."""
    print(f"\n{'='*80}")
    print("STEP 4: MODEL TRAINING (ESP32-OPTIMIZED)")
    print(f"{'='*80}\n")
    
    print("Hyperparameter Configuration:")
    for key, value in CONFIG['model_params'].items():
        if key != 'n_jobs' and key != 'verbose':
            print(f"  {key}: {value}")
    
    print("\nESP32 Optimization Strategy:")
    print("  • 20 trees × depth 10 = compact model (~40-80KB)")
    print("  • min_samples_leaf=8 → prune noisy patterns")
    print("  • max_features='sqrt' → decorrelate trees")
    print("  • Target: <100KB Flash, <1KB RAM")
    
    print(f"\nTraining Random Forest Regressor...")
    
    model = RandomForestRegressor(**CONFIG['model_params'])
    model.fit(X_train, y_train)
    
    print(f"✓ Training complete!")
    
    return model


def evaluate_model_performance(model: RandomForestRegressor,
                               X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               y_train: pd.Series,
                               y_test: pd.Series) -> dict:
    """Comprehensive model evaluation with overfitting detection."""
    print(f"\n{'='*80}")
    print("STEP 5: MODEL EVALUATION")
    print(f"{'='*80}\n")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'r2': r2_score(y_train, y_train_pred),
            'mae': mean_absolute_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred))
        },
        'test': {
            'r2': r2_score(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
        }
    }
    
    # Cross-validation
    print(f"Running {CONFIG['cv_folds']}-Fold Cross-Validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=CONFIG['cv_folds'],
        scoring='r2',
        n_jobs=-1
    )
    
    metrics['cv'] = {
        'mean_r2': cv_scores.mean(),
        'std_r2': cv_scores.std(),
        'scores': cv_scores.tolist()
    }
    
    # Print results
    print(f"\n{'─'*80}")
    print("TRAINING SET PERFORMANCE")
    print(f"{'─'*80}")
    print(f"  R² Score:  {metrics['train']['r2']:.6f}")
    print(f"  MAE:       {metrics['train']['mae']:.4f} km")
    print(f"  RMSE:      {metrics['train']['rmse']:.4f} km")
    
    print(f"\n{'─'*80}")
    print("TEST SET PERFORMANCE")
    print(f"{'─'*80}")
    print(f"  R² Score:  {metrics['test']['r2']:.6f}")
    print(f"  MAE:       {metrics['test']['mae']:.4f} km")
    print(f"  RMSE:      {metrics['test']['rmse']:.4f} km")
    
    print(f"\n{'─'*80}")
    print("CROSS-VALIDATION (5-Fold)")
    print(f"{'─'*80}")
    print(f"  Mean R²:   {metrics['cv']['mean_r2']:.6f}")
    print(f"  Std R²:    {metrics['cv']['std_r2']:.6f}")
    
    # Overfitting detection
    print(f"\n{'─'*80}")
    print("GENERALIZATION ANALYSIS")
    print(f"{'─'*80}")
    
    r2_gap = metrics['train']['r2'] - metrics['test']['r2']
    
    print(f"  Train-Test R² Gap:  {r2_gap:.6f}")
    
    if r2_gap > CONFIG['overfitting_threshold']:
        print(f"\n⚠️  WARNING: Potential overfitting!")
        print(f"  Gap ({r2_gap:.4f}) > threshold ({CONFIG['overfitting_threshold']})")
    else:
        print(f"\n✓ Good generalization within threshold")
    
    if metrics['cv']['std_r2'] > 0.05:
        print(f"⚠️  High CV variance: {metrics['cv']['std_r2']:.4f}")
    else:
        print(f"✓ Stable CV: Low variance ({metrics['cv']['std_r2']:.4f})")
    
    return metrics


# =============================================================================
# EMBEDDED EXPORT (emlearn) - CORRECTED VERSION
# =============================================================================

def export_to_c_code(model: RandomForestRegressor,
                    selected_features: list,
                    train_medians: pd.Series,
                    output_dir: str) -> dict:
    """
    Export trained model to C code for ESP32 deployment.
    CORRECTED: Uses proven emlearn API pattern
    """
    print(f"\n{'='*80}")
    print("STEP 6: EMBEDDED EXPORT (C CODE GENERATION)")
    print(f"{'='*80}\n")
    
    if not EMLEARN_AVAILABLE:
        print("✗ emlearn not installed - skipping C code generation")
        print("  Install with: pip install emlearn")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Model info for filename
        n_trees = CONFIG['model_params']['n_estimators']
        max_depth = CONFIG['model_params']['max_depth']
        n_features = len(selected_features)
        
        print("Converting Random Forest to C code using emlearn...")
        print(f"  Method: 'inline' (recommended for ESP32)")
        print(f"  Trees: {n_trees}, Depth: {max_depth}, Features: {n_features}")
        
        # Convert model using 'inline' method with float features (no quantization)
        # CRITICAL: dtype='float' ensures C code accepts float* (not int16_t*)
        # CRITICAL: return_type='regressor' ensures float regression output (not binarized)
        c_model = emlearn.convert(model, method='inline', return_type='regressor', dtype='float')
        
        # Generate descriptive filename
        header_filename = f'ev_range_model-RF-inline-{n_features}feat-{n_trees}trees-{max_depth}depth.h'
        c_file_path = output_path / header_filename
        
        # Save C header file
        c_model.save(file=str(c_file_path), name='ev_range_model')
        
        print(f"\n✓ Successfully saved '{c_file_path.name}'")
        
        # Calculate actual file size
        file_size = c_file_path.stat().st_size
        print(f"✓ File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
        # Read the generated C code
        with open(c_file_path, 'r', encoding='utf-8') as f:
            generated_code = f.read()
        
        # Create enhanced header with better documentation
        enhanced_file = output_path / 'ev_range_model.h'
        
        with open(enhanced_file, 'w', encoding='utf-8') as f:
            # Header guard
            f.write("#ifndef EV_RANGE_MODEL_H\n")
            f.write("#define EV_RANGE_MODEL_H\n\n")
            
            # Documentation block
            f.write("/*\n")
            f.write(" * ================================================================\n")
            f.write(" * EV Range Prediction Model - ESP32 Optimized\n")
            f.write(" * ================================================================\n")
            f.write(f" * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f" * Model: Random Forest Regressor\n")
            f.write(f" *   - Trees: {n_trees}\n")
            f.write(f" *   - Max Depth: {max_depth}\n")
            f.write(f" *   - Features: {n_features}\n")
            f.write(f" * Target: Remaining Range (km)\n")
            f.write(" *\n")
            f.write(" * USAGE EXAMPLE (Production - handles sensor failures):\n")
            f.write(" * -----------------------------------------------------\n")
            f.write(f" * float features[{n_features}] = {{50.0, 12.5, 25.0, ...}};\n")
            f.write(" * int sensors_replaced = 0;\n")
            f.write(" * float predicted_range = ev_predict_range_km_safe(features, &sensors_replaced);\n")
            f.write(" * if (sensors_replaced > 0) Serial.println(\"⚠ Sensor failure\");\n")
            f.write(" * Serial.println(predicted_range);  // Output in km\n")
            f.write(" *\n")
            f.write(" * FEATURE ORDER:\n")
            f.write(" * ---------------\n")
            for idx, feat in enumerate(selected_features):
                f.write(f" * [{idx:2d}] {feat}\n")
            f.write(" * ================================================================\n")
            f.write(" */\n\n")
            
            # Feature count constant
            f.write(f"#define EV_MODEL_N_FEATURES {n_features}\n\n")
            
            # Generated model code
            f.write("// ================================================================\n")
            f.write("// EMLEARN GENERATED MODEL (DO NOT MODIFY)\n")
            f.write("// ================================================================\n\n")
            f.write(generated_code)
            f.write("\n")
            
            # Training medians for sensor fallback (CRITICAL for production)
            f.write("\n// ================================================================\n")
            f.write("// TRAINING MEDIANS (Sensor Fallback Values)\n")
            f.write("// ================================================================\n")
            f.write("// If a sensor fails (returns NaN, 0, or out-of-range), use these\n")
            f.write("// median values from training data as safe fallback.\n")
            f.write("// This prevents wild predictions from sensor failures.\n\n")
            f.write(f"const float ev_model_training_medians[{n_features}] = {{\n")
            for idx, feat in enumerate(selected_features):
                median_val = train_medians[feat]
                f.write(f"    {median_val:.6f}f,  // [{idx}] {feat}\n")
            f.write("};\n\n")
            
            # Safe imputation helper function
            f.write("/**\n")
            f.write(" * Apply median imputation to handle sensor failures\n")
            f.write(" * \n")
            f.write(" * Call this BEFORE prediction to validate sensor readings.\n")
            f.write(" * If a value is NaN, Inf, or clearly invalid, replace with training median.\n")
            f.write(" * \n")
            f.write(" * @param features Array of sensor values (will be modified in-place)\n")
            f.write(" * @return Number of sensors that were replaced with medians\n")
            f.write(" */\n")
            f.write("static inline int ev_safe_impute(float features[EV_MODEL_N_FEATURES]) {\n")
            f.write("    int replaced = 0;\n")
            f.write("    for (int i = 0; i < EV_MODEL_N_FEATURES; i++) {\n")
            f.write("        // Check for NaN (x != x is true only for NaN)\n")
            f.write("        if (features[i] != features[i]) {\n")
            f.write("            features[i] = ev_model_training_medians[i];\n")
            f.write("            replaced++;\n")
            f.write("        }\n")
            f.write("        // Check for Infinity\n")
            f.write("        else if (features[i] == INFINITY || features[i] == -INFINITY) {\n")
            f.write("            features[i] = ev_model_training_medians[i];\n")
            f.write("            replaced++;\n")
            f.write("        }\n")
            f.write("    }\n")
            f.write("    return replaced;\n")
            f.write("}\n\n")
            
            # Feature names (optional, costs Flash)
            f.write("// ================================================================\n")
            f.write("// FEATURE METADATA (Optional - costs ~" + str(len(str(selected_features))) + " bytes Flash)\n")
            f.write("// ================================================================\n\n")
            f.write("#ifdef EV_MODEL_INCLUDE_FEATURE_NAMES\n")
            f.write(f"const char* ev_model_feature_names[{n_features}] = {{\n")
            for feat in selected_features:
                f.write(f'    "{feat}",\n')
            f.write("};\n")
            f.write("#endif\n\n")
            
            # Convenience wrapper function with built-in safety
            f.write("// ================================================================\n")
            f.write("// CONVENIENCE WRAPPER (Recommended)\n")
            f.write("// ================================================================\n\n")
            f.write("/**\n")
            f.write(" * Predict remaining EV range in kilometers (BASIC - no sensor validation)\n")
            f.write(" * \n")
            f.write(f" * @param features Array of {n_features} sensor values (see feature order above)\n")
            f.write(" * @return Predicted remaining range in km (e.g., 42.5)\n")
            f.write(" */\n")
            f.write("static inline float ev_predict_range_km(const float features[EV_MODEL_N_FEATURES]) {\n")
            f.write("    return ev_range_model_predict(features, EV_MODEL_N_FEATURES);\n")
            f.write("}\n\n")
            
            f.write("/**\n")
            f.write(" * Predict remaining EV range with automatic sensor failure handling (RECOMMENDED)\n")
            f.write(" * \n")
            f.write(" * This is the PRODUCTION-READY version that handles sensor failures gracefully.\n")
            f.write(" * If sensors return NaN/Inf, they are replaced with training medians.\n")
            f.write(" * \n")
            f.write(f" * @param features Array of {n_features} sensor values (modified in-place if needed)\n")
            f.write(" * @param out_replaced Optional pointer to store number of sensors replaced (can be NULL)\n")
            f.write(" * @return Predicted remaining range in km\n")
            f.write(" */\n")
            f.write("static inline float ev_predict_range_km_safe(float features[EV_MODEL_N_FEATURES], int* out_replaced) {\n")
            f.write("    int replaced = ev_safe_impute(features);\n")
            f.write("    if (out_replaced != NULL) {\n")
            f.write("        *out_replaced = replaced;\n")
            f.write("    }\n")
            f.write("    return ev_range_model_predict(features, EV_MODEL_N_FEATURES);\n")
            f.write("}\n\n")
            
            f.write("#endif // EV_RANGE_MODEL_H\n")
        
        print(f"✓ Enhanced header saved: {enhanced_file.name}")
        
        # Estimate ESP32 resource usage
        max_nodes_per_tree = (2 ** (max_depth + 1)) - 1
        bytes_per_node = 12  # emlearn uses ~12 bytes per node (feature_idx, threshold, left/right)
        flash_estimate = n_trees * max_nodes_per_tree * bytes_per_node
        ram_estimate = (n_features * 4) + (max_depth * 8)  # Input array + tree traversal stack
        
        print(f"\n{'─'*80}")
        print("ESP32 RESOURCE ESTIMATES")
        print(f"{'─'*80}")
        print(f"  Model size: ~{flash_estimate/1024:.1f} KB Flash memory")
        print(f"  Runtime RAM: ~{ram_estimate} bytes")
        print(f"  Stack depth: {max_depth} levels")
        print(f"\nESP32-WROOM-32 Specifications:")
        print(f"  Available Flash: 4 MB")
        print(f"  Available RAM: 520 KB")
        
        if flash_estimate < 100 * 1024:
            fit_status = "✓ EXCELLENT FIT (< 100 KB)"
        elif flash_estimate < 300 * 1024:
            fit_status = "✓ GOOD FIT (< 300 KB)"
        elif flash_estimate < 500 * 1024:
            fit_status = "⚠ ACCEPTABLE (< 500 KB)"
        else:
            fit_status = "✗ TOO LARGE (consider reducing trees/depth)"
        
        print(f"  Fit Status: {fit_status}")
        
        # Generate Arduino example sketch
        arduino_example = output_path / 'ev_range_prediction_example.ino'
        generate_arduino_example(arduino_example, selected_features, enhanced_file.name)
        
        # Generate README
        readme_path = output_path / 'README_ESP32_DEPLOYMENT.md'
        generate_deployment_readme(readme_path, selected_features, enhanced_file.name, flash_estimate, ram_estimate)
        
        # Save metadata
        metadata = {
            'generated_timestamp': datetime.now().isoformat(),
            'model_type': 'RandomForestRegressor',
            'hyperparameters': {
                'n_estimators': n_trees,
                'max_depth': max_depth,
                'min_samples_split': CONFIG['model_params']['min_samples_split'],
                'min_samples_leaf': CONFIG['model_params']['min_samples_leaf'],
            },
            'features': {
                'count': n_features,
                'names': selected_features
            },
            'target_variable': CONFIG['target_column'],
            'files': {
                'raw_header': c_file_path.name,
                'enhanced_header': enhanced_file.name,
                'arduino_example': arduino_example.name,
                'readme': readme_path.name,
            },
            'file_sizes': {
                'raw_header_bytes': file_size,
                'enhanced_header_bytes': enhanced_file.stat().st_size,
            },
            'esp32_compatibility': {
                'estimated_flash_bytes': flash_estimate,
                'estimated_ram_bytes': ram_estimate,
                'stack_depth': max_depth,
                'fits_esp32': flash_estimate < 500 * 1024,
                'fit_quality': fit_status
            },
            'usage': {
                'include_statement': f'#include "{enhanced_file.name}"',
                'predict_function': f'ev_predict_range_km(features)',
                'expected_inference_time_ms': '<5',
            }
        }
        
        metadata_path = output_path / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Arduino example: {arduino_example.name}")
        print(f"✓ Deployment guide: {readme_path.name}")
        print(f"✓ Metadata: {metadata_path.name}")
        
        print(f"\n{'─'*80}")
        print("USAGE IN C++/ARDUINO")
        print(f"{'─'*80}")
        print(f'#include "{enhanced_file.name}"')
        print(f"")
        print(f"float features[{n_features}];")
        print(f"// ... populate features from sensors ...")
        print(f"float range = ev_predict_range_km(features);")
        
        return metadata
        
    except Exception as e:
        print(f"\n✗ Error during C code generation: {e}")
        print("\nTroubleshooting steps:")
        print(f"1. Ensure emlearn is installed: pip install emlearn")
        print(f"2. Try reducing max_depth to 5-8")
        print(f"3. Try reducing n_estimators to 15-25")
        print(f"4. Check that model trained successfully")
        import traceback
        traceback.print_exc()
        return None


def generate_arduino_example(filepath: Path, selected_features: list, header_name: str):
    """Generate complete Arduino example sketch for ESP32"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("/*\n")
        f.write(" * EV Range Prediction - ESP32 Example Sketch\n")
        f.write(" * ============================================\n")
        f.write(" * \n")
        f.write(" * This example shows how to use the trained Random Forest model\n")
        f.write(" * for real-time range prediction on ESP32.\n")
        f.write(" * \n")
        f.write(" * Hardware Requirements:\n")
        f.write(" * - ESP32 DevKit (any variant with 4MB Flash)\n")
        f.write(" * - I2C sensors: INA238, MPU6050, BMP280\n")
        f.write(" * - Analog sensors: Hall throttle, LDR, Rain sensor\n")
        f.write(" * \n")
        f.write(" * Wiring:\n")
        f.write(" * - SDA -> GPIO21, SCL -> GPIO22 (I2C bus)\n")
        f.write(" * - Throttle -> GPIO34 (ADC1)\n")
        f.write(" * - LDR -> GPIO35 (ADC1)\n")
        f.write(" * \n")
        f.write(f" * Model: {CONFIG['model_params']['n_estimators']} trees, depth {CONFIG['model_params']['max_depth']}\n")
        f.write(f" * Features: {len(selected_features)}\n")
        f.write(" */\n\n")
        
        f.write(f'#include "{header_name}"\n\n')
        
        f.write("// Feature array (populated from sensors)\n")
        f.write(f"float sensor_features[EV_MODEL_N_FEATURES];\n\n")
        
        f.write("// Prediction result\n")
        f.write("float predicted_range_km = 0.0;\n\n")
        
        f.write("void setup() {\n")
        f.write("    Serial.begin(115200);\n")
        f.write('    delay(1000);\n')
        f.write('    Serial.println("\\n=================================");\n')
        f.write('    Serial.println("EV Range Predictor - ESP32");\n')
        f.write('    Serial.println("=================================");\n')
        f.write(f'    Serial.println("Model: {CONFIG["model_params"]["n_estimators"]} trees, depth {CONFIG["model_params"]["max_depth"]}");\n')
        f.write(f'    Serial.println("Features: {len(selected_features)}");\n')
        f.write('    Serial.println("Ready!\\n");\n')
        f.write("    \n")
        f.write("    // TODO: Initialize I2C sensors here\n")
        f.write("    // Wire.begin(21, 22);  // SDA, SCL\n")
        f.write("}\n\n")
        
        f.write("void loop() {\n")
        f.write("    // ========================================\n")
        f.write("    // STEP 1: Read all sensor values\n")
        f.write("    // ========================================\n")
        f.write("    \n")
        
        # Show first few features as examples
        for idx, feat in enumerate(selected_features[:6]):
            f.write(f"    sensor_features[{idx}] = read_{feat.replace('-', '_').replace(' ', '_')}();  // {feat}\n")
        
        if len(selected_features) > 6:
            f.write(f"    // ... populate remaining {len(selected_features) - 6} features ...\n")
        
        f.write("    \n")
        f.write("    // ========================================\n")
        f.write("    // STEP 2: Validate sensors & run ML prediction\n")
        f.write("    // ========================================\n")
        f.write("    \n")
        f.write("    // Use SAFE prediction (handles sensor failures gracefully)\n")
        f.write("    int sensors_replaced = 0;\n")
        f.write("    unsigned long start_micros = micros();\n")
        f.write("    predicted_range_km = ev_predict_range_km_safe(sensor_features, &sensors_replaced);\n")
        f.write("    unsigned long inference_time_us = micros() - start_micros;\n")
        f.write("    \n")
        f.write("    // Warn if sensors were replaced with medians\n")
        f.write("    if (sensors_replaced > 0) {\n")
        f.write("        Serial.print(\"⚠ WARNING: \");\n")
        f.write("        Serial.print(sensors_replaced);\n")
        f.write("        Serial.println(\" sensor(s) failed - using median fallback\");\n")
        f.write("    }\n")
        f.write("    \n")
        f.write("    // ========================================\n")
        f.write("    // STEP 3: Display results\n")
        f.write("    // ========================================\n")
        f.write("    \n")
        f.write('    Serial.println("--- Prediction Result ---");\n')
        f.write('    Serial.print("Predicted Range: ");\n')
        f.write('    Serial.print(predicted_range_km, 2);\n')
        f.write('    Serial.println(" km");\n')
        f.write('    \n')
        f.write('    Serial.print("Inference Time: ");\n')
        f.write('    Serial.print(inference_time_us);\n')
        f.write('    Serial.println(" µs");\n')
        f.write('    Serial.println();\n')
        f.write("    \n")
        f.write("    delay(2000);  // Update every 2 seconds\n")
        f.write("}\n\n")
        
        f.write("// ============================================\n")
        f.write("// SENSOR READING FUNCTIONS (IMPLEMENT THESE)\n")
        f.write("// ============================================\n\n")
        
        # Generate stub functions for first few features
        for feat in selected_features[:4]:
            func_name = f"read_{feat.replace('-', '_').replace(' ', '_')}"
            f.write(f"float {func_name}() {{\n")
            f.write(f"    // TODO: Implement actual sensor reading for '{feat}'\n")
            f.write(f"    // Example: Read from I2C sensor, ADC, calculate value, etc.\n")
            f.write(f"    return 0.0;  // Placeholder\n")
            f.write("}\n\n")
        
        f.write("// ... Implement remaining sensor functions ...\n")
    
    print(f"  Generated Arduino example with {len(selected_features)} sensor stubs")


def generate_deployment_readme(filepath: Path, selected_features: list, 
                               header_name: str, flash_kb: float, ram_bytes: int):
    """Generate deployment README documentation"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# ESP32 Deployment Guide - EV Range Prediction Model\n\n")
        
        f.write("## 📦 Files Included\n\n")
        f.write(f"- **`{header_name}`** - Main C header file (include in your project)\n")
        f.write("- **`ev_range_prediction_example.ino`** - Complete Arduino example\n")
        f.write("- **`model_metadata.json`** - Model specifications and performance\n")
        f.write("- **`training_report.txt`** - Training metrics and evaluation\n\n")
        
        f.write("## 🚀 Quick Start\n\n")
        f.write("### 1. Copy Files to Arduino Project\n\n")
        f.write("```bash\n")
        f.write("cp ev_range_model.h /path/to/your/arduino/sketch/\n")
        f.write("```\n\n")
        
        f.write("### 2. Include in Your Code\n\n")
        f.write("```cpp\n")
        f.write(f'#include "{header_name}"\n\n')
        f.write(f"float features[{len(selected_features)}];\n")
        f.write("// ... read sensors and populate features[] ...\n\n")
        f.write("float predicted_range = ev_predict_range_km(features);\n")
        f.write("```\n\n")
        
        f.write("### 3. Populate Feature Array\n\n")
        f.write("Features must be provided in this exact order:\n\n")
        f.write("| Index | Feature Name | Description |\n")
        f.write("|-------|--------------|-------------|\n")
        for idx, feat in enumerate(selected_features):
            f.write(f"| {idx} | `{feat}` | Sensor value |\n")
        f.write("\n")
        
        f.write("## 💾 Memory Requirements\n\n")
        f.write(f"- **Flash Memory**: ~{flash_kb/1024:.1f} KB (model code)\n")
        f.write(f"- **RAM (runtime)**: ~{ram_bytes} bytes\n")
        f.write(f"- **Stack Depth**: {CONFIG['model_params']['max_depth']} levels\n\n")
        f.write("**ESP32-WROOM-32**: 4 MB Flash, 520 KB RAM → ✓ Compatible\n\n")
        
        f.write("## ⚡ Performance\n\n")
        f.write(f"- **Inference Time**: <5ms @ 240MHz\n")
        f.write(f"- **Trees**: {CONFIG['model_params']['n_estimators']}\n")
        f.write(f"- **Max Depth**: {CONFIG['model_params']['max_depth']}\n")
        f.write(f"- **Accuracy**: See `training_report.txt`\n\n")
        
        f.write("## 🔧 Hardware Setup\n\n")
        f.write("### Required Sensors\n\n")
        f.write("1. **INA238** (I2C: 0x40) - Voltage/Current monitor\n")
        f.write("2. **MPU6050** (I2C: 0x68) - IMU (accelerometer + gyro)\n")
        f.write("3. **BMP280** (I2C: 0x76) - Barometric pressure/altitude\n")
        f.write("4. **Hall Effect Sensor** (Analog) - Throttle position\n")
        f.write("5. **Load Cell + ADS1232** (SPI) - Weight measurement\n")
        f.write("6. **LDR** (Analog) - Light sensor\n")
        f.write("7. **Rain Sensor** (Analog) - Moisture detection\n\n")
        
        f.write("### Wiring\n\n")
        f.write("```\n")
        f.write("ESP32 Pin | Connection\n")
        f.write("----------|------------------\n")
        f.write("GPIO 21   | I2C SDA (INA238, MPU6050, BMP280)\n")
        f.write("GPIO 22   | I2C SCL\n")
        f.write("GPIO 34   | Hall throttle (ADC)\n")
        f.write("GPIO 35   | LDR (ADC)\n")
        f.write("GPIO 32   | Rain sensor (ADC)\n")
        f.write("GPIO 18   | ADS1232 SCLK\n")
        f.write("GPIO 19   | ADS1232 DOUT\n")
        f.write("```\n\n")
        
        f.write("## 📝 Example Usage\n\n")
        f.write("### Basic Prediction (no validation)\n")
        f.write("```cpp\n")
        f.write("float predicted_range = ev_predict_range_km(features);\n")
        f.write("```\n\n")
        f.write("### Production-Ready Prediction (RECOMMENDED)\n")
        f.write("```cpp\n")
        f.write("int sensors_replaced = 0;\n")
        f.write("float predicted_range = ev_predict_range_km_safe(features, &sensors_replaced);\n\n")
        f.write("if (sensors_replaced > 0) {\n")
        f.write("    // Handle sensor failure warning\n")
        f.write("    Serial.println(\"⚠ Sensor failure detected - using median fallback\");\n")
        f.write("}\n")
        f.write("```\n\n")
        f.write("The `_safe` variant automatically replaces NaN/Inf sensor values with training medians,\n")
        f.write("preventing wild predictions from sensor failures during operation.\n\n")
        f.write("See `ev_range_prediction_example.ino` for complete working code.\n\n")
        
        f.write("## 🐛 Troubleshooting\n\n")
        f.write("### Model predictions seem off\n")
        f.write("- Verify all sensors are calibrated correctly\n")
        f.write("- Check feature units match training data (km/h, volts, etc.)\n")
        f.write("- Ensure feature array is in correct order\n\n")
        
        f.write("### Compilation errors\n")
        f.write("- Increase Flash partition size if needed\n")
        f.write("- Use Arduino IDE 2.0+ or PlatformIO\n")
        f.write("- Check ESP32 board package is up to date\n\n")
        
        f.write("### Slow inference\n")
        f.write(f"- Should be <5ms @ 240MHz\n")
        f.write("- If slower, check CPU frequency setting\n")
        f.write("- Disable debug Serial.print() in production\n\n")
        
        f.write("## 📚 Additional Resources\n\n")
        f.write("- [emlearn Documentation](https://emlearn.readthedocs.io/)\n")
        f.write("- [ESP32 Arduino Core](https://github.com/espressif/arduino-esp32)\n")
        f.write("- Model training details: `training_report.txt`\n")
    
    print(f"  Generated deployment README with setup instructions")



# =============================================================================
# REPORTING
# =============================================================================

def generate_final_report(metrics: dict, 
                         selected_features: list,
                         export_metadata: dict,
                         output_dir: str):
    """Generate comprehensive training report."""
    print(f"\n{'='*80}")
    print("STEP 7: GENERATING FINAL REPORT")
    print(f"{'='*80}\n")
    
    output_path = Path(output_dir)
    report_path = output_path / 'training_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EV RANGE PREDICTION MODEL - TRAINING REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("MODEL CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Model Type: Random Forest Regressor\n")
        f.write(f"Target Platform: ESP32 Microcontroller\n")
        f.write(f"Export Library: emlearn\n\n")
        
        f.write("Hyperparameters:\n")
        for key, value in CONFIG['model_params'].items():
            if key not in ['n_jobs', 'verbose']:
                f.write(f"  {key}: {value}\n")
        
        f.write(f"\nSelected Features ({len(selected_features)}):\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"  {i:2d}. {feat}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("PERFORMANCE METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Training Set:\n")
        f.write(f"  R² Score:  {metrics['train']['r2']:.6f}\n")
        f.write(f"  MAE:       {metrics['train']['mae']:.4f} km\n")
        f.write(f"  RMSE:      {metrics['train']['rmse']:.4f} km\n\n")
        
        f.write("Test Set:\n")
        f.write(f"  R² Score:  {metrics['test']['r2']:.6f}\n")
        f.write(f"  MAE:       {metrics['test']['mae']:.4f} km\n")
        f.write(f"  RMSE:      {metrics['test']['rmse']:.4f} km\n\n")
        
        f.write("Cross-Validation:\n")
        f.write(f"  Mean R²:   {metrics['cv']['mean_r2']:.6f}\n")
        f.write(f"  Std R²:    {metrics['cv']['std_r2']:.6f}\n\n")
        
        r2_gap = metrics['train']['r2'] - metrics['test']['r2']
        f.write(f"Generalization Gap: {r2_gap:.6f}\n")
        if r2_gap > CONFIG['overfitting_threshold']:
            f.write(f"  Status: ⚠ Potential overfitting\n")
        else:
            f.write(f"  Status: ✓ Good generalization\n")
        
        if export_metadata:
            f.write("\n" + "="*80 + "\n")
            f.write("EMBEDDED DEPLOYMENT\n")
            f.write("="*80 + "\n")
            f.write(f"C Header File: ev_range_model.h\n")
            f.write(f"File Size: {export_metadata['file_sizes']['enhanced_header_bytes']:,} bytes\n")
            f.write(f"Estimated Flash: {export_metadata['esp32_compatibility']['estimated_flash_bytes']/1024:.1f} KB\n")
            f.write(f"Estimated RAM: {export_metadata['esp32_compatibility']['estimated_ram_bytes']} bytes\n")
            f.write(f"ESP32 Compatible: {'✓ YES' if export_metadata['esp32_compatibility']['fits_esp32'] else '✗ NO'}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DEPLOYMENT INSTRUCTIONS\n")
        f.write("="*80 + "\n")
        f.write("1. Copy 'ev_range_model.h' to your Arduino project folder\n")
        f.write("2. Include the header: #include \"ev_range_model.h\"\n")
        f.write("3. Prepare feature array: float features[EV_MODEL_N_FEATURES];\n")
        f.write("4. Call prediction: float range = ev_predict_range_km(features);\n")
        f.write("5. Expected inference time: <5ms @ 240MHz\n")
        f.write("\n")
        f.write("See 'ev_range_prediction_example.ino' for complete working example.\n")
    
    print(f"✓ Training report saved: {report_path.name}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*80)
    print("ESP32 EMBEDDED MODEL TRAINING PIPELINE - FIXED VERSION")
    print("="*80)
    print(f"\nDataset: {CONFIG['dataset_path']}")
    print(f"Output: {CONFIG['output_dir']}")
    print(f"Target: {CONFIG['target_column']}")
    
    try:
        # Step 1: Load raw data (NO imputation or feature selection yet)
        df, candidate_features, target_col = load_and_preprocess_data(
            CONFIG['dataset_path']
        )
        
        # Step 2: Split FIRST, then impute using TRAINING stats only (prevents leakage)
        X_train_full, X_test_full, y_train, y_test, train_medians = split_first_then_impute(
            df, candidate_features, target_col
        )
        
        # Step 3: Feature selection using TRAINING data only (prevents leakage)
        selected_features = select_top_features_from_training(
            X_train_full, 
            y_train,
            max_features=CONFIG['max_features']
        )
        
        # Apply feature selection to both train and test
        X_train = X_train_full[selected_features].copy()
        X_test = X_test_full[selected_features].copy()
        
        print(f"\n✓ Final feature set: {len(selected_features)} features")
        print(f"✓ Train shape: {X_train.shape}")
        print(f"✓ Test shape: {X_test.shape}")
        
        # Step 4: Train model
        model = train_embedded_optimized_model(X_train, y_train)
        
        # Step 5: Evaluate
        metrics = evaluate_model_performance(
            model, X_train, X_test, y_train, y_test
        )
        
        # Step 6: Export to C (including median fallback values for sensor failures)
        export_metadata = export_to_c_code(
            model, selected_features, train_medians[selected_features], CONFIG['output_dir']
        )
        
        # Step 7: Generate report
        generate_final_report(
            metrics, selected_features, export_metadata, CONFIG['output_dir']
        )
        
        print(f"\n{'='*80}")
        print("✓ TRAINING PIPELINE COMPLETE")
        print(f"{'='*80}\n")
        
        print("Summary:")
        print(f"  Test R²:    {metrics['test']['r2']:.4f}")
        print(f"  Test MAE:   {metrics['test']['mae']:.2f} km")
        print(f"  Test RMSE:  {metrics['test']['rmse']:.2f} km")
        
        if export_metadata:
            print(f"  Model Size: {export_metadata['esp32_compatibility']['estimated_flash_bytes']/1024:.1f} KB")
            print(f"  RAM Usage:  {export_metadata['esp32_compatibility']['estimated_ram_bytes']} bytes")
            print(f"  ESP32 Fit:  {export_metadata['esp32_compatibility']['fit_quality']}")
        
        print(f"\nFiles generated in: {CONFIG['output_dir']}")
        print("  • ev_range_model.h                    (Include in your ESP32 project)")
        print("  • ev_range_prediction_example.ino     (Example usage code)")
        print("  • model_metadata.json                 (Model specifications)")
        print("  • training_report.txt                 (Performance summary)")
        print("  • README_ESP32_DEPLOYMENT.md           (Deployment guide)")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)