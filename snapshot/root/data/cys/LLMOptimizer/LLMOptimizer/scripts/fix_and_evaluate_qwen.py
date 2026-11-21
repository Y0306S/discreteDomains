#!/usr/bin/env python3
"""
Post-processing script to fix Qwen model results and evaluate the configurations.

This script:
1. Finds all JSON result files with <think> tags (failed parsing)
2. Extracts the actual configurations from after the </think> block
3. Evaluates each configuration on the actual benchmark to get real performance metrics
4. Updates the JSON files with the proper results (loss, accuracy, cost)
"""

import json
import os

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPERIMENT_RUNS_DIR = ROOT / "experiment_runs"

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def extract_configs_from_think_block(raw_content: str) -> List[Dict]:
    """
    Extract configuration recommendations from content with <think> tags.
    Returns list of configs or empty list if extraction fails.
    """
    if not raw_content or '<think>' not in raw_content:
        return []
    
    # Extract content after think tags
    for tag in ['</think>', '</thinking>']:
        if tag in raw_content:
            parts = raw_content.split(tag, 1)
            if len(parts) > 1:
                after_think = parts[1].strip()
                break
    else:
        return []
    
    # Try to parse JSON
    try:
        data = json.loads(after_think)
        if 'recommendations' in data:
            return data['recommendations']
        return [data]
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object
    decoder = json.JSONDecoder()
    start = after_think.find('{')
    
    while start != -1:
        try:
            payload, end_idx = decoder.raw_decode(after_think, start)
            if isinstance(payload, dict):
                if 'recommendations' in payload:
                    return payload['recommendations']
                return [payload]
            start = after_think.find('{', start + 1)
        except json.JSONDecodeError:
            start = after_think.find('{', start + 1)
    
    return []


def load_openml_dataset(openml_id: int):
    """Load dataset from OpenML cache or fetch it."""
    cache_dir = Path(__file__).parent / 'openml_cache'
    cache_file = cache_dir / f'dataset_{openml_id}.pkl'
    
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Try to fetch from openml
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(data_id=openml_id, as_frame=True, parser='auto')
        
        X = data.data
        y = data.target
        
        # Cache it
        cache_dir.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((X, y), f)
        
        return X, y
    except Exception as e:
        print(f"    Error loading dataset {openml_id}: {e}")
        return None, None


def preprocess_data(X, y):
    """Preprocess data for ML models."""
    # Handle target encoding
    if y.dtype == 'object' or y.dtype.name == 'category':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Build preprocessor
    transformers = []
    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
    
    preprocessor = ColumnTransformer(transformers, remainder='passthrough')
    
    return X_train, X_test, y_train, y_test, preprocessor


def evaluate_lr_config(config: Dict, X_train, X_test, y_train, y_test, preprocessor) -> Dict:
    """Evaluate a Logistic Regression configuration."""
    # Extract parameters from config
    warm_start_config = config.get('warm_start_config', [])
    params = {item['name']: item['value'] for item in warm_start_config if 'name' in item and 'value' in item}
    
    alpha = params.get('alpha', 0.0001)
    eta0 = params.get('eta0', 0.01)
    
    try:
        # Build pipeline
        clf = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=float(alpha),
            learning_rate='constant',
            eta0=float(eta0),
            max_iter=1000,
            random_state=42,
            tol=1e-3
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        # Calculate log loss (handle edge cases)
        try:
            loss = log_loss(y_test, y_proba)
        except:
            loss = 1.0 - balanced_acc  # Approximate
        
        return {
            'loss': float(loss),
            'balanced_accuracy': float(balanced_acc),
            'accuracy': float(acc),
            'config': params,
            'error': None
        }
    except Exception as e:
        return {
            'loss': 1.0,
            'balanced_accuracy': 0.0,
            'accuracy': 0.0,
            'config': params,
            'error': str(e)
        }


def evaluate_rf_config(config: Dict, X_train, X_test, y_train, y_test, preprocessor) -> Dict:
    """Evaluate a Random Forest configuration."""
    warm_start_config = config.get('warm_start_config', [])
    params = {item['name']: item['value'] for item in warm_start_config if 'name' in item and 'value' in item}
    
    # Default RF parameters
    n_estimators = int(params.get('n_estimators', 100))
    max_depth = int(params.get('max_depth', 10)) if 'max_depth' in params else None
    min_samples_split = int(params.get('min_samples_split', 2))
    min_samples_leaf = int(params.get('min_samples_leaf', 1))
    max_features = params.get('max_features', 'sqrt')
    
    try:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        try:
            loss = log_loss(y_test, y_proba)
        except:
            loss = 1.0 - balanced_acc
        
        return {
            'loss': float(loss),
            'balanced_accuracy': float(balanced_acc),
            'accuracy': float(acc),
            'config': params,
            'error': None
        }
    except Exception as e:
        return {
            'loss': 1.0,
            'balanced_accuracy': 0.0,
            'accuracy': 0.0,
            'config': params,
            'error': str(e)
        }


def evaluate_nn_config(config: Dict, X_train, X_test, y_train, y_test, preprocessor) -> Dict:
    """Evaluate a Neural Network configuration."""
    warm_start_config = config.get('warm_start_config', [])
    params = {item['name']: item['value'] for item in warm_start_config if 'name' in item and 'value' in item}
    
    # Default NN parameters
    hidden_layer_sizes = params.get('hidden_layer_sizes', [50])
    if isinstance(hidden_layer_sizes, list):
        hidden_layer_sizes = tuple(hidden_layer_sizes)
    else:
        hidden_layer_sizes = (int(hidden_layer_sizes),)
    
    alpha = float(params.get('alpha', 0.0001))
    learning_rate_init = float(params.get('learning_rate_init', 0.001))
    batch_size = int(params.get('batch_size', 200))
    
    try:
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        try:
            loss = log_loss(y_test, y_proba)
        except:
            loss = 1.0 - balanced_acc
        
        return {
            'loss': float(loss),
            'balanced_accuracy': float(balanced_acc),
            'accuracy': float(acc),
            'config': params,
            'error': None
        }
    except Exception as e:
        return {
            'loss': 1.0,
            'balanced_accuracy': 0.0,
            'accuracy': 0.0,
            'config': params,
            'error': str(e)
        }


def evaluate_xgboost_config(config: Dict, X_train, X_test, y_train, y_test, preprocessor) -> Dict:
    """Evaluate an XGBoost configuration."""
    try:
        import xgboost as xgb
        
        warm_start_config = config.get('warm_start_config', [])
        params = {item['name']: item['value'] for item in warm_start_config if 'name' in item and 'value' in item}
        
        # Default XGBoost parameters
        xgb_params = {
            'n_estimators': int(params.get('n_estimators', 100)),
            'max_depth': int(params.get('max_depth', 6)),
            'learning_rate': float(params.get('learning_rate', 0.3)),
            'subsample': float(params.get('subsample', 1.0)),
            'colsample_bytree': float(params.get('colsample_bytree', 1.0)),
            'min_child_weight': int(params.get('min_child_weight', 1)),
            'gamma': float(params.get('gamma', 0)),
            'reg_alpha': float(params.get('reg_alpha', 0)),
            'reg_lambda': float(params.get('reg_lambda', 1)),
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        clf = xgb.XGBClassifier(**xgb_params)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)
        
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        
        try:
            loss = log_loss(y_test, y_proba)
        except:
            loss = 1.0 - balanced_acc
        
        return {
            'loss': float(loss),
            'balanced_accuracy': float(balanced_acc),
            'accuracy': float(acc),
            'config': params,
            'error': None
        }
    except ImportError:
        return {
            'loss': 1.0,
            'balanced_accuracy': 0.0,
            'accuracy': 0.0,
            'config': {},
            'error': 'XGBoost not installed'
        }
    except Exception as e:
        return {
            'loss': 1.0,
            'balanced_accuracy': 0.0,
            'accuracy': 0.0,
            'config': params if 'params' in dir() else {},
            'error': str(e)
        }


def parse_result_path(file_path: Path) -> Tuple[str, int, str]:
    """Parse benchmark type, dataset ID, and model from result file path."""
    parts = file_path.parts
    
    # Find benchmark type (LR, RF, NN, XGBOOST)
    benchmark = None
    for part in parts:
        if part.upper() in ['LR', 'RF', 'NN', 'XGBOOST', 'XGBOOST_ABLATION']:
            benchmark = part.upper()
            break
    
    # Find dataset ID (numeric folder)
    dataset_id = None
    for part in parts:
        if part.isdigit():
            dataset_id = int(part)
            break
    
    # Find model name
    model = None
    for part in parts:
        if 'qwen' in part.lower():
            model = part
            break
    
    return benchmark, dataset_id, model


def process_file(file_path: Path, dry_run: bool = True) -> Dict:
    """Process a single result file - extract and evaluate configs."""
    print(f"\nProcessing: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Parse path to get benchmark info
    benchmark, dataset_id, model = parse_result_path(file_path)
    
    if not benchmark or not dataset_id:
        return {'status': 'error', 'file': str(file_path), 'reason': 'Could not parse benchmark/dataset'}
    
    print(f"  Benchmark: {benchmark}, Dataset: {dataset_id}, Model: {model}")
    
    # Load dataset
    X, y = load_openml_dataset(dataset_id)
    if X is None:
        return {'status': 'error', 'file': str(file_path), 'reason': f'Could not load dataset {dataset_id}'}
    
    print(f"  Dataset loaded: {X.shape}")
    
    # Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(X, y)
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Process recommendations
    recommendations = data.get('recommendations', [])
    evaluated_count = 0
    
    for i, rec in enumerate(recommendations):
        if not rec.get('skipped', False):
            print(f"  Config {i+1}: Already evaluated (skipped={rec.get('skipped')})")
            continue
        
        raw_response = rec.get('raw_response', '')
        if not raw_response or '<think>' not in raw_response:
            print(f"  Config {i+1}: No think block to extract from")
            continue
        
        # Extract configs from think block
        configs = extract_configs_from_think_block(raw_response)
        if not configs:
            print(f"  Config {i+1}: Could not extract configs")
            continue
        
        # For batch recommendations, we need to match which config this is
        # The recommendation structure has 3 calls, each with its own config
        # We take the i-th config from the extracted recommendations
        if i < len(configs):
            config = configs[i]
        else:
            config = configs[0] if configs else None
        
        if not config:
            print(f"  Config {i+1}: No config available")
            continue
        
        print(f"  Config {i+1}: Evaluating...")
        
        # Evaluate based on benchmark type
        if benchmark == 'LR':
            result = evaluate_lr_config(config, X_train, X_test, y_train, y_test, preprocessor)
        elif benchmark == 'RF':
            result = evaluate_rf_config(config, X_train, X_test, y_train, y_test, preprocessor)
        elif benchmark == 'NN':
            result = evaluate_nn_config(config, X_train, X_test, y_train, y_test, preprocessor)
        elif benchmark in ['XGBOOST', 'XGBOOST_ABLATION']:
            result = evaluate_xgboost_config(config, X_train, X_test, y_train, y_test, preprocessor)
        else:
            print(f"  Config {i+1}: Unknown benchmark {benchmark}")
            continue
        
        if result['error']:
            print(f"  Config {i+1}: Evaluation error - {result['error']}")
        else:
            print(f"  Config {i+1}: loss={result['loss']:.4f}, balanced_acc={result['balanced_accuracy']:.4f}, acc={result['accuracy']:.4f}")
        
        # Update recommendation
        rec['reasoning_trace'] = config.get('reasoning_trace', '')
        rec['recommended_config'] = result['config']
        rec['evaluated_config'] = result['config']
        rec['loss'] = result['loss']
        rec['accuracy'] = result['accuracy']
        rec['balanced_accuracy'] = result['balanced_accuracy']
        rec['cost'] = None  # Could calculate if pricing available
        rec['skipped'] = False
        rec['error_reason'] = result['error']
        rec['evaluated'] = True
        
        evaluated_count += 1
    
    result_info = {
        'status': 'evaluated',
        'file': str(file_path),
        'benchmark': benchmark,
        'dataset_id': dataset_id,
        'evaluated_count': evaluated_count
    }
    
    if not dry_run and evaluated_count > 0:
        # Backup original
        backup_path = file_path.with_suffix('.json.backup')
        if not backup_path.exists():
            with open(backup_path, 'w') as f:
                json.dump(data, f, indent=2)
            result_info['backup'] = str(backup_path)
        
        # Write updated version
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  Saved updated file")
    
    return result_info


def find_qwen_result_files(base_dir: str) -> List[Path]:
    """Find all Qwen result JSON files."""
    base_path = Path(base_dir)
    json_files = []
    
    for qwen_dir in base_path.rglob('qwen-*'):
        if qwen_dir.is_dir():
            for json_file in qwen_dir.glob('*.json'):
                # Check if it needs fixing
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    recommendations = data.get('recommendations', [])
                    for rec in recommendations:
                        if rec.get('skipped', False):
                            raw = rec.get('raw_response', '')
                            if raw and '<think>' in raw:
                                json_files.append(json_file)
                                break
                except:
                    pass
    
    return json_files


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fix and evaluate Qwen model results with thinking tags'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results',
        help='Base directory containing results (default: results)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be evaluated without making changes'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Actually evaluate and update the files (creates backups)'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Process a specific file only'
    )
    
    args = parser.parse_args()
    
    dry_run = not args.fix
    
    print(f"Scanning for Qwen result files in: {args.results_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'FIX MODE (will modify files)'}")
    print()
    
    if args.file:
        json_files = [Path(args.file)]
    else:
        json_files = find_qwen_result_files(args.results_dir)
    
    print(f"Found {len(json_files)} files needing evaluation")
    print()
    
    if not json_files:
        print("No files need evaluation. Exiting.")
        return
    
    # Process files
    results = []
    for file_path in json_files:
        try:
            result = process_file(file_path, dry_run=dry_run)
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {file_path}: {e}")
            import traceback
            traceback.print_exc()
            results.append({'status': 'error', 'file': str(file_path), 'reason': str(e)})
    
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    evaluated = sum(1 for r in results if r['status'] == 'evaluated')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    for r in results:
        status = r['status'].upper()
        print(f"  [{status}] {r['file']}")
        if r['status'] == 'evaluated':
            print(f"           Evaluated {r.get('evaluated_count', 0)} configs")
        elif r['status'] == 'error':
            print(f"           Error: {r.get('reason', 'Unknown')}")
    
    print()
    print(f"Total: {len(results)} files")
    print(f"  Evaluated: {evaluated}")
    print(f"  Errors: {errors}")
    
    if dry_run and evaluated > 0:
        print()
        print("Run with --fix to apply changes (backups will be created)")


if __name__ == '__main__':
    main()
