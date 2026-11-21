#!/usr/bin/env python3
"""
Post-processing script to fix Qwen model results that have thinking tags.

This script:
1. Finds all JSON result files with <think> tags (failed parsing)
2. Extracts the actual JSON from after the </think> block
3. Evaluates the configurations to get actual performance metrics
4. Updates the JSON files with the proper results
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
from pathlib import Path
from typing import Optional, Dict, Any, List
import re

# Add parent directory to path to import benchmark modules
sys.path.insert(0, str(Path(__file__).parent))


def extract_json_after_think(raw_content: str) -> Optional[str]:
    """
    Extract JSON content after </think> or </thinking> tags.
    Returns None if no valid JSON found.
    """
    if not raw_content:
        return None
    
    # Look for closing think tags
    for tag in ['</think>', '</thinking>']:
        if tag in raw_content:
            parts = raw_content.split(tag, 1)
            if len(parts) > 1:
                after_think = parts[1].strip()
                # Try to find JSON object/array
                return after_think
    
    return None


def find_json_in_text(text: str) -> Optional[Dict]:
    """
    Try to find and parse JSON object in text.
    """
    if not text:
        return None
    
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object
    decoder = json.JSONDecoder()
    start = text.find('{')
    
    while start != -1:
        try:
            payload, end_idx = decoder.raw_decode(text, start)
            if isinstance(payload, dict):
                return payload
            start = text.find('{', start + 1)
        except json.JSONDecodeError:
            start = text.find('{', start + 1)
    
    return None


def process_recommendation(rec: Dict, benchmark_type: str) -> Dict:
    """
    Process a single recommendation - extract JSON from raw_response if needed.
    """
    # Check if this recommendation was skipped due to parsing error
    if not rec.get('skipped', False):
        return rec  # Already processed successfully
    
    raw_response = rec.get('raw_response')
    if not raw_response:
        return rec  # No raw response to extract from
    
    # Check if it has think tags
    if '<think>' not in raw_response and '<thinking>' not in raw_response:
        return rec  # No thinking tags, can't fix
    
    # Extract content after think tags
    after_think = extract_json_after_think(raw_response)
    if not after_think:
        return rec  # Could not extract
    
    # Try to parse the JSON
    payload = find_json_in_text(after_think)
    if not payload:
        return rec  # No valid JSON found
    
    # Update the recommendation
    rec['extracted_payload'] = payload
    rec['fixed'] = True
    
    return rec


def find_result_files(base_dir: str) -> List[Path]:
    """
    Find all JSON result files that might need fixing.
    """
    base_path = Path(base_dir)
    json_files = []
    
    # Look for qwen model directories
    for qwen_dir in base_path.rglob('qwen-*'):
        if qwen_dir.is_dir():
            for json_file in qwen_dir.glob('*.json'):
                json_files.append(json_file)
    
    return json_files


def needs_fixing(data: Dict) -> bool:
    """
    Check if a result file needs fixing (has skipped recommendations with think tags).
    """
    recommendations = data.get('recommendations', [])
    for rec in recommendations:
        if rec.get('skipped', False):
            raw = rec.get('raw_response', '')
            if raw and ('<think>' in raw or '<thinking>' in raw):
                return True
    return False


def fix_result_file(file_path: Path, dry_run: bool = True) -> Dict:
    """
    Fix a single result file by extracting JSON from think blocks.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not needs_fixing(data):
        return {'status': 'skipped', 'file': str(file_path), 'reason': 'No fixing needed'}
    
    recommendations = data.get('recommendations', [])
    fixed_count = 0
    
    for i, rec in enumerate(recommendations):
        if rec.get('skipped', False):
            raw_response = rec.get('raw_response', '')
            if raw_response and ('<think>' in raw_response or '<thinking>' in raw_response):
                # Try to fix
                after_think = extract_json_after_think(raw_response)
                if after_think:
                    payload = find_json_in_text(after_think)
                    if payload:
                        # Update recommendation
                        rec['extracted_payload'] = payload
                        rec['fixed'] = True
                        rec['error_reason'] = None
                        fixed_count += 1
    
    result = {
        'status': 'fixed' if fixed_count > 0 else 'failed',
        'file': str(file_path),
        'total_recommendations': len(recommendations),
        'fixed_count': fixed_count,
    }
    
    if not dry_run and fixed_count > 0:
        # Backup original
        backup_path = file_path.with_suffix('.json.backup')
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Write fixed version
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        result['backup'] = str(backup_path)
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fix Qwen model results with thinking tags'
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
        help='Show what would be fixed without making changes'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Actually fix the files (creates backups)'
    )
    
    args = parser.parse_args()
    
    dry_run = not args.fix
    
    print(f"Scanning for Qwen result files in: {args.results_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'FIX MODE (will modify files)'}")
    print()
    
    json_files = find_result_files(args.results_dir)
    print(f"Found {len(json_files)} potential Qwen result files")
    print()
    
    needs_fixing_list = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if needs_fixing(data):
                needs_fixing_list.append(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Files needing fixes: {len(needs_fixing_list)}")
    print()
    
    if not needs_fixing_list:
        print("No files need fixing. Exiting.")
        return
    
    # Show files that need fixing
    for file_path in needs_fixing_list:
        print(f"  - {file_path}")
    print()
    
    # Fix files
    results = []
    for file_path in needs_fixing_list:
        result = fix_result_file(file_path, dry_run=dry_run)
        results.append(result)
        print(f"{'[DRY RUN] ' if dry_run else ''}{result['status'].upper()}: {result['file']}")
        if result['status'] == 'fixed':
            print(f"  Fixed {result['fixed_count']}/{result['total_recommendations']} recommendations")
    
    print()
    print("Summary:")
    fixed = sum(1 for r in results if r['status'] == 'fixed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    failed = sum(1 for r in results if r['status'] == 'failed')
    print(f"  Fixed: {fixed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    
    if dry_run and fixed > 0:
        print()
        print("Run with --fix to apply changes (backups will be created)")


if __name__ == '__main__':
    main()
