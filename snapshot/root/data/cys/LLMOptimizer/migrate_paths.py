#!/usr/bin/env python3
"""
migrate_paths.py
----------------
Replace hardcoded absolute paths in the LLMOptimizer codebase when migrating
to a new server.

Usage:
    python migrate_paths.py --old-project-root /jizhicfs/pkuhetu/bcxu \\
                            --new-project-root /new/path/to/project \\
                            --old-conda-root /jizhicfs/johnnyslin/anaconda3 \\
                            --new-conda-root /new/path/to/anaconda3 \\
                            --dry-run

    # Then apply:
    python migrate_paths.py --old-project-root /jizhicfs/pkuhetu/bcxu \\
                            --new-project-root /new/path/to/project \\
                            --old-conda-root /jizhicfs/johnnyslin/anaconda3 \\
                            --new-conda-root /new/path/to/anaconda3
"""

import argparse
import os
import re
from pathlib import Path

# File extensions to process
EXTENSIONS = {'.py', '.sh', '.yaml', '.yml', '.json', '.txt', '.md'}

# Files/directories to skip
SKIP_PATTERNS = [
    '.git',
    '__pycache__',
    '.pyc',
    'migration_exports',
]


def should_process(path: Path) -> bool:
    """Check if a file should be processed."""
    path_str = str(path)
    for skip in SKIP_PATTERNS:
        if skip in path_str:
            return False
    return path.suffix in EXTENSIONS


def replace_in_file(filepath: Path, replacements: list, dry_run: bool = False) -> int:
    """Apply replacements to a file. Returns number of replacements made."""
    try:
        content = filepath.read_text(encoding='utf-8')
    except (UnicodeDecodeError, PermissionError):
        return 0
    
    original = content
    count = 0
    
    for old, new in replacements:
        if old in content:
            occurrences = content.count(old)
            content = content.replace(old, new)
            count += occurrences
    
    if count > 0 and not dry_run:
        filepath.write_text(content, encoding='utf-8')
    
    return count


def main():
    parser = argparse.ArgumentParser(description='Migrate hardcoded paths in LLMOptimizer')
    parser.add_argument('--old-project-root', default='/jizhicfs/pkuhetu/bcxu',
                        help='Old project root path')
    parser.add_argument('--new-project-root', required=True,
                        help='New project root path')
    parser.add_argument('--old-conda-root', default='/jizhicfs/johnnyslin/anaconda3',
                        help='Old conda installation path')
    parser.add_argument('--new-conda-root', required=True,
                        help='New conda installation path')
    parser.add_argument('--project-dir', default='/jizhicfs/pkuhetu/bcxu/LLMOptimizer',
                        help='Path to LLMOptimizer directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without applying')
    
    args = parser.parse_args()
    
    replacements = [
        (args.old_project_root, args.new_project_root),
        (args.old_conda_root, args.new_conda_root),
    ]
    
    project_dir = Path(args.project_dir)
    if not project_dir.exists():
        print(f"ERROR: Project directory not found: {project_dir}")
        return 1
    
    total_files = 0
    total_replacements = 0
    affected_files = []
    
    for filepath in project_dir.rglob('*'):
        if not filepath.is_file():
            continue
        if not should_process(filepath):
            continue
        
        count = replace_in_file(filepath, replacements, dry_run=args.dry_run)
        if count > 0:
            total_files += 1
            total_replacements += count
            affected_files.append((filepath.relative_to(project_dir), count))
    
    mode = "DRY RUN (no changes made)" if args.dry_run else "APPLIED"
    print(f"\n{'='*60}")
    print(f"Path Migration: {mode}")
    print(f"{'='*60}")
    print(f"Project dir: {project_dir}")
    print(f"Replacements:")
    print(f"  {args.old_project_root} → {args.new_project_root}")
    print(f"  {args.old_conda_root} → {args.new_conda_root}")
    print(f"\nFiles affected: {total_files}")
    print(f"Total replacements: {total_replacements}")
    
    if affected_files:
        print(f"\nTop affected files:")
        for rel_path, count in sorted(affected_files, key=lambda x: -x[1])[:20]:
            print(f"  {count:4d}  {rel_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
