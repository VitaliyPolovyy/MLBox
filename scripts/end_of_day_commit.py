#!/usr/bin/env python3
"""
End-of-Day Git Commit Script
Automatically categorizes and commits all changes made during the day
with a structured, categorized commit message.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import re

# Category definitions - maps file patterns to categories
CATEGORIES = {
    "🔧 LabelGuard Service": [
        "mlbox/services/LabelGuard/",
    ],
    "🥜 Peanuts Service": [
        "mlbox/services/peanuts/",
    ],
    "📦 Dependencies": [
        "poetry.lock",
        "pyproject.toml",
        "requirements.txt",
    ],
    "🚀 Deployment & Infrastructure": [
        "deploy",
        "deployments/",
        "ray_serv.yaml",
        "Dockerfile",
    ],
    "🛠️ Utilities & Shared Libraries": [
        "mlbox/utils/",
    ],
    "📊 Training Scripts": [
        "scripts/train/",
        "notebooks/",
    ],
    "📝 Documentation": [
        ".md",
        "docs/",
        "README",
    ],
    "🗄️ Data & Schemas": [
        "datatypes.py",
        "json-schemas/",
        "etalon.sql",
    ],
    "🗑️ Cleanup": [
        # Will be detected as deleted files
    ],
    "⚙️ Configuration": [
        "settings.py",
        ".env",
        "config",
    ],
    "📁 Project Structure": [
        # Other files
    ],
}

# Priority order for categorization (first match wins)
CATEGORY_PRIORITY = [
    "🔧 LabelGuard Service",
    "🥜 Peanuts Service",
    "📦 Dependencies",
    "🚀 Deployment & Infrastructure",
    "🛠️ Utilities & Shared Libraries",
    "📊 Training Scripts",
    "📝 Documentation",
    "🗄️ Data & Schemas",
    "⚙️ Configuration",
    "🗑️ Cleanup",
    "📁 Project Structure",
]


def get_git_status() -> List[str]:
    """Get list of modified, added, and deleted files from git."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True
    )
    return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]


def categorize_file(file_path: str, status: str) -> str:
    """Categorize a file based on its path and status."""
    # Check for deleted files first
    if status.startswith("D"):
        return "🗑️ Cleanup"
    
    # Check each category in priority order
    for category in CATEGORY_PRIORITY:
        patterns = CATEGORIES.get(category, [])
        for pattern in patterns:
            if pattern in file_path:
                return category
    
    # Default category
    return "📁 Project Structure"


def analyze_changes() -> Dict[str, List[Tuple[str, str]]]:
    """Analyze git changes and group them by category."""
    status_lines = get_git_status()
    categorized = defaultdict(list)
    
    for line in status_lines:
        # Parse git status line: "XY filename" or "?? filename"
        if len(line) >= 3:
            status = line[:2].strip()
            file_path = line[3:].strip()
            
            category = categorize_file(file_path, status)
            
            # Determine action symbol
            if status.startswith("D"):
                action = "🗑️ Deleted"
            elif status.startswith("A") or status == "??":
                action = "➕ Added"
            elif status.startswith("M"):
                action = "📝 Modified"
            elif status.startswith("R"):
                action = "🔄 Renamed"
            else:
                action = "📝 Changed"
            
            categorized[category].append((file_path, action))
    
    return dict(categorized)


def get_change_stats() -> Tuple[int, int]:
    """Get total lines added and deleted from git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--shortstat"],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output like: "37 files changed, 2742 insertions(+), 1776 deletions(-)"
        match = re.search(r'(\d+)\s+insertions?', result.stdout)
        insertions = int(match.group(1)) if match else 0
        match = re.search(r'(\d+)\s+deletions?', result.stdout)
        deletions = int(match.group(1)) if match else 0
        return insertions, deletions
    except:
        return 0, 0


def generate_commit_message(categorized_changes: Dict[str, List[Tuple[str, str]]]) -> str:
    """Generate a formatted commit message with categories."""
    today = datetime.now().strftime("%Y-%m-%d")
    insertions, deletions = get_change_stats()
    
    lines = [
        f"End of day commit - {today}",
        "",
        "## Summary",
        f"- Total changes: {sum(len(files) for files in categorized_changes.values())} files",
        f"- Lines added: {insertions:+d}",
        f"- Lines deleted: {deletions:+d}",
        "",
        "## Changes by Category",
        "",
    ]
    
    # Add categories in priority order
    for category in CATEGORY_PRIORITY:
        if category in categorized_changes:
            files = categorized_changes[category]
            lines.append(f"### {category}")
            
            # Group by action
            by_action = defaultdict(list)
            for file_path, action in files:
                by_action[action].append(file_path)
            
            for action in ["➕ Added", "📝 Modified", "🔄 Renamed", "🗑️ Deleted", "📝 Changed"]:
                if action in by_action:
                    files_list = by_action[action]
                    lines.append(f"  {action}:")
                    for file_path in sorted(files_list):
                        # Truncate very long paths
                        display_path = file_path if len(file_path) < 80 else file_path[:77] + "..."
                        lines.append(f"    - {display_path}")
            
            lines.append("")
    
    return "\n".join(lines)


def commit_and_push(message: str, dry_run: bool = False) -> bool:
    """Stage all changes, commit, and push to remote."""
    try:
        # Stage all changes
        print("📦 Staging all changes...")
        subprocess.run(["git", "add", "-A"], check=True)
        
        # Check if there are any changes to commit
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True
        )
        if result.returncode == 0:
            print("ℹ️  No changes to commit.")
            return False
        
        if dry_run:
            print("\n" + "="*80)
            print("DRY RUN - Commit message that would be used:")
            print("="*80)
            print(message)
            print("="*80)
            return True
        
        # Commit
        print("💾 Committing changes...")
        subprocess.run(
            ["git", "commit", "-m", message],
            check=True
        )
        
        # Push
        print("🚀 Pushing to remote...")
        subprocess.run(["git", "push"], check=True)
        
        print("✅ Successfully committed and pushed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="End-of-day git commit with categorized changes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be committed without actually committing"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Commit but don't push to remote"
    )
    
    args = parser.parse_args()
    
    print("🔍 Analyzing today's changes...")
    categorized_changes = analyze_changes()
    
    if not categorized_changes:
        print("ℹ️  No changes detected.")
        return
    
    print(f"📊 Found changes in {len(categorized_changes)} categories")
    for category, files in categorized_changes.items():
        print(f"  {category}: {len(files)} files")
    
    message = generate_commit_message(categorized_changes)
    
    if args.dry_run:
        commit_and_push(message, dry_run=True)
    else:
        success = commit_and_push(message, dry_run=False)
        if success and not args.no_push:
            # Push is already done in commit_and_push
            pass


if __name__ == "__main__":
    main()

