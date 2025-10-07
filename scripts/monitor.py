#!/usr/bin/env python3
"""
Simple monitoring script for MLBox
Usage: python scripts/monitor.py --metric error_count
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

def count_errors(log_file: Path, hours: int = 1) -> int:
    """Count errors in the last N hours"""
    if not log_file.exists():
        return 0
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    error_count = 0
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parse timestamp from error log format: "2024-01-15T10:30:45.123 | ERROR | ..."
                timestamp_str = line.split(' | ')[0]
                timestamp = datetime.fromisoformat(timestamp_str)
                
                if timestamp >= cutoff_time:
                    error_count += 1
            except (IndexError, ValueError):
                # Skip malformed lines
                continue
    
    return error_count

def count_requests(log_file: Path, hours: int = 1) -> int:
    """Count requests in the last N hours"""
    if not log_file.exists():
        return 0
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    request_count = 0
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
            
        for log_entry in logs:
            try:
                timestamp = datetime.fromisoformat(log_entry.get('timestamp', ''))
                if timestamp >= cutoff_time:
                    request_count += 1
            except (ValueError, KeyError):
                continue
                
    except (json.JSONDecodeError, FileNotFoundError):
        return 0
    
    return request_count

def main():
    parser = argparse.ArgumentParser(description="MLBox Monitoring Script")
    parser.add_argument("--metric", required=True, 
                       choices=["error_count", "request_count", "response_count"])
    parser.add_argument("--hours", type=int, default=1, 
                       help="Time window in hours (default: 1)")
    parser.add_argument("--service", default="peanuts",
                       help="Service name (default: peanuts)")
    parser.add_argument("--base-dir", default="/app",
                       help="Base directory (default: /app)")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    try:
        if args.metric == "error_count":
            error_log = base_dir / "logs" / "errors.log"
            count = count_errors(error_log, args.hours)
            print(count)
            
        elif args.metric == "request_count":
            request_log = base_dir / "logs" / args.service / "requests.log"
            count = count_requests(request_log, args.hours)
            print(count)
            
        elif args.metric == "response_count":
            response_log = base_dir / "logs" / args.service / "responses.log"
            count = count_requests(response_log, args.hours)
            print(count)
            
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    import json
    main() 