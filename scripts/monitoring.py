#!/usr/bin/env python3
"""
MLBox Monitoring Script for Zabbix
This script provides various metrics for monitoring the MLBox system
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlbox.utils.logger import get_logger
from mlbox.utils.metadata_collector import get_metadata_collector
from mlbox.settings import ROOT_DIR

def get_request_stats(hours: int = 1) -> Dict[str, Any]:
    """Get request statistics for the last N hours"""
    logger = get_logger(ROOT_DIR)
    return logger.get_request_stats()

def get_error_count(hours: int = 1) -> int:
    """Get error count for the last N hours"""
    logger = get_logger(ROOT_DIR)
    logs_dir = logger.logs_dir / "errors"
    
    error_count = 0
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    # Count errors in error log files
    for log_file in logs_dir.glob("*.log"):
        if log_file.name == "errors.log":
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            error_count += 1
            except Exception:
                continue
    
    return error_count

def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics"""
    collector = get_metadata_collector()
    return collector.get_performance_metrics()

def get_system_health() -> Dict[str, Any]:
    """Get system health metrics"""
    import psutil
    
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / (1024**3),
        "disk_percent": disk.percent,
        "disk_free_gb": disk.free / (1024**3)
    }

def check_service_health() -> Dict[str, Any]:
    """Check if the service is healthy"""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/peanuts/health", timeout=5)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "response_time": response.elapsed.total_seconds(),
            "status_code": response.status_code
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "response_time": None,
            "status_code": None
        }

def get_log_file_size(log_type: str) -> int:
    """Get size of log files in bytes"""
    logger = get_logger(ROOT_DIR)
    logs_dir = logger.logs_dir / log_type
    
    total_size = 0
    if logs_dir.exists():
        for file_path in logs_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    
    return total_size

def main():
    parser = argparse.ArgumentParser(description="MLBox Monitoring Script")
    parser.add_argument("--metric", required=True, 
                       choices=["request_stats", "error_count", "performance", 
                               "system_health", "service_health", "log_size"])
    parser.add_argument("--hours", type=int, default=1, 
                       help="Time window in hours (default: 1)")
    parser.add_argument("--log-type", choices=["requests", "responses", "errors", "performance"],
                       help="Log type for log_size metric")
    
    args = parser.parse_args()
    
    try:
        if args.metric == "request_stats":
            stats = get_request_stats(args.hours)
            print(json.dumps(stats, indent=2))
            
        elif args.metric == "error_count":
            count = get_error_count(args.hours)
            print(count)
            
        elif args.metric == "performance":
            metrics = get_performance_metrics()
            print(json.dumps(metrics, indent=2))
            
        elif args.metric == "system_health":
            health = get_system_health()
            print(json.dumps(health, indent=2))
            
        elif args.metric == "service_health":
            health = check_service_health()
            print(json.dumps(health, indent=2))
            
        elif args.metric == "log_size":
            if not args.log_type:
                print("ERROR: --log-type is required for log_size metric", file=sys.stderr)
                sys.exit(1)
            size = get_log_file_size(args.log_type)
            print(size)
            
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 