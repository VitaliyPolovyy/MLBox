#!/usr/bin/env python3
"""
Test script to verify Ray logging functionality
"""

import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import ray
from mlbox.utils.logger import get_logger
from mlbox.settings import ROOT_DIR

# Initialize Ray
ray.init()

@ray.remote
def test_worker_logging():
    """Test function to verify logging works in Ray workers"""
    # Get logger in Ray worker
    worker_logger = get_logger(ROOT_DIR)
    
    # Test logging
    worker_logger.info("TestWorker", "This is a test log from Ray worker")
    worker_logger.info("TestWorker", "Processing some data...")
    
    # Simulate some work
    time.sleep(1)
    
    worker_logger.info("TestWorker", "Work completed successfully")
    
    return "Worker completed"

def main():
    print("Testing Ray logging functionality...")
    
    # Test main process logging
    main_logger = get_logger(ROOT_DIR)
    main_logger.info("MainProcess", "Starting Ray logging test")
    
    # Submit Ray task
    result_ref = test_worker_logging.remote()
    
    # Wait for result
    result = ray.get(result_ref)
    
    main_logger.info("MainProcess", f"Ray task completed: {result}")
    main_logger.info("MainProcess", "Ray logging test completed")
    
    print("Test completed. Check app.log for logging output.")
    
    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    main()

