#!/usr/bin/env python3
"""
Test script for MLBox Simple Logger System
"""

import os
import sys
import json
import uuid
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from mlbox.utils.logger import get_logger, get_artifact_service
from mlbox.settings import ROOT_DIR, LOG_LEVEL

def test_simple_logger_system():
    """Test the simple logger system"""
    
    print(f"Testing MLBox Simple Logger System with LOG_LEVEL={LOG_LEVEL}")
    print(f"Project root: {ROOT_DIR}")
    
    # Initialize logger and artifact service
    app_logger = get_logger(ROOT_DIR)
    artifact_service = get_artifact_service(ROOT_DIR)
    
    # Test logger levels
    print("\n=== Testing Logger Levels ===")
    app_logger.debug("general", "Debug message - only visible in DEBUG mode")
    app_logger.info("general", "Info message - application started")
    app_logger.warning("general", "Warning message - resource usage high")
    app_logger.error("general", "Error message - something went wrong")
    
    # Test request processing simulation
    print("\n=== Testing Request Processing ===")
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    
    # Simulate request data
    request_data = {
        "client_ip": "127.0.0.1",
        "file": "test_image.jpg",
        "service_code": "1",
        "alias": "TEST",
        "key": "test_key",
        "response_method": "HTTP_POST",
        "response_endpoint": "http://test.com"
    }
    
    # Log request
    app_logger.info("peanuts", f"Request received | request_id={request_id} | file={request_data.get('file', 'unknown')} | alias={request_data.get('alias', '')} | key={request_data.get('key', '')}")
    
    # Save request data as artifact
    artifact_service.save_artifact("peanuts", f"test_request_{request_id}.json", request_data)
    
    # Simulate response data
    response_data = {
        "processing_time_seconds": 1.5,
        "status": "SUCCESS",
        "message": "Test completed successfully",
        "output_xlsx_path": "/path/to/test.xlsx"
    }
    
    # Log response
    processing_time = response_data.get("processing_time_seconds", 0)
    app_logger.info("peanuts", f"Response sent | request_id={request_id} | status=SUCCESS | time={processing_time:.2f}s")
    
    # Save response data as artifact
    artifact_service.save_artifact("peanuts", f"test_response_{request_id}.json", response_data)
    
    print("\n=== Testing Artifact Service ===")
    
    # Test saving different data types
    test_text = "Test artifact content"
    artifact_service.save_artifact("peanuts", f"test_text_{request_id}.txt", test_text)
    
    test_dict = {"key": "value", "number": 42}
    artifact_service.save_artifact("peanuts", f"test_dict_{request_id}.json", test_dict)
    
    print("\n=== Log Files Created ===")
    logs_dir = ROOT_DIR / "logs"
    
    # Check app.log
    if (logs_dir / "app.log").exists():
        print("✅ app.log created")
        with open(logs_dir / "app.log", "r") as f:
            print("App log content:")
            print(f.read())
    else:
        print("❌ app.log not created")
    
    # Check artifacts directory
    print("\n=== Artifacts Directory ===")
    artifacts_dir = ROOT_DIR / "artifacts"
    if artifacts_dir.exists():
        print("✅ artifacts directory created")
        for item in artifacts_dir.rglob("*"):
            if item.is_file():
                print(f"  📄 {item.relative_to(ROOT_DIR)}")
    else:
        print("❌ artifacts directory not created")
    
    print(f"\n✅ Simple logger system test completed with LOG_LEVEL={LOG_LEVEL}")

if __name__ == "__main__":
    test_simple_logger_system() 