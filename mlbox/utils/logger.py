import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import traceback

class MLBoxLogger:
    """Simple logging system for MLBox with request/response tracking and artifact storage"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.logs_dir = base_dir / "logs"
        self.artifacts_dir = base_dir / "artifacts"
        
    def _write_json_log(self, log_file: Path, data: Dict[str, Any]):
        """Write JSON log entry to file"""
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing logs or create new list
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Add new log entry
        logs.append(data)
        
        # Write back to file
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False, default=str)
    
    def log_request(self, service: str, request_data: Dict[str, Any]) -> str:
        """Log a new request and return request ID"""
        request_id = str(uuid.uuid4())
        
        # Format request data according to user specification
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "client_ip": request_data.get("client_ip", "unknown"),
            "input_image_filename": request_data.get("file", "unknown"),
            "service_code": request_data.get("service_code", "1"),
            "alias": request_data.get("alias", ""),
            "key": request_data.get("key", ""),
            "response_method": request_data.get("response_method", ""),
            "response_endpoint": request_data.get("response_endpoint", "")
        }
        
        # Save to service-specific request log (single file)
        log_file = self.logs_dir / service / "peanut_requests.log"
        self._write_json_log(log_file, log_entry)
        
        return request_id
    
    def log_response(self, service: str, request_id: str, response_data: Dict[str, Any]):
        """Log a response with the exact format specified by user"""
        
        # Format response data according to user specification
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": response_data.get("processing_time_seconds", 0) * 1000,  # Convert to milliseconds
            "status": response_data.get("status", "unknown"),
            "error_message": response_data.get("error_message"),
            "output_xlsx_path": response_data.get("output_xlsx_path"),
            "message": response_data.get("message", ""),
            "output_excel": response_data.get("output_xlsx_path")  # Duplicate field as specified
        }
        
        # Save to service-specific response log (single file)
        log_file = self.logs_dir / service / "peanut_response.log"
        self._write_json_log(log_file, log_entry)
    
    def log_error(self, service: str, request_id: Optional[str], error: Exception, 
                  context: Optional[Dict[str, Any]] = None):
        """Log an error"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "service": service,
            "request_id": request_id or "system",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        # Save to error log (simple text format for Zabbix)
        error_log = self.logs_dir / "errors.log"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        
        with open(error_log, 'a', encoding='utf-8') as f:
            f.write(f"{error_entry['timestamp']} | ERROR | {service} | {request_id or 'system'} | {error_entry['error_type']}: {error_entry['error_message']}\n")
    
    def save_artifact(self, service: str, artifact_type: str, 
                     file_path: Path, request_id: str, metadata: Optional[Dict[str, Any]] = None):
        """Save an artifact file"""
        # Create artifact directory structure (simple, no date-based organization)
        artifact_dir = self.artifacts_dir / service / artifact_type
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate artifact filename with request_id
        original_filename = file_path.name
        artifact_filename = f"{request_id}_{original_filename}"
        artifact_path = artifact_dir / artifact_filename
        
        # Copy file to artifact location
        if file_path.exists():
            import shutil
            shutil.copy2(file_path, artifact_path)
            return str(artifact_path)
        
        return None

# Global logger instance
_mlbox_logger = None

def get_logger(base_dir: Path) -> MLBoxLogger:
    """Get MLBox logger instance"""
    return MLBoxLogger(base_dir) 