import json
import time
import psutil
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class SystemMetadata:
    """System information metadata"""
    cpu_count: int
    memory_total_gb: float
    memory_available_gb: float
    platform: str
    python_version: str
    timestamp: str

@dataclass
class RequestMetadata:
    """Request-specific metadata"""
    request_id: str
    client_ip: Optional[str]
    user_agent: Optional[str]
    content_type: str
    content_length: int
    timestamp: str
    image_filename: str
    image_size_bytes: int
    image_hash: str
    image_dimensions: Optional[tuple]
    json_params: Dict[str, Any]

@dataclass
class ProcessingMetadata:
    """Processing stage metadata"""
    stage: str
    start_time: float
    end_time: float
    duration_seconds: float
    memory_usage_mb: float
    cpu_percent: float
    model_version: Optional[str]
    confidence_threshold: Optional[float]
    detected_objects: Optional[int]
    processing_errors: int

@dataclass
class ResultMetadata:
    """Result-specific metadata"""
    status: str
    message: str
    excel_file_path: Optional[str]
    excel_file_size: Optional[int]
    total_peanuts_detected: Optional[int]
    quality_score: Optional[float]
    processing_time_total: float
    timestamp: str

class MetadataCollector:
    """Comprehensive metadata collection for MLBox requests"""
    
    def __init__(self):
        self.request_metadata: Optional[RequestMetadata] = None
        self.processing_stages: List[ProcessingMetadata] = []
        self.result_metadata: Optional[ResultMetadata] = None
        self.system_metadata: Optional[SystemMetadata] = None
        
    def collect_system_metadata(self) -> SystemMetadata:
        """Collect current system information"""
        memory = psutil.virtual_memory()
        
        self.system_metadata = SystemMetadata(
            cpu_count=psutil.cpu_count(),
            memory_total_gb=memory.total / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            platform=platform.platform(),
            python_version=platform.python_version(),
            timestamp=datetime.now().isoformat()
        )
        
        return self.system_metadata
    
    def start_request_metadata(self, request_id: str, image_data: bytes, 
                              image_filename: str, json_params: Dict[str, Any],
                              client_info: Optional[Dict[str, str]] = None) -> RequestMetadata:
        """Start collecting request metadata"""
        
        # Calculate image hash for integrity
        image_hash = hashlib.md5(image_data).hexdigest()
        
        # Get image dimensions if possible
        try:
            from PIL import Image
            from io import BytesIO
            image = Image.open(BytesIO(image_data))
            image_dimensions = image.size
        except Exception:
            image_dimensions = None
        
        self.request_metadata = RequestMetadata(
            request_id=request_id,
            client_ip=client_info.get('client_ip') if client_info else None,
            user_agent=client_info.get('user_agent') if client_info else None,
            content_type="multipart/form-data",
            content_length=len(image_data),
            timestamp=datetime.now().isoformat(),
            image_filename=image_filename,
            image_size_bytes=len(image_data),
            image_hash=image_hash,
            image_dimensions=image_dimensions,
            json_params=json_params
        )
        
        return self.request_metadata
    
    def start_processing_stage(self, stage: str, model_version: Optional[str] = None,
                              confidence_threshold: Optional[float] = None) -> str:
        """Start tracking a processing stage"""
        stage_id = f"{stage}_{int(time.time())}"
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        processing_metadata = ProcessingMetadata(
            stage=stage,
            start_time=time.time(),
            end_time=0.0,
            duration_seconds=0.0,
            memory_usage_mb=memory_info.rss / (1024**2),
            cpu_percent=process.cpu_percent(),
            model_version=model_version,
            confidence_threshold=confidence_threshold,
            detected_objects=None,
            processing_errors=0
        )
        
        self.processing_stages.append(processing_metadata)
        return stage_id
    
    def end_processing_stage(self, stage: str, detected_objects: Optional[int] = None,
                           processing_errors: int = 0):
        """End tracking a processing stage"""
        for stage_meta in self.processing_stages:
            if stage_meta.stage == stage and stage_meta.end_time == 0.0:
                stage_meta.end_time = time.time()
                stage_meta.duration_seconds = stage_meta.end_time - stage_meta.start_time
                stage_meta.detected_objects = detected_objects
                stage_meta.processing_errors = processing_errors
                
                # Update current memory and CPU usage
                process = psutil.Process()
                memory_info = process.memory_info()
                stage_meta.memory_usage_mb = memory_info.rss / (1024**2)
                stage_meta.cpu_percent = process.cpu_percent()
                break
    
    def create_result_metadata(self, status: str, message: str, 
                             excel_file_path: Optional[str] = None,
                             total_peanuts_detected: Optional[int] = None,
                             quality_score: Optional[float] = None) -> ResultMetadata:
        """Create result metadata"""
        
        # Calculate total processing time
        total_processing_time = sum(stage.duration_seconds for stage in self.processing_stages)
        
        # Get Excel file size if it exists
        excel_file_size = None
        if excel_file_path and Path(excel_file_path).exists():
            excel_file_size = Path(excel_file_path).stat().st_size
        
        self.result_metadata = ResultMetadata(
            status=status,
            message=message,
            excel_file_path=excel_file_path,
            excel_file_size=excel_file_size,
            total_peanuts_detected=total_peanuts_detected,
            quality_score=quality_score,
            processing_time_total=total_processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        return self.result_metadata
    
    def get_complete_metadata(self) -> Dict[str, Any]:
        """Get complete metadata for the request"""
        if not self.request_metadata:
            raise ValueError("Request metadata not initialized")
        
        complete_metadata = {
            "request": asdict(self.request_metadata),
            "processing_stages": [asdict(stage) for stage in self.processing_stages],
            "result": asdict(self.result_metadata) if self.result_metadata else None,
            "system": asdict(self.system_metadata) if self.system_metadata else None,
            "summary": self._create_summary()
        }
        
        return complete_metadata
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create a summary of the request processing"""
        if not self.request_metadata:
            return {}
        
        summary = {
            "request_id": self.request_metadata.request_id,
            "total_processing_time": sum(stage.duration_seconds for stage in self.processing_stages),
            "total_stages": len(self.processing_stages),
            "peak_memory_usage": max(stage.memory_usage_mb for stage in self.processing_stages) if self.processing_stages else 0,
            "total_processing_errors": sum(stage.processing_errors for stage in self.processing_stages),
            "success": self.result_metadata.status == "success" if self.result_metadata else False
        }
        
        return summary
    
    def save_metadata(self, file_path: Path):
        """Save complete metadata to file"""
        metadata = self.get_complete_metadata()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        if not self.processing_stages:
            return {}
        
        return {
            "total_processing_time": sum(stage.duration_seconds for stage in self.processing_stages),
            "avg_stage_time": sum(stage.duration_seconds for stage in self.processing_stages) / len(self.processing_stages),
            "peak_memory_usage_mb": max(stage.memory_usage_mb for stage in self.processing_stages),
            "total_errors": sum(stage.processing_errors for stage in self.processing_stages),
            "stages_completed": len([s for s in self.processing_stages if s.end_time > 0])
        }

# Global metadata collector instance
_metadata_collector = None

def get_metadata_collector() -> MetadataCollector:
    """Get or create global metadata collector instance"""
    global _metadata_collector
    if _metadata_collector is None:
        _metadata_collector = MetadataCollector()
    return _metadata_collector 