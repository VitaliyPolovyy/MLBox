from pathlib import Path
from typing import Any
from functools import lru_cache
from loguru import logger
from mlbox.settings import LOG_LEVEL

class Logger:
    """Simple logger for all application events with structured format"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.logs_dir = base_dir / "logs"
        
        # Create logs directory
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure Loguru for app.log
        self._configure_loguru()
    
    def _configure_loguru(self):
        """Configure Loguru with both console and file output (Loguru defaults)"""
        import sys
        
        # Remove default stderr handler
        logger.remove()
        
        # Console output (always enabled)
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>"
        )
        
        # File output (always enabled)
        log_file_path = self.logs_dir / "app.log"
        logger.add(
            str(log_file_path),
            level=LOG_LEVEL,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip",
            enqueue=True
        )
    
    def debug(self, service: str, message: str):
        formatted_message = f"{service} | {message}"
        logger.debug(formatted_message)
    
    def info(self, service: str, message: str):
        formatted_message = f"{service} | {message}"
        logger.info(formatted_message)
    
    def warning(self, service: str, message: str):
        formatted_message = f"{service} | {message}"
        logger.warning(formatted_message)
    
    def error(self, service: str, message: str):
        # Escape curly braces in message to prevent loguru format string interpretation
        escaped_message = message.replace('{', '{{').replace('}', '}}')
        formatted_message = f"{service} | {escaped_message}"
        logger.error(formatted_message, exc_info=False)  # exc_info=False since tracebacks are included in message
    
    @property
    def level(self) -> str:
        """Get the current log level"""
        return LOG_LEVEL

class ArtifactService:
    """Simple artifact storage service"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.artifacts_dir = base_dir / "artifacts"
        
        # Create artifacts directory
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.current_service = None
    
    def get_service_dir(self, service: str) -> Path:
        """Get the directory path for a specific service"""
        service_dir = self.artifacts_dir / service
        service_dir.mkdir(parents=True, exist_ok=True)
        return service_dir
    
    def save_artifact(self, service: str, file_name: str, data: Any, sub_folder: str = None):
        """Save data as artifact to artifacts/service_name/filename or artifacts/service_name/sub_folder/filename"""
        # Get service directory (creates if doesn't exist)
        self.service_dir = self.get_service_dir(service)

        # Create artifact file path
        if sub_folder:
            artifact_dir = self.service_dir / sub_folder
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / file_name
        else:
            artifact_path = self.service_dir / file_name
        
        try:
            # Handle different data types
            if isinstance(data, (dict, list)):
                # Save JSON data
                import json
                with open(artifact_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str)
            elif isinstance(data, str):
                # Save text data
                with open(artifact_path, 'w', encoding='utf-8') as f:
                    f.write(data)
            elif isinstance(data, bytes):
                # Save binary data
                with open(artifact_path, 'wb') as f:
                    f.write(data)
            elif hasattr(data, 'save'):
                # Save PIL Image or similar objects with save method
                data.save(artifact_path)
            else:
                # Try to convert to string
                with open(artifact_path, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            
            if sub_folder:
                logger.debug(f"Artifact saved: {service}/{sub_folder}/{file_name}")
            else:
                logger.debug(f"Artifact saved: {service}/{file_name}")
            return str(artifact_path)
            
        except Exception as e:
            logger.error(f"Failed to save artifact {service}/{file_name}: {e}")
            return None

@lru_cache(maxsize=None)
def get_logger(base_dir: Path) -> Logger:
    return Logger(base_dir)

@lru_cache(maxsize=None)
def get_artifact_service(base_dir: Path) -> ArtifactService:
    return ArtifactService(base_dir)