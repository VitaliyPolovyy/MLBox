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
        """Configure Loguru to write to app.log with proper log level"""
        # Remove default stderr handler
        logger.remove()
        
        # Ensure the log file path is properly handled
        log_file_path = self.logs_dir / "app.log"
        
        # Create an empty log file if it doesn't exist
        try:
            log_file_path.touch(exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create log file {log_file_path}: {e}")
            # Fallback to console-only logging
            logger.add(
                lambda msg: print(msg, end=""),
                level=LOG_LEVEL,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
            )
            return
        
        # Add app.log handler with LOG_LEVEL from settings
        try:
            logger.add(
                str(log_file_path),  # Convert to string to avoid path issues
                level=LOG_LEVEL,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
                rotation="10 MB",
                retention="30 days",
                compression="zip",
                enqueue=True
            )
        except Exception as e:
            print(f"Warning: Could not add file handler for {log_file_path}: {e}")
            # Fallback to console-only logging
            logger.add(
                lambda msg: print(msg, end=""),
                level=LOG_LEVEL,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
            )
            return
        
        # Add console handler for development
        if LOG_LEVEL in ['DEBUG', 'INFO']:
            logger.add(
                lambda msg: print(msg, end=""),
                level=LOG_LEVEL,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
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
        formatted_message = f"{service} | {message}"
        logger.error(formatted_message)
    
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
    
    def save_artifact(self, service: str, file_name: str, data: Any):
        """Save data as artifact to artifacts/service_name/filename"""
        # Create service directory
        service_dir = self.artifacts_dir / service
        service_dir.mkdir(parents=True, exist_ok=True)
        
        # Create artifact file path
        artifact_path = service_dir / file_name
        
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
            
            logger.info(f"Artifact saved: {service}/{file_name}")
            return str(artifact_path)
            
        except Exception as e:
            logger.error(f"Failed to save artifact {service}/{file_name}: {e}")
            return None

@lru_cache(maxsize=None)
def get_logger(base_dir: Path) -> Logger:
    logger.remove()
    return Logger(base_dir)

@lru_cache(maxsize=None)
def get_artifact_service(base_dir: Path) -> ArtifactService:
    return ArtifactService(base_dir)