# MLBox Simple Logger System

## Overview

MLBox uses a **simple logging system** with Loguru for comprehensive logging and separate artifact storage:

- **Single Logger**: All application events with levels (DEBUG, INFO, WARNING, ERROR)
- **Artifact Service**: Simple file storage for all services

## Architecture

### **1. Single Logger (All Events)**
```python
class Logger:
    def __init__(self, base_dir: Path):
        # Writes to logs/app.log
        # All application events with levels
        # Environment-based level control
    
    def debug(self, message: str)
    def info(self, message: str)
    def warning(self, message: str)
    def error(self, message: str)
```

### **2. Artifact Service (File Storage)**
```python
class ArtifactService:
    def __init__(self, base_dir: Path):
        # Manages artifacts/ directory
        # Simple file storage for all services
    
    def save_artifact(self, service: str, file_path: Path, request_id: str)
```

## Structure

```
logs/
├── app.log (all events with levels: DEBUG, INFO, WARNING, ERROR)
artifacts/
├── peanuts/ (all peanuts artifacts)
├── ocr_service/ (all OCR artifacts)
└── image_processing/ (all image processing artifacts)
```

## Configuration

### Environment Variable

Set `LOG_LEVEL` environment variable to control logging:

```bash
# Development (default)
export LOG_LEVEL=INFO

# Production
export LOG_LEVEL=ERROR

# Debug mode
export LOG_LEVEL=DEBUG
```

### Docker

```bash
# Development
docker run -e LOG_LEVEL=INFO your-app

# Production
docker run -e LOG_LEVEL=ERROR your-app
```

## Log Levels

| Level | Description | Production | Development |
|-------|-------------|------------|-------------|
| `DEBUG` | Detailed debugging | ❌ | ✅ |
| `INFO` | General information | ❌ | ✅ |
| `WARNING` | Warning messages | ✅ | ✅ |
| `ERROR` | Error messages | ✅ | ✅ |
| `CRITICAL` | Critical errors | ✅ | ✅ |

## Usage

### Logger (All Events)

```python
from mlbox.utils.logger import get_logger
from mlbox.settings import ROOT_DIR

# Initialize logger
app_logger = get_logger(ROOT_DIR)

# System events
app_logger.info("Application started")
app_logger.warning("Resource usage high")
app_logger.error("System error occurred")
app_logger.debug("Debug information")

# Business events
app_logger.info("Request received | service=peanuts | request_id=abc123 | file=image.jpg")
app_logger.info("Response sent | service=peanuts | request_id=abc123 | status=SUCCESS | time=1.5s")
app_logger.error("Processing error | service=peanuts | request_id=abc123 | error=Invalid input")
```

### Artifact Service (File Storage)

```python
from mlbox.utils.logger import get_artifact_service

# Initialize artifact service
artifact_service = get_artifact_service(ROOT_DIR)

# Save any file as artifact
artifact_path = artifact_service.save_artifact(
    service="peanuts",
    file_path=image_path,
    request_id=request_id
)

# Save JSON data as artifact
import json
request_json_path = Path("request_data.json")
with open(request_json_path, 'w') as f:
    json.dump(request_data, f, indent=2)
artifact_service.save_artifact("peanuts", request_json_path, request_id)
```

### Complete Example

```python
from mlbox.utils.logger import get_logger, get_artifact_service
from mlbox.settings import ROOT_DIR
import json
import uuid

# Initialize services
app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)

# Generate request ID
request_id = str(uuid.uuid4())

# System startup
app_logger.info("Peanuts service starting")

# Process request
app_logger.info(f"Request received | service=peanuts | request_id={request_id} | file=image.jpg")

# Save request data as artifact
request_data = {"client_ip": "127.0.0.1", "file": "image.jpg"}
request_json_path = Path("request_data.json")
with open(request_json_path, 'w') as f:
    json.dump(request_data, f, indent=2)
artifact_service.save_artifact("peanuts", request_json_path, request_id)

# Save input image
artifact_service.save_artifact("peanuts", image_path, request_id)

# Process and log response
app_logger.info(f"Response sent | service=peanuts | request_id={request_id} | status=SUCCESS | time=1.5s")

# Save response data as artifact
response_data = {"status": "SUCCESS", "processing_time": 1.5}
response_json_path = Path("response_data.json")
with open(response_json_path, 'w') as f:
    json.dump(response_data, f, indent=2)
artifact_service.save_artifact("peanuts", response_json_path, request_id)

# Save result file
artifact_service.save_artifact("peanuts", result_path, request_id)

# System event
app_logger.info("Request processing completed")
```

## Features

### ✅ Implemented
- Single logger with environment-based levels
- Simple artifact storage
- Request correlation with request IDs
- Automatic directory creation
- Error handling and recovery
- Log rotation and compression
- Clean separation of logging and artifacts

### 🔄 Future Enhancements
- Async logging
- Log validation
- Data sanitization
- Health checks
- Performance metrics

## Testing

Run the test script to verify the simple logger system:

```bash
python test_logger.py
```

This will:
- Test all log levels
- Test request processing simulation
- Test artifact storage
- Verify file creation and structure
- Display log content

## Monitoring

### Log Analysis

Use standard tools to analyze logs:

```bash
# View all events
tail -f logs/app.log

# Filter by level
grep "ERROR" logs/app.log
grep "WARNING" logs/app.log

# Filter by service
grep "service=peanuts" logs/app.log

# Search for specific request
grep "request_id=abc123" logs/app.log

# Count events by level
grep "ERROR" logs/app.log | wc -l
```

### Artifact Management

```bash
# List artifacts by service
ls -la artifacts/peanuts/

# Find artifacts by request ID
find artifacts/ -name "*abc123*"

# Check artifact sizes
du -sh artifacts/peanuts/
```

## Troubleshooting

### Common Issues

1. **Log files not created**
   - Check directory permissions
   - Verify LOG_LEVEL is set correctly

2. **High disk usage**
   - Logs rotate automatically at 10MB
   - Old logs are compressed and deleted after 30 days

3. **Missing log entries**
   - Check LOG_LEVEL setting
   - Verify logger initialization

### Debug Mode

For debugging logging issues:

```bash
export LOG_LEVEL=DEBUG
python your-app.py
```

This will show all log levels and console output.

## Adding New Services

To add a new service:

1. **Use existing logger:**
```python
app_logger.info(f"Request received | service=ocr_service | request_id={request_id}")
```

2. **Use existing artifact service:**
```python
artifact_service.save_artifact("ocr_service", document_path, request_id)
```

3. **Artifacts automatically organized:**
```
artifacts/
├── peanuts/ (existing)
└── ocr_service/ (new service)
```

**That's it! No additional configuration needed.** 