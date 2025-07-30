# MLBox Logging and Monitoring System

## 📋 Overview

The MLBox system includes a comprehensive logging and monitoring infrastructure that captures detailed metadata about every request, processing stage, and result. This system is designed for production monitoring, debugging, and performance analysis.

## 🏗️ Architecture

### Directory Structure
```
MLBox/
├── 📁 logs/
│   ├── 📁 requests/          # Request logs (JSON format)
│   ├── 📁 responses/         # Response logs (JSON format)
│   ├── 📁 errors/            # Error logs (for Zabbix monitoring)
│   ├── 📁 performance/       # Performance metrics
│   └── 📁 audit/             # Audit trail
├── 📁 artifacts/
│   ├── 📁 images/            # Saved input images
│   ├── 📁 results/           # Generated Excel files
│   └── 📁 metadata/          # Complete request metadata
└── 📁 tmp/                   # Temporary processing files
```

## 📊 Metadata Collection

### What is Metadata?

**Metadata** is "data about data" - additional information that describes, explains, or provides context about your main data. In MLBox, metadata helps us understand and track everything about each request and response.

### Types of Metadata Collected

#### 1. Request Metadata
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:45.123456",
  "client_ip": "192.168.1.100",
  "user_agent": "curl/7.68.0",
  "image_filename": "D21010101000001__14972__V223-011__0003.jpg",
  "image_size_bytes": 2048576,
  "image_hash": "a1b2c3d4e5f6...",
  "image_dimensions": [1920, 1080],
  "json_params": {
    "alias": "DMS",
    "key": "9127673     1",
    "response_method": "HTTP_POST_REQUEST",
    "response_endpoint": "https://ite.roshen.com:4433/WS/api/_MLBOX_HANDLE_RESPONSE"
  }
}
```

#### 2. Processing Metadata
```json
{
  "stage": "image_preprocessing",
  "start_time": 1705311045.123,
  "end_time": 1705311047.568,
  "duration_seconds": 2.445,
  "memory_usage_mb": 512.5,
  "cpu_percent": 45.2,
  "model_version": "v1.2.3",
  "confidence_threshold": 0.8,
  "detected_objects": 15,
  "processing_errors": 0
}
```

#### 3. Result Metadata
```json
{
  "status": "success",
  "message": "Processing completed successfully",
  "excel_file_path": "/artifacts/results/2024-01-15/550e8400_103045_report.xlsx",
  "excel_file_size": 1024000,
  "total_peanuts_detected": 150,
  "quality_score": 0.92,
  "processing_time_total": 5.67,
  "timestamp": "2024-01-15T10:30:50.789012"
}
```

#### 4. System Metadata
```json
{
  "cpu_count": 8,
  "memory_total_gb": 16.0,
  "memory_available_gb": 8.5,
  "platform": "Linux-5.4.0-x86_64",
  "python_version": "3.11.9",
  "timestamp": "2024-01-15T10:30:45.123456"
}
```

## 🔍 Logging Features

### 1. Request Tracking
- **Unique Request IDs**: Each request gets a UUID for tracking
- **Image Integrity**: MD5 hash verification of input images
- **Client Information**: IP address, user agent, request details
- **Timing**: Precise timestamps for all operations

### 2. Performance Monitoring
- **Processing Time**: Per-stage and total processing time
- **Resource Usage**: Memory and CPU consumption
- **Throughput**: Requests per second, batch processing metrics
- **Bottleneck Detection**: Identify slow processing stages

### 3. Error Tracking
- **Structured Error Logs**: JSON format for easy parsing
- **Stack Traces**: Complete error context
- **Error Classification**: Different error types and stages
- **Zabbix Integration**: Error logs formatted for monitoring

### 4. Artifact Storage
- **Input Images**: Saved with metadata for debugging
- **Result Files**: Excel reports with processing context
- **Metadata Files**: Complete request lifecycle information
- **Organized Storage**: Date-based directory structure

## 🚀 Usage Examples

### 1. Basic Request Processing
```python
from mlbox.utils.logger import get_logger
from mlbox.utils.metadata_collector import get_metadata_collector

# Initialize logging
logger = get_logger(ROOT_DIR)
collector = get_metadata_collector()

# Start request tracking
request_id = logger.start_request({
    "image_filename": "test.jpg",
    "image_size_bytes": 1024000,
    "alias": "TEST",
    "key": "12345"
})

# Track processing stages
collector.start_processing_stage("image_preprocessing")
# ... processing code ...
collector.end_processing_stage("image_preprocessing", detected_objects=10)

# Log response
logger.log_response(
    request_id=request_id,
    response_data={"status": "success", "excel_file": "result.xlsx"},
    processing_time=5.67,
    status="success"
)
```

### 2. Error Handling
```python
try:
    # Processing code
    pass
except Exception as e:
    logger.log_error(
        request_id=request_id,
        error=e,
        context={"stage": "processing", "model_version": "v1.2.3"}
    )
```

### 3. Artifact Saving
```python
# Save input image
logger.save_artifact(
    request_id=request_id,
    artifact_type="images",
    file_path=Path("input.jpg"),
    metadata={"original_filename": "test.jpg"}
)

# Save result file
logger.save_artifact(
    request_id=request_id,
    artifact_type="results",
    file_path=Path("result.xlsx"),
    metadata={"result_type": "excel_report"}
)
```

## 📈 Monitoring with Zabbix

### Monitoring Script Usage

The `scripts/monitoring.py` script provides various metrics for Zabbix:

```bash
# Get request statistics
python scripts/monitoring.py --metric request_stats --hours 1

# Get error count
python scripts/monitoring.py --metric error_count --hours 1

# Get performance metrics
python scripts/monitoring.py --metric performance

# Get system health
python scripts/monitoring.py --metric system_health

# Check service health
python scripts/monitoring.py --metric service_health

# Get log file sizes
python scripts/monitoring.py --metric log_size --log-type errors
```

### Zabbix Configuration

#### 1. UserParameter Configuration
Add to `/etc/zabbix/zabbix_agentd.conf`:
```ini
UserParameter=mlbox.request_stats,python /path/to/MLBox/scripts/monitoring.py --metric request_stats --hours 1
UserParameter=mlbox.error_count,python /path/to/MLBox/scripts/monitoring.py --metric error_count --hours 1
UserParameter=mlbox.system_health,python /path/to/MLBox/scripts/monitoring.py --metric system_health
UserParameter=mlbox.service_health,python /path/to/MLBox/scripts/monitoring.py --metric service_health
```

#### 2. Key Metrics to Monitor
- **Request Rate**: Number of requests per minute
- **Error Rate**: Number of errors per hour
- **Processing Time**: Average processing time
- **Memory Usage**: Peak memory consumption
- **Disk Usage**: Log and artifact storage
- **Service Health**: HTTP endpoint availability

#### 3. Alert Thresholds
```yaml
# High error rate
Error Count > 10 per hour

# Slow processing
Average Processing Time > 10 seconds

# High memory usage
Memory Usage > 80%

# Service down
Service Health != "healthy"

# Disk space low
Disk Usage > 90%
```

## 🔧 Configuration

### Environment Variables
```bash
# Log level
MLBOX_LOG_LEVEL=INFO

# Log retention (days)
MLBOX_LOG_RETENTION=30

# Artifact retention (days)
MLBOX_ARTIFACT_RETENTION=90

# Max log file size (MB)
MLBOX_MAX_LOG_SIZE=100
```

### Log Rotation
Logs are automatically rotated:
- **Daily rotation**: New log files each day
- **Compression**: Old logs are compressed with gzip
- **Retention**: Configurable retention period
- **Size limits**: Maximum file size limits

## 📊 Analytics and Reporting

### 1. Request Analytics
```python
# Get daily statistics
stats = logger.get_request_stats("2024-01-15")
print(f"Total requests: {stats['total_requests']}")
print(f"Success rate: {stats['successful_requests'] / stats['total_requests'] * 100:.1f}%")
print(f"Average processing time: {stats['avg_processing_time']:.2f}s")
```

### 2. Performance Analysis
```python
# Get performance metrics
metrics = collector.get_performance_metrics()
print(f"Peak memory usage: {metrics['peak_memory_usage_mb']:.1f} MB")
print(f"Total processing time: {metrics['total_processing_time']:.2f}s")
print(f"Total errors: {metrics['total_errors']}")
```

### 3. Error Analysis
```python
# Analyze error patterns
error_logs = logger.logs_dir / "errors"
for log_file in error_logs.glob("*.json"):
    with open(log_file, 'r') as f:
        errors = json.load(f)
        for error in errors:
            print(f"Error type: {error['error_type']}")
            print(f"Error message: {error['error_message']}")
            print(f"Context: {error['context']}")
```

## 🛠️ Maintenance

### 1. Log Cleanup
```bash
# Clean old logs (older than 30 days)
find /path/to/MLBox/logs -name "*.log" -mtime +30 -delete

# Clean old artifacts (older than 90 days)
find /path/to/MLBox/artifacts -name "*" -mtime +90 -delete
```

### 2. Disk Space Monitoring
```bash
# Check log directory sizes
du -sh /path/to/MLBox/logs/*
du -sh /path/to/MLBox/artifacts/*
```

### 3. Performance Optimization
- **Log compression**: Enable gzip compression for old logs
- **Batch writing**: Write logs in batches for better performance
- **Async logging**: Use async logging for high-throughput scenarios
- **Log level adjustment**: Reduce log level in production if needed

## 🔒 Security Considerations

### 1. Data Privacy
- **PII Protection**: Ensure no personal data in logs
- **Image Anonymization**: Consider anonymizing saved images
- **Access Control**: Restrict access to log directories

### 2. Log Security
- **File Permissions**: Set appropriate file permissions
- **Encryption**: Consider encrypting sensitive log data
- **Audit Trail**: Maintain audit logs for log access

### 3. Compliance
- **Data Retention**: Follow data retention policies
- **GDPR Compliance**: Ensure compliance with data protection regulations
- **Industry Standards**: Follow industry-specific logging standards

## 📝 Best Practices

### 1. Logging Best Practices
- **Structured Logging**: Use JSON format for machine readability
- **Consistent Format**: Maintain consistent log format across all components
- **Appropriate Levels**: Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- **Context Information**: Include relevant context in log messages

### 2. Monitoring Best Practices
- **Proactive Monitoring**: Set up alerts before issues occur
- **Baseline Establishment**: Establish performance baselines
- **Trend Analysis**: Monitor trends over time
- **Capacity Planning**: Use metrics for capacity planning

### 3. Maintenance Best Practices
- **Regular Cleanup**: Schedule regular log and artifact cleanup
- **Backup Strategy**: Implement backup strategy for important logs
- **Documentation**: Keep documentation updated
- **Testing**: Test monitoring and alerting systems regularly

## 🆘 Troubleshooting

### Common Issues

#### 1. High Disk Usage
```bash
# Check disk usage
df -h /path/to/MLBox

# Find large files
find /path/to/MLBox -type f -size +100M

# Clean up old files
find /path/to/MLBox/logs -mtime +7 -delete
```

#### 2. Slow Processing
```bash
# Check system resources
top
htop
iostat

# Analyze performance logs
tail -f /path/to/MLBox/logs/performance/performance.log
```

#### 3. Service Not Responding
```bash
# Check service status
curl http://localhost:8000/peanuts/health

# Check logs
tail -f /path/to/MLBox/logs/errors/errors.log

# Check system resources
free -h
df -h
```

### Debug Mode
Enable debug logging for troubleshooting:
```python
import logging
logging.getLogger('mlbox').setLevel(logging.DEBUG)
```

## 📚 Additional Resources

- [Loguru Documentation](https://loguru.readthedocs.io/)
- [Zabbix Documentation](https://www.zabbix.com/documentation)
- [Python Logging Best Practices](https://docs.python.org/3/howto/logging.html)
- [Monitoring and Observability](https://landing.google.com/sre/sre-book/chapters/monitoring-distributed-systems/) 