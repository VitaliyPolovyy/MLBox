# MLBox Project - Claude Code Configuration

## Project Overview
MLBox is a machine learning toolkit with various ML services including:
- **LabelGuard**: Label verification service using OCR and text comparison
- **Peanuts**: ML service for peanut-related processing
- Utilities for logging, artifact management, and deployment

## Development Environment
- **Python Version**: 3.11
- **Virtual Environment**: `.venv/bin/python`
- **Package Manager**: Poetry
- **Platform**: WSL2 on Windows
- **Default Log Level**: DEBUG (detailed artifacts enabled by default)

## Code Style & Conventions
- Follow existing code patterns and imports
- Use proper logging through `app_logger` instead of print statements
- Import `field` from dataclasses when using default_factory
- Use type hints consistently
- Prefer editing existing files over creating new ones

## Testing & Quality
- **Lint Command**: `ruff check .` (if available)
- **Type Check**: `mypy .` (if available)
- **Test Command**: Check README for project-specific test commands
- Always run linting/type checking after significant changes

## Artifact Management
- **Debug artifacts**: Saved to `artifacts/{service_name}/` when LOG_LEVEL=DEBUG
- **Naming convention**: `{input_filename}_{artifact_type}_{index}.{ext}`
- **Storage strategy**: 
  - INFO/ERROR: Save input + output + metrics only
  - DEBUG: Save all intermediate steps for troubleshooting

## Service-Specific Notes

### LabelGuard Service
- Uses PaddleOCR for layout detection and text extraction
- Processes PIL.Image inputs (converts RGB→BGR for OpenCV)
- Saves text blocks as artifacts in debug mode
- Located in `mlbox/services/LabelGuard/`

### Logging
- Use `app_logger.info(service_name, message)` format
- Available levels: DEBUG, INFO, WARNING, ERROR
- Logs saved to `logs/app.log`

## Common Commands
```bash
# Run LabelGuard service
.venv/bin/python "mlbox/services/LabelGuard/labelguard.py"

# Run with debug artifacts
LOG_LEVEL=DEBUG .venv/bin/python "mlbox/services/LabelGuard/labelguard.py"

# Check artifacts
ls -la artifacts/labelguard/
```

## Development Log (DEVELOPMENT.md)
- **Purpose**: Maintain session continuity and track project progress
- **Format**: Chronological entries with file focus
- **Required confirmation**: Always ask user before writing entries
- **Categories to include**:
  - Main achievement (file + what was implemented)
  - Issues encountered (major blockers/fixes)
  - Key decisions (architectural choices)
  - Performance notes (speed, accuracy, resources)
  - Dependencies added (new libraries/models)
  - Next steps (clear direction for continuation)

## Important Notes
- Always use absolute paths for file operations
- PIL.Image inputs should be converted to RGB before OpenCV processing
- WSL2 may have image display limitations - artifacts save correctly to disk
- Avoid creating new files unless explicitly required
- Check existing patterns before implementing new features