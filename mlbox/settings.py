import os
from pathlib import Path

# Defining the root directory dynamically based on this file's location
ROOT_DIR = Path(__file__).parent.parent

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')
