import os
from pathlib import Path

# Defining the root directory dynamically based on this file's location
ROOT_DIR = Path(__file__).parent.parent
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
