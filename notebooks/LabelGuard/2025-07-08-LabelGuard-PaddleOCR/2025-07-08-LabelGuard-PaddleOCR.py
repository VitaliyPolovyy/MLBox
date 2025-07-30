#!/usr/bin/env python3
"""
Simple Text Block Detector using PP-Structure
Focuses on layout detection only:
- Paragraphs vs Tables
- Spatial separation
- Visual separators
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any

# Core imports
from paddleocr import PPStructure
print(PPStructure)