#!/bin/bash

echo "Updating requirements.txt from pyproject.toml..."

# Create a temporary Python script to parse pyproject.toml
cat > /tmp/parse_deps.py << 'EOF'
import tomllib
import sys

def parse_dependencies():
    with open('pyproject.toml', 'rb') as f:
        data = tomllib.load(f)
    
    dependencies = data.get('project', {}).get('dependencies', [])
    
    with open('requirements.txt', 'w') as f:
        f.write("# Core ML and serving dependencies\n")
        f.write("ray[serve]>=2.48.0,<3.0.0\n")
        f.write("starlette>=0.47.2,<0.48.0\n")
        f.write("loguru>=0.7.3,<0.8.0\n\n")
        
        f.write("# Image processing\n")
        f.write("pillow>=11.3.0,<12.0.0\n")
        f.write("opencv-python-headless<4.12\n\n")
        
        f.write("# Data processing and scientific computing\n")
        f.write("numpy>=1.26,<2.0\n")
        f.write("pandas\n")
        f.write("scipy<1.13\n")
        f.write("openpyxl\n\n")
        
        f.write("# Deep learning\n")
        f.write("torch @ https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-linux_x86_64.whl\n")
        f.write("torchvision @ https://download.pytorch.org/whl/cpu/torchvision-0.17.2%2Bcpu-cp311-cp311-linux_x86_64.whl\n")
        f.write("ultralytics\n")
        f.write("supervision\n\n")
        
        f.write("# ML and AI tools\n")
        f.write("huggingface_hub\n")
        f.write("pycocotools>=2.0.10,<3.0.0\n\n")
        
        f.write("# Utilities\n")
        f.write("jsonschema>=4.25.0,<5.0.0\n")
        f.write("httpx\n")
        f.write("python-dotenv\n")
        f.write("dataclasses-json>=0.6.7,<0.7.0\n")
        f.write("python-multipart>=0.0.20,<0.0.21\n")

if __name__ == "__main__":
    parse_dependencies()
    print("requirements.txt updated successfully!")
EOF

# Run the script
python /tmp/parse_deps.py

# Clean up
rm /tmp/parse_deps.py

echo "requirements.txt has been updated!"