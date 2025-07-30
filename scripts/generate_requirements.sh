#!/bin/bash

# Generate requirements.txt from Poetry
echo "Generating requirements.txt from Poetry..."

# Generate production requirements
poetry export --only=main --without-hashes --format=requirements.txt --output=requirements.txt

# Generate development requirements
poetry export --with=dev --without-hashes --format=requirements.txt --output=requirements-dev.txt

echo "Generated requirements.txt and requirements-dev.txt"
echo "Production dependencies: $(wc -l < requirements.txt) packages"
echo "Development dependencies: $(wc -l < requirements-dev.txt) packages"