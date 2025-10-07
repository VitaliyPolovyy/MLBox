#!/bin/bash
# start_dev.sh — load MLBox environment

# Move to project folder
cd "/mnt/c/My storage/Python projects/MLBox" || exit 1

# Load credentials
source "$HOME/credentials/.env.mlbox"

echo "MLBox environment loaded ✅"

# Run command or start shell
if [ $# -gt 0 ]; then
    "$@"
else
    bash
fi
