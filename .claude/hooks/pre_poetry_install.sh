#!/bin/bash

# Claude Code Pre-Hook for poetry install
# This script runs before any poetry install command

# Get the command from Claude Code environment
COMMAND=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.command // empty')

# Check if this is a poetry install command
if [[ "$COMMAND" == *"poetry install"* ]]; then
    echo "🔍 Claude Code Hook: Detecting poetry install command..."
    
    # Check if .venv directory exists
    if [ -d ".venv" ]; then
        # Calculate .venv size in bytes
        VENV_SIZE_BYTES=$(du -sb .venv 2>/dev/null | cut -f1)
        
        # Convert to human readable format
        if command -v numfmt >/dev/null 2>&1; then
            VENV_SIZE_HUMAN=$(numfmt --to=iec --suffix=B $VENV_SIZE_BYTES)
        else
            # Fallback if numfmt is not available
            if [ $VENV_SIZE_BYTES -gt 1073741824 ]; then
                VENV_SIZE_HUMAN="$(echo "scale=1; $VENV_SIZE_BYTES/1073741824" | bc -l)GB"
            elif [ $VENV_SIZE_BYTES -gt 1048576 ]; then
                VENV_SIZE_HUMAN="$(echo "scale=1; $VENV_SIZE_BYTES/1048576" | bc -l)MB"
            elif [ $VENV_SIZE_BYTES -gt 1024 ]; then
                VENV_SIZE_HUMAN="$(echo "scale=1; $VENV_SIZE_BYTES/1024" | bc -l)KB"
            else
                VENV_SIZE_HUMAN="${VENV_SIZE_BYTES}B"
            fi
        fi
        
        # Store the size in a temporary file for the post-hook
        echo "$VENV_SIZE_BYTES" > .claude/.venv_size_before
        echo "📊 Before poetry install: $VENV_SIZE_HUMAN"
    else
        echo "📊 No .venv directory found - will be created during poetry install"
        echo "0" > .claude/.venv_size_before
    fi
fi