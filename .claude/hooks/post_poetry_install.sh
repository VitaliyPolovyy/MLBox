#!/bin/bash

# Claude Code Post-Hook for poetry install
# This script runs after any poetry install command

# Get the command from Claude Code environment
COMMAND=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.command // empty')

# Check if this is a poetry install command
if [[ "$COMMAND" == *"poetry install"* ]]; then
    echo "🔍 Claude Code Hook: Poetry install completed, measuring .venv size..."
    
    # Check if .venv directory exists now
    if [ -d ".venv" ]; then
        # Calculate current .venv size in bytes
        VENV_SIZE_BYTES_AFTER=$(du -sb .venv 2>/dev/null | cut -f1)
        
        # Read the before size
        if [ -f ".claude/.venv_size_before" ]; then
            VENV_SIZE_BYTES_BEFORE=$(cat .claude/.venv_size_before)
        else
            VENV_SIZE_BYTES_BEFORE=0
        fi
        
        # Calculate difference
        VENV_SIZE_DIFF=$((VENV_SIZE_BYTES_AFTER - VENV_SIZE_BYTES_BEFORE))
        
        # Convert sizes to human readable format
        format_size() {
            local size_bytes=$1
            if command -v numfmt >/dev/null 2>&1; then
                numfmt --to=iec --suffix=B $size_bytes
            else
                # Fallback if numfmt is not available
                if [ $size_bytes -gt 1073741824 ]; then
                    echo "$(echo "scale=1; $size_bytes/1073741824" | bc -l)GB"
                elif [ $size_bytes -gt 1048576 ]; then
                    echo "$(echo "scale=1; $size_bytes/1048576" | bc -l)MB"
                elif [ $size_bytes -gt 1024 ]; then
                    echo "$(echo "scale=1; $size_bytes/1024" | bc -l)KB"
                else
                    echo "${size_bytes}B"
                fi
            fi
        }
        
        VENV_SIZE_BEFORE_HUMAN=$(format_size $VENV_SIZE_BYTES_BEFORE)
        VENV_SIZE_AFTER_HUMAN=$(format_size $VENV_SIZE_BYTES_AFTER)
        
        # Format the difference
        if [ $VENV_SIZE_DIFF -gt 0 ]; then
            VENV_SIZE_DIFF_HUMAN="+$(format_size $VENV_SIZE_DIFF)"
            DIFF_COLOR="📈"
        elif [ $VENV_SIZE_DIFF -lt 0 ]; then
            VENV_SIZE_DIFF_HUMAN="-$(format_size $((0 - VENV_SIZE_DIFF)))"
            DIFF_COLOR="📉"
        else
            VENV_SIZE_DIFF_HUMAN="±0B"
            DIFF_COLOR="📊"
        fi
        
        # Display results
        echo "📊 .venv Size Report:"
        echo "   Before: $VENV_SIZE_BEFORE_HUMAN"
        echo "   After:  $VENV_SIZE_AFTER_HUMAN ($DIFF_COLOR $VENV_SIZE_DIFF_HUMAN)"
        
        # Clean up temporary file
        rm -f .claude/.venv_size_before
        
    else
        echo "⚠️  No .venv directory found after poetry install"
        rm -f .claude/.venv_size_before
    fi
fi