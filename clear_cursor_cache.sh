#!/bin/bash
# Script to clear Cursor's cache directories
# This will help resolve update check errors

echo "Clearing Cursor cache directories..."

# Close Cursor if running (optional - user should do this manually)
# pkill -f cursor-server

# Clear cache directories
if [ -d ~/.cursor-server/data/CachedExtensionVSIXs ]; then
    echo "Clearing extension cache..."
    rm -rf ~/.cursor-server/data/CachedExtensionVSIXs/*
fi

if [ -d ~/.cursor-server/data/CachedProfilesData ]; then
    echo "Clearing profile cache..."
    rm -rf ~/.cursor-server/data/CachedProfilesData/*
fi

# Clear old logs (keep recent ones)
if [ -d ~/.cursor-server/data/logs ]; then
    echo "Clearing old logs..."
    find ~/.cursor-server/data/logs -type f -mtime +7 -delete 2>/dev/null
fi

# Clear cursor-agent cache if it exists
if [ -d ~/.local/share/cursor-agent ]; then
    echo "Clearing cursor-agent cache..."
    # Keep the versions directory but clear any temp files
    find ~/.local/share/cursor-agent -type f -name "*.tmp" -o -name "*.log" 2>/dev/null | xargs rm -f 2>/dev/null
fi

# Clear any temp directories
if [ -d /tmp ]; then
    echo "Clearing Cursor temp files..."
    rm -rf /tmp/cursor-* 2>/dev/null
    rm -rf /tmp/vscode-* 2>/dev/null
fi

echo "Cache cleared! Please restart Cursor."
echo ""
echo "To manually check for updates:"
echo "1. Visit: https://cursor.com/en-US/downloads"
echo "2. Download the latest Linux version"
echo "3. Install it to update Cursor"

