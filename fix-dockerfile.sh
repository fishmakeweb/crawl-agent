#!/bin/bash
# Quick fix for Dockerfile.training escaped quotes issue

set -e

cd /root/projects/crawldata/MCP-Servers/self-learning-agent

echo "🔧 Fixing Dockerfile.training..."

# Backup original
cp Dockerfile.training Dockerfile.training.backup

# Replace with fixed version
mv Dockerfile.training.fixed Dockerfile.training

echo "✅ Dockerfile.training fixed"
echo ""
echo "Now run:"
echo "  cd /root/projects/crawldata/MCP-Servers/self-learning-agent/nginx"
echo "  sudo ./redeploy-socketio.sh"
