#!/bin/bash
# Complete reset of Self-Learning Agent knowledge and resources

echo "🗑️  COMPLETE RESET: Self-Learning Agent"
echo "========================================"
echo ""

# Stop containers first
echo "1️⃣  Stopping containers..."
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
docker-compose -f docker-compose.self-learning.yml down
echo "   ✓ Containers stopped"
echo ""

# Clear volumes (Qdrant, Neo4j, Redis data)
echo "2️⃣  Clearing Docker volumes..."
docker volume rm self-learning-agent_qdrant_storage 2>/dev/null || echo "   - Qdrant volume already deleted"
docker volume rm self-learning-agent_neo4j_data 2>/dev/null || echo "   - Neo4j volume already deleted"
docker volume rm self-learning-agent_neo4j_logs 2>/dev/null || echo "   - Neo4j logs already deleted"
docker volume rm self-learning-agent_redis_data 2>/dev/null || echo "   - Redis volume already deleted"
echo "   ✓ Volumes cleared"
echo ""

# Restart containers with fresh data
echo "3️⃣  Restarting containers..."
docker-compose -f docker-compose.self-learning.yml up -d
echo "   ✓ Containers restarted with fresh state"
echo ""

# Wait for services to be ready
echo "4️⃣  Waiting for services to start..."
sleep 10
echo "   ✓ Services should be ready"
echo ""

echo "✅ RESET COMPLETE!"
echo ""
echo "📊 System Status:"
echo "   - Qdrant: Fresh vector database"
echo "   - Neo4j: Fresh graph database"
echo "   - Redis: Fresh cache"
echo "   - All patterns, resources, and knowledge cleared"
echo ""
echo "🎯 You can now start training from scratch!"
