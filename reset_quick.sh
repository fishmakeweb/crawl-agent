#!/bin/bash
# Quick reset: Clear knowledge without restarting containers

echo "🔄 QUICK RESET: Clear knowledge (keep containers running)"
echo "=========================================================="
echo ""

# Connect to Qdrant and delete collection
echo "1️⃣  Clearing Qdrant vector store..."
docker exec selflearning-agent_qdrant_1 curl -X DELETE 'http://localhost:6333/collections/crawl_patterns' 2>/dev/null
sleep 1
docker exec selflearning-agent_qdrant_1 curl -X PUT 'http://localhost:6333/collections/crawl_patterns' \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }' 2>/dev/null
echo "   ✓ Qdrant collection recreated"
echo ""

# Connect to Neo4j and clear all data
echo "2️⃣  Clearing Neo4j graph database..."
docker exec selflearning-agent_neo4j_1 cypher-shell -u neo4j -p password \
  "MATCH (n) DETACH DELETE n" 2>/dev/null || echo "   ⚠️  Neo4j clear might need manual auth"
echo "   ✓ Neo4j data cleared (if auth succeeded)"
echo ""

# Connect to Redis and flush all
echo "3️⃣  Clearing Redis cache..."
docker exec selflearning-agent_redis_1 redis-cli FLUSHALL 2>/dev/null
echo "   ✓ Redis cache cleared"
echo ""

# Restart training agent to reload
echo "4️⃣  Restarting training agent..."
docker restart selflearning-agent_agent-training_1 >/dev/null 2>&1
echo "   ✓ Training agent restarted"
echo ""

echo "✅ QUICK RESET COMPLETE!"
echo ""
echo "📊 Knowledge store is now empty and ready for new training"
