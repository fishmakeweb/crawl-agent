#!/bin/bash
# Reset all knowledge and resources

echo "🗑️  Resetting Self-Learning Agent Knowledge Store..."

BASE_URL="http://localhost:8001"

# 1. Clear Redis buffers
echo "1️⃣  Clearing Redis buffers..."
curl -X DELETE "$BASE_URL/buffers/clear" 2>/dev/null
echo ""

# 2. Clear pending commits
echo "2️⃣  Clearing pending commits..."
curl -X DELETE "$BASE_URL/pending-commits/clear" 2>/dev/null
echo ""

# 3. Clear job queue
echo "3️⃣  Clearing job queue..."
curl -X DELETE "$BASE_URL/queue/clear" 2>/dev/null
echo ""

# 4. Reset resources (versions)
echo "4️⃣  Resetting resource versions..."
curl -X POST "$BASE_URL/resources/reset" 2>/dev/null
echo ""

# 5. Clear Qdrant collections (knowledge store)
echo "5️⃣  Clearing Qdrant vector store..."
curl -X DELETE "$BASE_URL/knowledge/clear" 2>/dev/null
echo ""

# 6. Clear Neo4j graph database
echo "6️⃣  Clearing Neo4j graph database..."
curl -X DELETE "$BASE_URL/graph/clear" 2>/dev/null
echo ""

echo "✅ Reset complete!"
echo ""
echo "📊 Checking status..."
curl -s "$BASE_URL/stats" | jq '.'

