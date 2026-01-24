# 🗑️ Reset Knowledge Store Guide

## Available Reset Options

### 🔴 **Option 1: Complete Reset** (Recommended for full clean start)

Xóa toàn bộ data và restart containers:

```bash
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
./reset_all.sh
```

**What it does:**
- ✅ Stop all containers
- ✅ Delete Docker volumes (Qdrant, Neo4j, Redis)
- ✅ Restart containers with fresh databases
- ✅ All patterns, resources, knowledge cleared

**When to use:**
- Start training from scratch
- Clear corrupted data
- Major version upgrade

---

### 🟡 **Option 2: Quick Reset** (Keep containers running)

Clear knowledge mà không restart containers:

```bash
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
./reset_quick.sh
```

**What it does:**
- ✅ Clear Qdrant collections
- ✅ Clear Neo4j graph data
- ✅ Flush Redis cache
- ✅ Restart training agent only
- ⚠️ Containers keep running

**When to use:**
- Quick knowledge reset during development
- Don't want to wait for container startup
- Testing new patterns

---

### 🟢 **Option 3: API Reset** (Selective reset - if endpoints exist)

```bash
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
./reset_knowledge.sh
```

**Note:** This requires API endpoints to be implemented. Currently most endpoints don't exist yet.

---

## Manual Reset Commands

### Reset Qdrant (Vector Database)

```bash
# Delete collection
docker exec selflearning-agent_qdrant_1 curl -X DELETE 'http://localhost:6333/collections/crawl_patterns'

# Recreate collection
docker exec selflearning-agent_qdrant_1 curl -X PUT 'http://localhost:6333/collections/crawl_patterns' \
  -H 'Content-Type: application/json' \
  -d '{"vectors": {"size": 768, "distance": "Cosine"}}'
```

### Reset Neo4j (Graph Database)

```bash
# Delete all nodes and relationships
docker exec selflearning-agent_neo4j_1 cypher-shell -u neo4j -p password \
  "MATCH (n) DETACH DELETE n"

# Verify empty
docker exec selflearning-agent_neo4j_1 cypher-shell -u neo4j -p password \
  "MATCH (n) RETURN count(n)"
```

### Reset Redis (Cache)

```bash
# Flush all data
docker exec selflearning-agent_redis_1 redis-cli FLUSHALL

# Verify empty
docker exec selflearning-agent_redis_1 redis-cli DBSIZE
```

---

## Verify Reset

After reset, check stats:

```bash
# Via API
curl http://localhost:8001/stats | jq '.'

# Expected output:
# {
#   "total_patterns": 0,
#   "pending_commits": 0,
#   "buffer_size": 0
# }
```

Or check directly:

```bash
# Qdrant count
docker exec selflearning-agent_qdrant_1 curl -s 'http://localhost:6333/collections/crawl_patterns' | jq '.result.points_count'

# Neo4j count
docker exec selflearning-agent_neo4j_1 cypher-shell -u neo4j -p password \
  "MATCH (n) RETURN count(n) as total"

# Redis count
docker exec selflearning-agent_redis_1 redis-cli DBSIZE
```

---

## ⚠️ Important Notes

1. **Backup before reset**: If you have valuable patterns, export them first
2. **Training will restart**: Agent needs to learn everything from scratch
3. **Resources cleared**: All learned prompts, configs, domain patterns gone
4. **Cannot undo**: Once reset, data is permanently deleted

---

## Export Knowledge (Before Reset)

If you want to save patterns before reset:

```python
# Export patterns from Qdrant
import asyncio
from knowledge.hybrid_knowledge_store import HybridKnowledgeStore

async def export_patterns():
    store = HybridKnowledgeStore(...)
    patterns = await store.get_all_patterns()
    
    import json
    with open('patterns_backup.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"Exported {len(patterns)} patterns")

asyncio.run(export_patterns())
```

---

## Quick Reference

| Command | Impact | Downtime | Speed |
|---------|--------|----------|-------|
| `./reset_all.sh` | **Complete** | ~30s | Slow |
| `./reset_quick.sh` | **Knowledge only** | ~5s | Fast |
| Manual commands | **Selective** | Minimal | Fastest |

**Recommendation:** Use `reset_all.sh` for clean start, `reset_quick.sh` for development.
