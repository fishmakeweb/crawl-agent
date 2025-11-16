```markdown
# Self-Learning Web Crawler Agent

A self-improving web crawler system built on **Microsoft Agent-Lightning** that autonomously learns from execution results and user feedback.

---

## 🎯 Features

- **Dual-Service Architecture**: Production (stable) + Training (learning) services  
- **Hybrid Knowledge Store**: Vector (`Qdrant`) + Graph (`Neo4j`) + Cache (`Redis`)  
- **RL-Based Resource Management**: Autonomous controller manages storage limits  
- **Cost-Optimized Gemini**: Multi-tier caching, batching, local LLM fallback  
- **Feedback Validation**: "Do you mean...?" clarification for ambiguous feedback  
- **N-Rollout Learning**: Updates resources every N rollouts (default: 5)

---

## 📁 Project Structure

```
self-learning-agent/
├── config.py                       # Configuration management
├── gemini_client.py                # Optimized Gemini client
├── agents/
│   └── shared_crawler_agent.py    # Core agent (production + training)
├── algorithms/
│   └── self_improving_algorithm.py # Learning algorithm
├── knowledge/
│   ├── hybrid_knowledge_store.py   # 3-tier storage
│   └── rl_controller.py            # Autonomous resource manager
├── training/
│   └── trainer_service.py          # Training orchestration
└── ui/
    └── src/                        # React training UI
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export MODE="training"  # or "production"
export LIGHTNING_STORE_URL="postgresql://postgres:password@localhost:5432/lightning"
export QDRANT_HOST="localhost"
export NEO4J_URI="bolt://localhost:7687"
export REDIS_HOST="localhost"
```

### 3. Start Infrastructure (Docker Compose)

```bash
docker-compose -f docker-compose.self-learning.yml up -d
```

> Starts: **Qdrant**, **Neo4j**, **Redis**, **PostgreSQL**, and optional **Ollama**

### 4. Run the Agent

#### Training Mode

```python
from config import Config
from gemini_client import GeminiClient
from agents.shared_crawler_agent import SharedCrawlerAgent

config = Config()
gemini = GeminiClient(config.gemini)
agent = SharedCrawlerAgent(gemini, mode="training")

task = {
    "url": "https://example.com",
    "user_description": "Extract product names and prices",
    "extraction_schema": {
        "required": ["product_name", "price"]
    }
}

result = await agent.execute_crawl(task)
```

#### Production Mode

```python
agent = SharedCrawlerAgent(gemini, mode="production")
# Uses frozen resources, no learning
```

---

## 🏗️ Architecture

### Dual-Service Model

```
┌─────────────────────────────────────┐
│      Shared Agent Core              │
│   (SharedCrawlerAgent + Crawl4AI)   │
└─────────────────────────────────────┘
         ↓                    ↓
  ┌─────────────┐      ┌─────────────┐
  │ Production  │      │  Training   │
  │  Service    │      │  Service    │
  │ • Frozen    │      │ • Active    │
  │ • Fast      │      │ • Learning  │
  │ • Public    │      │ • Feedback  │
  └─────────────┘      └─────────────┘
                             ↓
                   ┌─────────────────┐
                   │ Hybrid Store    │
                   │ • Vector        │
                   │ • Graph         │
                   │ • Cache         │
                   └─────────────────┘
```

### Learning Flow

1. User submits crawl job  
2. Agent executes with current resources  
3. User provides natural language feedback  
4. Gemini interprets feedback (confidence scoring)  
5. If confidence < 0.7 → ask clarification  
6. After N rollouts → learning cycle triggered  
7. Algorithm analyzes feedback + patterns  
8. Gemini generates improved prompts  
9. Resources updated (version++)  
10. Next job uses improved resources

---

## 🧠 Hybrid Knowledge Store

**3-Tier Retrieval:**
1. **Cache (Redis)**: Hot patterns, 1-hour TTL  
2. **Vector (Qdrant)**: Semantic similarity search  
3. **Graph (Neo4j)**: Relationship reasoning  

**Consolidation Strategy:**
- Domain grouping  
- Semantic clustering (DBSCAN, threshold: 0.85)  
- Frequency-based merging  

---

## 🤖 RL Resource Controller

**Autonomous Management**  
Monitors: storage, redundancy, retrieval frequency  
Decides: scale, prune, consolidate, archive  
Learns: policy updates based on outcomes

**Actions:**
- `no_action`  
- `increase_limits`  
- `decrease_limits`  
- `run_pruning`  
- `run_consolidation`  
- `run_summarization`  
- `archive_old`

---

## 💸 Gemini Cost Optimization

**Multi-Tier Strategy:**
1. **TTL Cache** (1 hour): ~70% API reduction  
2. **Local LLM Fallback** (Ollama): Large tasks (>5000 chars)  
3. **Async Batching**: Up to 8 requests → 1 call  
4. **Direct API**: Only when needed  

**Cost Tracking:**
- Usage statistics  
- Estimated cost per session  
- Savings from cache/batching

---

## ⚙️ Configuration

### Training Config

```python
UPDATE_FREQUENCY = 5
MAX_ROLLOUTS_PER_SESSION = 100
PARALLEL_RUNNERS = 4

# Auto-adjusted by RL controller
INITIAL_KNOWLEDGE_STORE_MAX_SIZE_GB = 2.0
INITIAL_PATTERN_RETENTION_DAYS = 30
INITIAL_MIN_PATTERN_FREQUENCY = 3
```

### Gemini Config

```python
MODEL = "gemini-2.0-flash-exp"
CACHE_ENABLED = True
CACHE_TTL_SECONDS = 3600
BATCHING_ENABLED = True
MAX_BATCH_SIZE = 8
LOCAL_LLM_ENABLED = False
```

---

## 🧪 Testing

### Test Gemini Client

```python
response = await client.generate("What is web scraping?")
interpretation = await client.interpret_feedback(
    "The prices are wrong and missing product descriptions",
    context={"url": "example.com", "fields": {"price", "name"}}
)
print(client.get_stats())
```

### Test Knowledge Store

```python
await store.store_pattern({...})
results = await store.retrieve_patterns({
    "domain": "example.com",
    "intent": "extract products"
})
```

### Test RL Controller

```python
action_name, params = await rl_controller.decide_action(metrics)
reward = await rl_controller.execute_action(action_name, params, store)
await rl_controller.learn_from_outcome(action_name, reward)
```

---

## 🔧 Advanced Usage

### Custom Learning Algorithm

```python
from agentlightning import Algorithm

class MyCustomAlgorithm(Algorithm):
    def run(self, train_dataset, val_dataset):
        store = self.get_store()
        # Custom logic
```

### Resource Freezing (Production)

```python
freeze_resources(store=lightning_store, version=10, output_path="./frozen_resources_v10.json")

# In production
export FROZEN_RESOURCES_PATH="./frozen_resources_v10.json"
export MODE="production"
```

---

## 📈 Monitoring

| Component          | Key Metrics |
|--------------------|-----------|
| **Knowledge Store** | `vector_size_mb`, `graph_nodes`, `cache_hit_rate`, `total_patterns`, `pattern_redundancy` |
| **Gemini Client**   | `gemini_calls`, `cache_hits`, `local_llm_calls`, `batched_requests`, `estimated_cost_usd`, `estimated_savings_usd` |
| **RL Controller**   | Decision history, action outcomes, policy evolution |

---

---

## 📄 License

Part of the **CrawlData microservices platform**.

---
```