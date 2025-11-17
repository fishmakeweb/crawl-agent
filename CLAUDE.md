# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A self-learning web crawler agent system built on Microsoft Agent-Lightning framework. The system autonomously improves its web scraping and data extraction capabilities by learning from execution results and user feedback.

**Key Characteristics:**
- **Dual-Service Architecture**: Production mode (fast, frozen resources) vs Training mode (active learning)
- **AI-Powered Learning**: Uses Google Gemini for pattern extraction and prompt generation
- **Hybrid Knowledge Store**: Vector (Qdrant) + Graph (Neo4j) + Cache (Redis) storage
- **Cost-Optimized**: Multi-model routing, batching, and caching reduce API costs by 70%

## Running the Application

### Setup

```bash
# Install dependencies
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
pip install -r requirements.txt

# Set required environment variables
export GEMINI_API_KEY="your-api-key"
export MODE="training"  # or "production"
export LIGHTNING_STORE_URL="postgresql://postgres:password@localhost:5432/lightning"
export QDRANT_HOST="localhost"
export NEO4J_URI="bolt://localhost:7687"
export REDIS_HOST="localhost"
```

### Infrastructure (Required)

```bash
# Start PostgreSQL, Qdrant, Neo4j, Redis
docker-compose -f docker-compose.self-learning.yml up -d
```

### Run Services

```bash
# Training Mode (port 8091) - Active learning enabled
python training_agent.py
# or with Uvicorn:
uvicorn training_agent:socket_app --host 0.0.0.0 --port 8091

# Production Mode (port 8000) - Frozen resources, optimized for speed
python production_agent.py
# or with Uvicorn:
uvicorn production_agent:app --host 0.0.0.0 --port 8000
```

### UI Development

```bash
cd ui
npm install
npm start      # Development server (port 3000)
npm run build  # Production build
npm test       # Run tests
```

### Docker Build

```bash
# From MCP-Servers directory
cd /root/projects/crawldata/MCP-Servers

# Build training service
docker build -t self-learning-agent-training:latest -f self-learning-agent/Dockerfile.training .

# Build production service
docker build -t self-learning-agent-production:latest -f self-learning-agent/Dockerfile.production .
```

## Architecture

### Core Components

1. **Shared Crawler Agent** (`agents/shared_crawler_agent.py`)
   - Dual-mode execution (production vs training)
   - Wraps Crawl4AI for web scraping
   - Integrates with Agent-Lightning for rollout tracking

2. **Self-Improving Algorithm** (`algorithms/self_improving_algorithm.py`)
   - **N-Rollout Learning**: Updates resources every N executions (default: 5)
   - Extracts successful patterns from high-reward executions
   - Analyzes failure patterns to avoid mistakes
   - Uses Gemini to generate improved extraction prompts

3. **Hybrid Knowledge Store** (`knowledge/hybrid_knowledge_store.py`)
   - **3-Tier Retrieval Strategy**:
     1. Redis Cache (hot patterns, 1-hour TTL)
     2. Qdrant Vector Store (semantic similarity search)
     3. Neo4j Graph Store (entity relationships, domain patterns)
   - Domain-based pattern organization
   - Semantic clustering (DBSCAN, threshold 0.85)

4. **RL Resource Controller** (`knowledge/rl_controller.py`)
   - Autonomous storage management
   - Actions: scale, prune, consolidate, archive
   - Monitors redundancy, retrieval frequency, storage limits

5. **Gemini Client** (`gemini_client.py`)
   - **Multi-Model Cost Routing**: Automatic model selection by task type
   - **3-Tier Caching**: Request → Batch → Semantic (70% API reduction)
   - **Async Batching**: Up to 8 requests → 1 API call
   - Local LLM fallback support (Ollama)

### Data Flow

```
User Request
    ↓
Training Agent (FastAPI + Socket.IO)
    ↓
Shared Crawler Agent
    ↓
Crawl4AI → Web Scraping
    ↓
Execution Result
    ↓
User Feedback (natural language)
    ↓
Gemini Interpretation (confidence scoring)
    ↓
Knowledge Store (Vector + Graph + Cache)
    ↓
Self-Improving Algorithm (every N rollouts)
    ↓
Updated Resources (prompts, configs, patterns)
    ↓
Next Crawl Uses Improved Resources
```

## Key Workflows

### N-Rollout Learning Cycle

The system does NOT update after every request. Instead, it follows an N-rollout pattern:

1. User submits crawl job (URL + description + schema)
2. Agent executes with current resources (prompts, configs, patterns)
3. User provides natural language feedback
4. Gemini interprets feedback with confidence scoring
5. **If confidence < 0.7** → System asks clarification: "Do you mean...?"
6. **After N rollouts (default 5)** → Trigger learning cycle
7. Algorithm analyzes successful patterns + failures + feedback history
8. Gemini generates improved extraction prompts
9. Resources updated (version++)
10. Subsequent jobs use improved resources

### Multi-Model Cost Routing

Different tasks use different Gemini models for cost optimization:

- **crawl** tasks: `gemini-2.0-flash` (fast, cheap: $0.15/$0.60 per 1M tokens)
- **feedback** interpretation: `gemini-2.5-flash` (reasoning: $0.30/$2.50)
- **clarification**: `learnlm-2.0-flash` (pedagogical: $0.25/$1.00)
- **prompt_gen**: `gemini-2.5-flash` (creative)
- **analysis**: `gemini-2.5-pro` in training mode only (deep analysis: $1.25/$5.00)
- **rl_decide**: `gemini-2.5-flash` (agentic planning)

Automatic fallback to lite models when rate limits exceeded.

### Pattern Consolidation

Triggered automatically by RL controller or manually via API:

1. Domain-based grouping
2. Semantic clustering (similarity threshold 0.85)
3. Frequency-based merging (high-usage patterns prioritized)
4. Redundancy reduction

### Resource Freezing (Production Deployment)

```python
# After training, freeze learned resources
freeze_resources(store=lightning_store, version=10, output_path="./frozen_resources_v10.json")

# Production service uses frozen resources (no learning overhead)
export FROZEN_RESOURCES_PATH="./frozen_resources_v10.json"
export MODE="production"
```

## Configuration

### Central Config (`config.py`)

Key settings:
- `UPDATE_FREQUENCY = 5`: Update resources every 5 rollouts
- `MAX_ROLLOUTS_PER_SESSION = 100`: Training session limit
- `PARALLEL_RUNNERS = 4`: Concurrent training workers
- `INITIAL_KNOWLEDGE_STORE_MAX_SIZE_GB = 2.0`: Storage limit (auto-adjusted by RL)
- `SIMILARITY_THRESHOLD = 0.85`: Pattern clustering threshold
- `CACHE_TTL_SECONDS = 3600`: Redis cache expiration
- `MAX_BATCH_SIZE = 8`: Gemini request batching

### Environment Variables (`.env`)

Required:
- `GEMINI_API_KEY`: Google Gemini API key
- `MODE`: "training" or "production"
- `LIGHTNING_STORE_URL`: PostgreSQL connection string
- `QDRANT_HOST`: Qdrant server host
- `NEO4J_URI`: Neo4j bolt URI
- `REDIS_HOST`: Redis server host

Optional:
- `FROZEN_RESOURCES_PATH`: Path to frozen resources JSON (production mode)
- `OLLAMA_BASE_URL`: Local LLM fallback endpoint

## API Endpoints

### Training Service (port 8091)

- `GET /health` - Health check with metrics (update_cycle, pending_rollouts, gemini_stats, knowledge_metrics)
- `POST /train-crawl` - Submit training crawl job
- `POST /feedback` - Submit user feedback
- `GET /stats` - Training statistics
- `GET /knowledge/patterns` - Learned domain patterns
- `POST /knowledge/consolidate` - Trigger pattern consolidation
- `GET /rl/policy` - RL controller policy state
- `POST /rl/trigger` - Trigger RL decision
- `WS /socket.io/` - Socket.IO real-time updates

### Production Service (port 8000)

- `GET /health` - Health check
- `POST /crawl` - Execute production crawl
- `GET /stats` - Production statistics

### Socket.IO Events

**Server → Client:**
- `connected` - Connection established
- `job_completed` - Crawl job finished
- `feedback_received` - Feedback processed
- `pong` - Response to ping

**Client → Server:**
- `ping` - Keep-alive check

## File Structure

```
/root/projects/crawldata/MCP-Servers/self-learning-agent/
├── agents/
│   └── shared_crawler_agent.py          # Core agent (production + training)
├── algorithms/
│   └── self_improving_algorithm.py      # N-rollout learning algorithm
├── knowledge/
│   ├── hybrid_knowledge_store.py        # 3-tier storage (Vector/Graph/Cache)
│   └── rl_controller.py                 # Autonomous resource management
├── ui/                                  # React TypeScript UI
│   ├── src/
│   │   ├── App.tsx                      # Main UI component
│   │   ├── components/                  # UI components
│   │   ├── services/
│   │   │   ├── api.ts                   # HTTP API client
│   │   │   └── websocket.ts             # Socket.IO client
│   │   ├── hooks/                       # React hooks
│   │   └── types/                       # TypeScript types
│   ├── package.json                     # UI dependencies
│   └── Dockerfile                       # UI container
├── nginx/                               # Nginx reverse proxy config
├── config.py                            # Central configuration
├── gemini_client.py                     # Cost-optimized Gemini client
├── training_agent.py                    # Training service (FastAPI + Socket.IO)
├── production_agent.py                  # Production service (FastAPI)
├── requirements.txt                     # Python dependencies
├── Dockerfile.training                  # Training service container
├── Dockerfile.production                # Production service container
├── .env                                 # Environment variables
├── README.md                            # Main documentation
├── DEPLOYMENT.md                        # Deployment guide (Nginx + Docker)
├── MULTI_MODEL_ROUTING.md               # Model selection strategy
├── SOCKETIO_INTEGRATION.md              # WebSocket real-time updates
└── INTEGRATION_GUIDE.py                 # Code examples for routing
```

## Deployment

### Current Production

- **Domain**: `train.fishmakeweb.id.vn`
- **Service**: Training mode on port 8091
- **Proxy**: Nginx with Cloudflare SSL certificates
- **Container**: `training-server` with persistent volume `./knowledge_db`

### Deployment Commands

```bash
# Quick deploy
cd /root/projects/crawldata/MCP-Servers/self-learning-agent/nginx
sudo ./deploy.sh

# Manual deploy
docker build -t self-learning-agent-training:latest -f Dockerfile.training .
docker run -d --name training-server --restart unless-stopped \
  -p 8091:8091 \
  --env-file .env \
  -v $(pwd)/knowledge_db:/app/knowledge_db \
  self-learning-agent-training:latest

# Configure Nginx
sudo cp nginx/train.fishmakeweb.id.vn.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/train.fishmakeweb.id.vn /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Testing & Monitoring

```bash
# Health check
curl -k https://train.fishmakeweb.id.vn/health | jq .

# Socket.IO connection test
curl -v https://train.fishmakeweb.id.vn/socket.io/?EIO=4&transport=polling

# Monitor logs
docker logs -f training-server
sudo tail -f /var/log/nginx/train.fishmakeweb.id.vn.access.log
```

## Important Concepts

### N-Rollout Pattern vs Immediate Learning

Unlike traditional online learning systems that update after every request, this system uses an N-rollout pattern:

- **Why?** Prevents overfitting to single examples, reduces API costs, ensures stable resources
- **When?** Every 5 rollouts by default (configurable via `UPDATE_FREQUENCY`)
- **How?** Batch analyzes successful patterns and failures, then generates improved prompts

### Feedback Confidence Scoring

User feedback is interpreted by Gemini with confidence scoring:

- **High confidence (≥0.7)**: Apply feedback to knowledge store
- **Low confidence (<0.7)**: Ask clarification: "Do you mean...?"
- **Benefits**: Prevents misinterpretation, improves learning accuracy

### Dual-Service Pattern

**Production Service:**
- Uses frozen resources (no learning overhead)
- Fast response times
- Predictable behavior
- Suitable for production workloads

**Training Service:**
- Active learning enabled
- Slower (analyzes patterns, updates resources)
- Experimental and evolving
- Suitable for improving the system

### Autonomous Resource Management

RL controller monitors storage and autonomously decides when to:
- **Scale**: Increase storage limits when quality patterns accumulate
- **Prune**: Remove low-frequency patterns
- **Consolidate**: Merge similar patterns
- **Archive**: Move old patterns to cold storage

No manual intervention required for storage optimization.
