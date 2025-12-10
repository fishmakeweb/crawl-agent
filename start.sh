#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Self-Learning Web Crawler Agent${NC}"
echo -e "${GREEN}Docker Compose Startup Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Detect and set project directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT=""
ENV_FILE=""

# Check if running from self-learning-agent directory
if [ -f "$SCRIPT_DIR/config.py" ]; then
    PROJECT_ROOT="$SCRIPT_DIR/../.."
    ENV_FILE_LOCATIONS=("$SCRIPT_DIR/.env.self-learning" "$PROJECT_ROOT/.env.self-learning")
elif [ -f "$SCRIPT_DIR/MCP-Servers/self-learning-agent/config.py" ]; then
    # Running from project root
    PROJECT_ROOT="$SCRIPT_DIR"
    ENV_FILE_LOCATIONS=("$PROJECT_ROOT/MCP-Servers/self-learning-agent/.env.self-learning" "$PROJECT_ROOT/.env.self-learning")
else
    echo -e "${RED}❌ Error: Must run from self-learning-agent directory or project root${NC}"
    echo -e "${YELLOW}   cd /root/projects/crawldata/MCP-Servers/self-learning-agent${NC}"
    echo -e "${YELLOW}   OR${NC}"
    echo -e "${YELLOW}   cd /root/projects/crawldata${NC}"
    exit 1
fi

# Make paths absolute
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo -e "${YELLOW}   Please start Docker and try again${NC}"
    exit 1
fi

# Find .env.self-learning file
for location in "${ENV_FILE_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
        ENV_FILE="$location"
        break
    fi
done

# Check if .env.self-learning exists
if [ -z "$ENV_FILE" ]; then
    echo -e "${YELLOW}⚠️  .env.self-learning not found!${NC}"
    echo -e "${YELLOW}   Creating from example...${NC}"

    # Try to find .env.example
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        ENV_EXAMPLE="$SCRIPT_DIR/.env.example"
        ENV_TARGET="$SCRIPT_DIR/.env.self-learning"
    elif [ -f "$PROJECT_ROOT/MCP-Servers/self-learning-agent/.env.example" ]; then
        ENV_EXAMPLE="$PROJECT_ROOT/MCP-Servers/self-learning-agent/.env.example"
        ENV_TARGET="$PROJECT_ROOT/MCP-Servers/self-learning-agent/.env.self-learning"
    else
        echo -e "${RED}❌ Error: .env.example not found${NC}"
        exit 1
    fi

    cp "$ENV_EXAMPLE" "$ENV_TARGET"
    echo -e "${RED}❌ Please edit $ENV_TARGET and add your GEMINI_API_KEY${NC}"
    echo -e "${YELLOW}   Then run this script again${NC}"
    exit 1
fi

# Check if GEMINI_API_KEY is set
if grep -q "your_gemini_api_key_here" "$ENV_FILE"; then
    echo -e "${RED}❌ Error: GEMINI_API_KEY not configured in $ENV_FILE${NC}"
    echo -e "${YELLOW}   Please edit the file and add your API key${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Environment configuration found: $ENV_FILE${NC}"
echo ""

# Parse command line arguments
PROFILE=""
MODE="up"

while [[ $# -gt 0 ]]; do
    case $1 in
        --with-local-llm)
            PROFILE="--profile with-local-llm"
            echo -e "${GREEN}🤖 Including Local LLM (Ollama)${NC}"
            shift
            ;;
        --with-monitoring)
            PROFILE="--profile with-monitoring"
            echo -e "${GREEN}📊 Including Monitoring (Grafana + Prometheus)${NC}"
            shift
            ;;
        --down)
            MODE="down"
            shift
            ;;
        --logs)
            MODE="logs"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./start.sh [--with-local-llm] [--with-monitoring] [--down] [--logs]"
            exit 1
            ;;
    esac
done

# Set Docker Compose file path
COMPOSE_FILE="$PROJECT_ROOT/MCP-Servers/self-learning-agent/docker-compose.self-learning.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo -e "${RED}❌ Error: Docker Compose file not found at $COMPOSE_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker Compose file: $COMPOSE_FILE${NC}"
echo -e "${GREEN}✅ Working directory: $PROJECT_ROOT${NC}"
echo ""

# Check for port conflicts
echo -e "${YELLOW}🔍 Checking for port conflicts...${NC}"
PORTS_IN_USE=""
if lsof -Pi :8004 -sTCP:LISTEN -t >/dev/null 2>&1; then
    PORTS_IN_USE="${PORTS_IN_USE}8004 "
fi
if lsof -Pi :8091 -sTCP:LISTEN -t >/dev/null 2>&1; then
    PORTS_IN_USE="${PORTS_IN_USE}8091 "
fi

if [ -n "$PORTS_IN_USE" ]; then
    echo -e "${YELLOW}⚠️  Warning: The following ports are already in use: ${PORTS_IN_USE}${NC}"
    echo -e "${YELLOW}   This may cause conflicts with Docker Compose services${NC}"
    echo -e "${YELLOW}   Consider stopping other services or changing ports in docker-compose.yml${NC}"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted by user${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ All required ports are available${NC}"
fi
echo ""

# Change to project root for Docker Compose
cd "$PROJECT_ROOT"

if [ "$MODE" == "down" ]; then
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    docker-compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
fi

if [ "$MODE" == "logs" ]; then
    echo -e "${GREEN}📋 Showing logs (Ctrl+C to exit)...${NC}"
    docker-compose -f "$COMPOSE_FILE" logs -f
    exit 0
fi

# Start services
echo -e "${GREEN}🚀 Starting Self-Learning Agent services...${NC}"
echo ""

docker-compose -f "$COMPOSE_FILE" $PROFILE up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
sleep 5

# Check service status
echo ""
echo -e "${GREEN}📊 Service Status:${NC}"
docker-compose -f "$COMPOSE_FILE" ps

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Self-Learning Agent Started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}🌐 Access Points:${NC}"
echo -e "  ${YELLOW}Training Agent:${NC}"
echo -e "    • API: ${YELLOW}http://localhost:8091${NC}"
echo -e "    • Docs: ${YELLOW}http://localhost:8091/docs${NC}"
echo -e "    • Health: ${YELLOW}http://localhost:8091/health${NC}"
echo ""
echo -e "  ${YELLOW}Production Agent:${NC}"
echo -e "    • API: ${YELLOW}http://localhost:8004${NC}"
echo -e "    • Docs: ${YELLOW}http://localhost:8004/docs${NC}"
echo ""
echo -e "  ${YELLOW}Infrastructure:${NC}"
echo -e "    • Qdrant (Vector): ${YELLOW}http://localhost:6333/dashboard${NC}"
echo -e "    • Neo4j (Graph): ${YELLOW}http://localhost:7474${NC} (user: neo4j, pass: password)"
echo -e "    • PostgreSQL: ${YELLOW}localhost:5432${NC} (user: postgres, pass: password)"
echo -e "    • Redis: ${YELLOW}localhost:6379${NC}"
echo ""
echo -e "${GREEN}📋 Commands:${NC}"
echo -e "  • View logs: ${YELLOW}cd $SCRIPT_DIR && ./start.sh --logs${NC}"
echo -e "  • Stop all services: ${YELLOW}cd $SCRIPT_DIR && ./start.sh --down${NC}"
echo -e "  • With Local LLM: ${YELLOW}cd $SCRIPT_DIR && ./start.sh --with-local-llm${NC}"
echo -e "  • With Monitoring: ${YELLOW}cd $SCRIPT_DIR && ./start.sh --with-monitoring${NC}"
echo ""
echo -e "${YELLOW}📝 Notes:${NC}"
echo -e "  • Training agent uses port 8091 (consistent across Docker Compose and production)"
echo -e "  • If you see port conflicts, stop other services first"
echo -e "  • For production deployment, use direct Docker run (see nginx/deploy.sh)"
echo ""
echo -e "${YELLOW}📋 Monitoring logs...${NC}"
echo -e "${YELLOW}   Press Ctrl+C to detach from logs${NC}"
echo ""

# Follow logs
docker-compose -f "$COMPOSE_FILE" logs -f
