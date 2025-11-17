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

# Check if running from correct directory
if [ ! -f "config.py" ]; then
    echo -e "${RED}❌ Error: Must run from self-learning-agent directory${NC}"
    echo -e "${YELLOW}   cd /root/projects/crawldata/MCP-Servers/self-learning-agent${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo -e "${YELLOW}   Please start Docker and try again${NC}"
    exit 1
fi

# Check if .env.self-learning exists
if [ ! -f "../../.env.self-learning" ]; then
    echo -e "${YELLOW}⚠️  .env.self-learning not found!${NC}"
    echo -e "${YELLOW}   Creating from example...${NC}"
    cp .env.example ../../.env.self-learning
    echo -e "${RED}❌ Please edit .env.self-learning and add your GEMINI_API_KEY${NC}"
    echo -e "${YELLOW}   Then run this script again${NC}"
    exit 1
fi

# Check if GEMINI_API_KEY is set
if grep -q "your_gemini_api_key_here" ../../.env.self-learning; then
    echo -e "${RED}❌ Error: GEMINI_API_KEY not configured in .env.self-learning${NC}"
    echo -e "${YELLOW}   Please edit .env.self-learning and add your API key${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Environment configuration found${NC}"
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

# Change to project root
cd ../..

if [ "$MODE" == "down" ]; then
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    docker-compose -f docker-compose.self-learning.yml down
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
fi

if [ "$MODE" == "logs" ]; then
    echo -e "${GREEN}📋 Showing logs (Ctrl+C to exit)...${NC}"
    docker-compose -f docker-compose.self-learning.yml logs -f
    exit 0
fi

# Start services
echo -e "${GREEN}🚀 Starting Self-Learning Agent services...${NC}"
echo ""

docker-compose -f docker-compose.self-learning.yml $PROFILE up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
sleep 5

# Check service status
echo ""
echo -e "${GREEN}📊 Service Status:${NC}"
docker-compose -f docker-compose.self-learning.yml ps

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Self-Learning Agent Started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Access Points:${NC}"
echo -e "  • Training Agent API: ${YELLOW}http://localhost:8001${NC}"
echo -e "  • Training Agent Docs: ${YELLOW}http://localhost:8001/docs${NC}"
echo -e "  • Production Agent API: ${YELLOW}http://localhost:8000${NC}"
echo -e "  • Training UI: ${YELLOW}http://localhost:3001${NC}"
echo -e "  • Qdrant Dashboard: ${YELLOW}http://localhost:6333/dashboard${NC}"
echo -e "  • Neo4j Browser: ${YELLOW}http://localhost:7474${NC}"
echo ""
echo -e "${GREEN}Commands:${NC}"
echo -e "  • View logs: ${YELLOW}./start.sh --logs${NC}"
echo -e "  • Stop services: ${YELLOW}./start.sh --down${NC}"
echo -e "  • Check health: ${YELLOW}curl http://localhost:8001/health${NC}"
echo ""
echo -e "${YELLOW}📋 Monitoring logs...${NC}"
echo -e "${YELLOW}   Press Ctrl+C to detach from logs${NC}"
echo ""

# Follow logs
docker-compose -f docker-compose.self-learning.yml logs -f
