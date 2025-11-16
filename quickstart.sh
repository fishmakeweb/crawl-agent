#!/bin/bash

# Quick Start Script for Self-Learning Agent System
# This script helps you get started quickly

set -e

echo "=========================================="
echo "Self-Learning Agent - Quick Start"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f "../../.env.self-learning" ]; then
    echo "⚠️  No .env.self-learning file found!"
    echo "Creating from template..."

    if [ -f "../../.env.self-learning.example" ]; then
        cp ../../.env.self-learning.example ../../.env.self-learning
        echo "✅ Created .env.self-learning from example"
        echo ""
        echo "⚠️  IMPORTANT: Edit .env.self-learning and add your:"
        echo "   - GEMINI_API_KEY"
        echo "   - POSTGRES_PASSWORD"
        echo "   - NEO4J_PASSWORD"
        echo ""
        read -p "Press Enter after editing .env.self-learning to continue..."
    else
        echo "❌ .env.self-learning.example not found!"
        exit 1
    fi
fi

# Source environment variables
source ../../.env.self-learning

# Check Gemini API key
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your-gemini-api-key-here" ]; then
    echo "❌ GEMINI_API_KEY not set in .env.self-learning"
    echo "   Get your API key from: https://makersuite.google.com/app/apikey"
    exit 1
fi

echo "✅ Configuration loaded"
echo ""

# Ask what to start
echo "What would you like to do?"
echo ""
echo "1) Start Full System (Production + Training + UI + Infrastructure)"
echo "2) Start Infrastructure Only (Qdrant, Neo4j, Redis, PostgreSQL)"
echo "3) Start Training Mode Only (Agent + UI + Infrastructure)"
echo "4) Start Production Mode Only (Agent + Infrastructure)"
echo "5) Stop All Services"
echo "6) View Logs"
echo "7) Run Tests"
echo ""
read -p "Enter choice [1-7]: " choice

case $choice in
    1)
        echo "🚀 Starting full system..."
        cd ../..
        docker-compose -f docker-compose.self-learning.yml up -d
        echo ""
        echo "✅ Services started!"
        echo ""
        echo "Access points:"
        echo "  - Production API: http://localhost:5014"
        echo "  - Training API: http://localhost:5020"
        echo "  - Training UI: http://localhost:3001"
        echo "  - Qdrant: http://localhost:6333/dashboard"
        echo "  - Neo4j: http://localhost:7474"
        echo "  - Grafana: http://localhost:3000 (if monitoring enabled)"
        echo ""
        echo "View logs: docker-compose -f docker-compose.self-learning.yml logs -f"
        ;;

    2)
        echo "🔧 Starting infrastructure only..."
        cd ../..
        docker-compose -f docker-compose.self-learning.yml up -d qdrant neo4j redis postgres
        echo ""
        echo "✅ Infrastructure started!"
        echo ""
        echo "  - Qdrant: http://localhost:6333"
        echo "  - Neo4j: http://localhost:7474"
        echo "  - Redis: localhost:6379"
        echo "  - PostgreSQL: localhost:5432"
        ;;

    3)
        echo "🎓 Starting training mode..."
        cd ../..
        docker-compose -f docker-compose.self-learning.yml up -d agent-training training-service training-ui qdrant neo4j redis postgres
        echo ""
        echo "✅ Training mode started!"
        echo ""
        echo "  - Training API: http://localhost:5020"
        echo "  - Training UI: http://localhost:3001"
        ;;

    4)
        echo "🏭 Starting production mode..."
        cd ../..
        docker-compose -f docker-compose.self-learning.yml up -d agent-production production-crawler qdrant neo4j redis postgres
        echo ""
        echo "✅ Production mode started!"
        echo ""
        echo "  - Production API: http://localhost:5014"
        ;;

    5)
        echo "🛑 Stopping all services..."
        cd ../..
        docker-compose -f docker-compose.self-learning.yml down
        echo "✅ All services stopped"
        ;;

    6)
        echo "📋 Viewing logs (Ctrl+C to exit)..."
        cd ../..
        docker-compose -f docker-compose.self-learning.yml logs -f
        ;;

    7)
        echo "🧪 Running tests..."
        echo ""
        echo "Test Gemini Client:"
        python -c "
import asyncio
from config import Config
from gemini_client import GeminiClient

async def test():
    config = Config()
    client = GeminiClient(config.gemini)
    response = await client.generate('What is 2+2?')
    print(f'Response: {response}')
    print(f'Stats: {client.get_stats()}')

asyncio.run(test())
"
        ;;

    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
