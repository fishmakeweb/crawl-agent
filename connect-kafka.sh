#!/bin/bash
# Connect Aspire's Kafka container to selflearning-network
# This allows agents to access Kafka directly via Docker DNS

set -e  # Exit on error

KAFKA_CONTAINER=$(docker ps --filter "name=kafka" --format "{{.Names}}" | head -1)

if [ -z "$KAFKA_CONTAINER" ]; then
    echo "❌ Kafka container not found. Make sure Aspire is running."
    exit 1
fi

echo "📡 Found Kafka container: $KAFKA_CONTAINER"

# Check if selflearning-network exists
if ! docker network inspect selflearning-network &>/dev/null; then
    echo "⚠️  Network 'selflearning-network' not found. Creating..."
    docker network create selflearning-network
    echo "✅ Network created"
fi

# Check if already connected
if docker network inspect selflearning-network | grep -q "$KAFKA_CONTAINER"; then
    echo "✅ Kafka is already connected to selflearning-network"
else
    echo "🔌 Connecting Kafka to selflearning-network..."
    docker network connect selflearning-network "$KAFKA_CONTAINER"
    echo "✅ Kafka connected successfully"
fi

# Get Kafka IP in selflearning-network (for verification)
KAFKA_IP=$(docker inspect "$KAFKA_CONTAINER" | grep -A 20 "selflearning-network" | grep "IPAddress" | head -1 | awk -F'"' '{print $4}')
echo ""
echo "📍 Kafka container details:"
echo "   Name: $KAFKA_CONTAINER"
echo "   IP in selflearning-network: $KAFKA_IP"
echo "   DNS name: $KAFKA_CONTAINER (auto-resolved by Docker)"
echo ""
echo "🔄 Updating .env with dynamic Kafka container name..."

# Update .env.self-learning with current Kafka container name
if [ -f .env.self-learning ]; then
    # Check if KAFKA_CONTAINER_NAME already exists
    if grep -q "^KAFKA_CONTAINER_NAME=" .env.self-learning; then
        # Update existing line
        sed -i "s/^KAFKA_CONTAINER_NAME=.*/KAFKA_CONTAINER_NAME=$KAFKA_CONTAINER/" .env.self-learning
    else
        # Add new line after KAFKA_BRIDGE section header
        sed -i "/^# KAFKA BRIDGE/a KAFKA_CONTAINER_NAME=$KAFKA_CONTAINER" .env.self-learning
    fi
    echo "✅ Updated KAFKA_CONTAINER_NAME=$KAFKA_CONTAINER in .env.self-learning"
else
    echo "⚠️  .env.self-learning not found. Please set KAFKA_CONTAINER_NAME=$KAFKA_CONTAINER manually"
fi

echo ""
echo "✅ Setup complete! Agents will connect to: $KAFKA_CONTAINER:9092"
echo ""
echo "Next steps:"
echo "  1. Start agents: docker-compose -f docker-compose.self-learning.yml --env-file .env.self-learning up -d"
echo "  2. Verify: docker logs selflearning-agent_agent-production_1 | grep Kafka"

