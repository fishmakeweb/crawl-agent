#!/bin/bash
# Deployment script for qdrant.fishmakeweb.id.vn

set -e

echo "🚀 Deploying Qdrant Dashboard to qdrant.fishmakeweb.id.vn"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root (use sudo)${NC}"
    exit 1
fi

# Check if htpasswd file exists
if [ ! -f /etc/nginx/.htpasswd-qdrant ]; then
    echo -e "${RED}❌ Authentication file not found!${NC}"
    echo -e "${YELLOW}Please run setup-auth.sh first to create credentials${NC}"
    exit 1
fi

# Step 1: Copy nginx configuration
echo -e "${YELLOW}📝 Copying nginx configuration...${NC}"
cp /root/projects/crawldata/MCP-Servers/self-learning-agent/nginx/qdrant.fishmakeweb.id.vn.conf /etc/nginx/sites-available/qdrant.fishmakeweb.id.vn

# Step 2: Enable site
echo -e "${YELLOW}🔗 Enabling site...${NC}"
ln -sf /etc/nginx/sites-available/qdrant.fishmakeweb.id.vn /etc/nginx/sites-enabled/

# Step 3: Test nginx configuration
echo -e "${YELLOW}🧪 Testing nginx configuration...${NC}"
if nginx -t; then
    echo -e "${GREEN}✅ Nginx configuration is valid${NC}"
else
    echo -e "${RED}❌ Nginx configuration test failed${NC}"
    exit 1
fi

# Step 4: Check if Qdrant container is running
echo -e "${YELLOW}🔍 Checking Qdrant container status...${NC}"
cd /root/projects/crawldata/MCP-Servers/self-learning-agent

if docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml ps | grep -q "qdrant.*Up"; then
    echo -e "${GREEN}✅ Qdrant container is running (ports 6333:REST, 6334:gRPC)${NC}"
else
    echo -e "${RED}⚠️  Qdrant container is not running!${NC}"
    echo -e "${YELLOW}Starting Qdrant...${NC}"
    
    docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml up -d qdrant
    
    # Wait for Qdrant to start
    echo -e "${YELLOW}⏳ Waiting for Qdrant to start...${NC}"
    sleep 10
    
    if docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml ps | grep -q "qdrant.*Up"; then
        echo -e "${GREEN}✅ Qdrant started successfully${NC}"
    else
        echo -e "${RED}❌ Failed to start Qdrant${NC}"
        docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml logs qdrant --tail 50
        exit 1
    fi
fi

# Step 5: Reload nginx
echo -e "${YELLOW}🔄 Reloading nginx...${NC}"
systemctl reload nginx
echo -e "${GREEN}✅ Nginx reloaded${NC}"

# Step 6: Verify deployment
echo -e "${YELLOW}🔍 Verifying deployment...${NC}"
sleep 2

# Test REST API endpoint
if curl -sSf http://localhost:6333 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo ""
    echo -e "${GREEN}🎉 Qdrant Dashboard is now live at:${NC}"
    echo -e "${GREEN}   https://qdrant.fishmakeweb.id.vn/dashboard${NC}"
    echo -e "${GREEN}   REST API: https://qdrant.fishmakeweb.id.vn${NC}"
    echo ""
    echo -e "${YELLOW}🔐 Authentication:${NC}"
    echo -e "   - Nginx Basic Auth: As configured in .htpasswd-qdrant"
    echo ""
    echo -e "${YELLOW}⚠️  Security Notes:${NC}"
    echo -e "   - gRPC port (6334) is NOT exposed externally (internal docker network only)"
    echo -e "   - For external gRPC access, containers connect via: grpc://qdrant:6334 (internal network)"
    echo -e "   - REST API requires basic auth for all operations"
    echo ""
    echo -e "${YELLOW}📊 Qdrant Status:${NC}"
    curl -s http://localhost:6333 | jq . 2>/dev/null || echo "Qdrant is running"
else
    echo -e "${RED}❌ Health check failed${NC}"
    echo -e "${YELLOW}Checking nginx logs:${NC}"
    tail -20 /var/log/nginx/qdrant.fishmakeweb.id.vn.error.log 2>/dev/null || echo "No error logs yet"
    echo ""
    echo -e "${YELLOW}Checking docker logs:${NC}"
    docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml logs qdrant --tail 20
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${YELLOW}📝 Useful commands:${NC}"
echo -e "   View nginx logs: ${YELLOW}sudo tail -f /var/log/nginx/qdrant.fishmakeweb.id.vn.access.log${NC}"
echo -e "   View docker logs: ${YELLOW}docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml logs qdrant -f${NC}"
echo -e "   Check container: ${YELLOW}docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml ps qdrant${NC}"
echo -e "   API test: ${YELLOW}curl -u username:password https://qdrant.fishmakeweb.id.vn/collections${NC}"
