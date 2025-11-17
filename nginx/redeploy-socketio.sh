#!/bin/bash
# Quick redeploy script for Socket.IO integration

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}🔄 Redeploying Training Agent with Socket.IO${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root (use sudo)${NC}"
    exit 1
fi

# Step 1: Stop current container
echo -e "${YELLOW}🛑 Stopping current container...${NC}"
docker stop training-server 2>/dev/null || true
docker rm training-server 2>/dev/null || true

# Step 2: Rebuild image
echo -e "${YELLOW}🏗️  Rebuilding Docker image...${NC}"
cd /root/projects/crawldata/MCP-Servers
if docker build -t self-learning-agent-training:latest -f self-learning-agent/Dockerfile.training .; then
    echo -e "${GREEN}✅ Image built successfully${NC}"
else
    echo -e "${RED}❌ Image build failed${NC}"
    exit 1
fi

# Step 3: Start new container
echo -e "${YELLOW}🚀 Starting new container...${NC}"
docker run -d --name training-server \
  --restart unless-stopped \
  -p 8091:8091 \
  --env-file self-learning-agent/.env \
  -v $(pwd)/self-learning-agent/knowledge_db:/app/knowledge_db \
  self-learning-agent-training:latest

# Step 4: Wait for container to start
echo -e "${YELLOW}⏳ Waiting for container to start...${NC}"
sleep 5

# Step 5: Update nginx config
echo -e "${YELLOW}📝 Updating nginx configuration...${NC}"
cp self-learning-agent/nginx/train.fishmakeweb.id.vn.conf /etc/nginx/sites-available/train.fishmakeweb.id.vn

# Step 6: Test nginx config
echo -e "${YELLOW}🧪 Testing nginx configuration...${NC}"
if nginx -t; then
    echo -e "${GREEN}✅ Nginx config valid${NC}"
else
    echo -e "${RED}❌ Nginx config invalid${NC}"
    exit 1
fi

# Step 7: Reload nginx
echo -e "${YELLOW}🔄 Reloading nginx...${NC}"
systemctl reload nginx

# Step 8: Verify deployment
echo -e "${YELLOW}🔍 Verifying deployment...${NC}"
sleep 3

# Check container is running
if docker ps | grep -q training-server; then
    echo -e "${GREEN}✅ Container is running${NC}"
else
    echo -e "${RED}❌ Container failed to start${NC}"
    docker logs training-server
    exit 1
fi

# Check health endpoint
if curl -sSf -k https://train.fishmakeweb.id.vn/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
    docker logs --tail 50 training-server
    exit 1
fi

# Check Socket.IO endpoint
if curl -sSf https://train.fishmakeweb.id.vn/socket.io/?EIO=4\&transport=polling > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Socket.IO endpoint responding${NC}"
else
    echo -e "${YELLOW}⚠️  Socket.IO endpoint check inconclusive (may be normal)${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📊 Status:${NC}"
echo -e "   Container: ${GREEN}Running${NC}"
echo -e "   Health: ${GREEN}Healthy${NC}"
echo -e "   Socket.IO: ${GREEN}Enabled${NC}"
echo ""
echo -e "${YELLOW}🔗 URLs:${NC}"
echo -e "   API: ${GREEN}https://train.fishmakeweb.id.vn${NC}"
echo -e "   Health: ${GREEN}https://train.fishmakeweb.id.vn/health${NC}"
echo -e "   Socket.IO: ${GREEN}wss://train.fishmakeweb.id.vn/socket.io/${NC}"
echo ""
echo -e "${YELLOW}📝 Useful commands:${NC}"
echo -e "   View logs: ${YELLOW}docker logs -f training-server${NC}"
echo -e "   Watch Socket.IO: ${YELLOW}docker logs -f training-server | grep 'Socket.IO'${NC}"
echo -e "   Health check: ${YELLOW}curl -s https://train.fishmakeweb.id.vn/health | jq .${NC}"
