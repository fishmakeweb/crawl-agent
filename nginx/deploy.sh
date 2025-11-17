#!/bin/bash
# Deployment script for train.fishmakeweb.id.vn

set -e

echo "🚀 Deploying Training Agent to train.fishmakeweb.id.vn"

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

# Step 1: Copy nginx configuration
echo -e "${YELLOW}📝 Copying nginx configuration...${NC}"
cp /root/projects/crawldata/MCP-Servers/self-learning-agent/nginx/train.fishmakeweb.id.vn.conf /etc/nginx/sites-available/train.fishmakeweb.id.vn

# Step 2: Enable site
echo -e "${YELLOW}🔗 Enabling site...${NC}"
ln -sf /etc/nginx/sites-available/train.fishmakeweb.id.vn /etc/nginx/sites-enabled/

# Step 3: Test nginx configuration
echo -e "${YELLOW}🧪 Testing nginx configuration...${NC}"
if nginx -t; then
    echo -e "${GREEN}✅ Nginx configuration is valid${NC}"
else
    echo -e "${RED}❌ Nginx configuration test failed${NC}"
    exit 1
fi

# Step 4: Check if training server is running
echo -e "${YELLOW}🔍 Checking training server status...${NC}"
if docker ps | grep -q training-server; then
    echo -e "${GREEN}✅ Training server is running on port 8091${NC}"
else
    echo -e "${RED}⚠️  Training server is not running!${NC}"
    echo -e "${YELLOW}Starting training server...${NC}"
    
    cd /root/projects/crawldata/MCP-Servers
    docker run -d --name training-server \
        --restart unless-stopped \
        -p 8091:8091 \
        --env-file self-learning-agent/.env \
        -v $(pwd)/self-learning-agent/knowledge_db:/app/knowledge_db \
        self-learning-agent-training:latest
    
    # Wait for server to start
    echo -e "${YELLOW}⏳ Waiting for server to start...${NC}"
    sleep 5
    
    if docker ps | grep -q training-server; then
        echo -e "${GREEN}✅ Training server started successfully${NC}"
    else
        echo -e "${RED}❌ Failed to start training server${NC}"
        docker logs training-server
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

if curl -sSf -k https://train.fishmakeweb.id.vn/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo ""
    echo -e "${GREEN}🎉 Training Agent is now live at:${NC}"
    echo -e "${GREEN}   https://train.fishmakeweb.id.vn${NC}"
    echo ""
    echo -e "${YELLOW}📊 Health Check:${NC}"
    curl -s -k https://train.fishmakeweb.id.vn/health | jq .
else
    echo -e "${RED}❌ Health check failed${NC}"
    echo -e "${YELLOW}Checking nginx logs:${NC}"
    tail -20 /var/log/nginx/train.fishmakeweb.id.vn.error.log
    echo ""
    echo -e "${YELLOW}Checking docker logs:${NC}"
    docker logs --tail 20 training-server
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${YELLOW}📝 Useful commands:${NC}"
echo -e "   View nginx logs: ${YELLOW}sudo tail -f /var/log/nginx/train.fishmakeweb.id.vn.access.log${NC}"
echo -e "   View docker logs: ${YELLOW}docker logs -f training-server${NC}"
echo -e "   Check status: ${YELLOW}curl -k https://train.fishmakeweb.id.vn/health${NC}"
