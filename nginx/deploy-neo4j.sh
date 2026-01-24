#!/bin/bash
# Deployment script for neo4j.fishmakeweb.id.vn

set -e

echo "🚀 Deploying Neo4j Browser to neo4j.fishmakeweb.id.vn"

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
if [ ! -f /etc/nginx/.htpasswd-neo4j ]; then
    echo -e "${RED}❌ Authentication file not found!${NC}"
    echo -e "${YELLOW}Please run setup-auth.sh first to create credentials${NC}"
    exit 1
fi

# Step 1: Copy nginx configuration
echo -e "${YELLOW}📝 Copying nginx configuration...${NC}"
cp /root/projects/crawldata/MCP-Servers/self-learning-agent/nginx/neo4j.fishmakeweb.id.vn.conf /etc/nginx/sites-available/neo4j.fishmakeweb.id.vn

# Step 2: Enable site
echo -e "${YELLOW}🔗 Enabling site...${NC}"
ln -sf /etc/nginx/sites-available/neo4j.fishmakeweb.id.vn /etc/nginx/sites-enabled/

# Step 3: Test nginx configuration
echo -e "${YELLOW}🧪 Testing nginx configuration...${NC}"
if nginx -t; then
    echo -e "${GREEN}✅ Nginx configuration is valid${NC}"
else
    echo -e "${RED}❌ Nginx configuration test failed${NC}"
    exit 1
fi

# Step 4: Check if Neo4j container is running
echo -e "${YELLOW}🔍 Checking Neo4j container status...${NC}"
cd /root/projects/crawldata/MCP-Servers/self-learning-agent

if docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml ps | grep -q "neo4j.*Up"; then
    echo -e "${GREEN}✅ Neo4j container is running (ports 7474:HTTP, 7687:Bolt)${NC}"
else
    echo -e "${RED}⚠️  Neo4j container is not running!${NC}"
    echo -e "${YELLOW}Starting Neo4j...${NC}"
    
    docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml up -d neo4j
    
    # Wait for Neo4j to start
    echo -e "${YELLOW}⏳ Waiting for Neo4j to start (this may take 30-60 seconds)...${NC}"
    sleep 30
    
    if docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml ps | grep -q "neo4j.*Up"; then
        echo -e "${GREEN}✅ Neo4j started successfully${NC}"
    else
        echo -e "${RED}❌ Failed to start Neo4j${NC}"
        docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml logs neo4j --tail 50
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

# Test HTTP endpoint (basic auth required for browser, but /db/manage/server/version doesn't need it)
if curl -sSf -k http://localhost:7474 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    echo ""
    echo -e "${GREEN}🎉 Neo4j Browser is now live at:${NC}"
    echo -e "${GREEN}   https://neo4j.fishmakeweb.id.vn${NC}"
    echo ""
    echo -e "${YELLOW}🔐 Authentication:${NC}"
    echo -e "   - Nginx Basic Auth: As configured in .htpasswd-neo4j"
    echo -e "   - Neo4j Login: neo4j / \${NEO4J_PASSWORD} (from .env.self-learning)"
    echo ""
    echo -e "${YELLOW}⚠️  Security Notes:${NC}"
    echo -e "   - Bolt protocol (7687) is NOT exposed externally (internal docker network only)"
    echo -e "   - For external Bolt access, use SSH tunnel: ssh -L 7687:localhost:7687 user@server"
    echo -e "   - Containers connect via: bolt://neo4j:7687 (internal network)"
else
    echo -e "${RED}❌ Health check failed${NC}"
    echo -e "${YELLOW}Checking nginx logs:${NC}"
    tail -20 /var/log/nginx/neo4j.fishmakeweb.id.vn.error.log 2>/dev/null || echo "No error logs yet"
    echo ""
    echo -e "${YELLOW}Checking docker logs:${NC}"
    docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml logs neo4j --tail 20
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${YELLOW}📝 Useful commands:${NC}"
echo -e "   View nginx logs: ${YELLOW}sudo tail -f /var/log/nginx/neo4j.fishmakeweb.id.vn.access.log${NC}"
echo -e "   View docker logs: ${YELLOW}docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml logs neo4j -f${NC}"
echo -e "   Check container: ${YELLOW}docker compose --env-file .env.self-learning -f docker-compose.self-learning.yml ps neo4j${NC}"
