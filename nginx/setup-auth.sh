#!/bin/bash
# Setup authentication for Neo4j and Qdrant web interfaces

set -e

echo "🔐 Setting up Basic Authentication for Knowledge Store Services"

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

# Check if apache2-utils is installed (provides htpasswd)
if ! command -v htpasswd &> /dev/null; then
    echo -e "${YELLOW}📦 Installing apache2-utils for htpasswd...${NC}"
    apt-get update && apt-get install -y apache2-utils
fi

echo ""
echo -e "${YELLOW}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║        Neo4j Browser Authentication Setup                ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Neo4j authentication
if [ -f /etc/nginx/.htpasswd-neo4j ]; then
    echo -e "${YELLOW}⚠️  /etc/nginx/.htpasswd-neo4j already exists${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Skipping Neo4j authentication setup${NC}"
    else
        rm /etc/nginx/.htpasswd-neo4j
        echo -e "${YELLOW}Enter username for Neo4j Browser access:${NC}"
        read -p "Username: " NEO4J_USERNAME
        echo -e "${YELLOW}Enter password for Neo4j Browser access:${NC}"
        htpasswd -c /etc/nginx/.htpasswd-neo4j "$NEO4J_USERNAME"
        chmod 644 /etc/nginx/.htpasswd-neo4j
        echo -e "${GREEN}✅ Neo4j authentication configured${NC}"
    fi
else
    echo -e "${YELLOW}Enter username for Neo4j Browser access:${NC}"
    read -p "Username: " NEO4J_USERNAME
    echo -e "${YELLOW}Enter password for Neo4j Browser access:${NC}"
    htpasswd -c /etc/nginx/.htpasswd-neo4j "$NEO4J_USERNAME"
    chmod 644 /etc/nginx/.htpasswd-neo4j
    echo -e "${GREEN}✅ Neo4j authentication configured${NC}"
fi

echo ""
echo -e "${YELLOW}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║       Qdrant Dashboard Authentication Setup              ║${NC}"
echo -e "${YELLOW}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Qdrant authentication
if [ -f /etc/nginx/.htpasswd-qdrant ]; then
    echo -e "${YELLOW}⚠️  /etc/nginx/.htpasswd-qdrant already exists${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Skipping Qdrant authentication setup${NC}"
    else
        rm /etc/nginx/.htpasswd-qdrant
        echo -e "${YELLOW}Enter username for Qdrant Dashboard access:${NC}"
        read -p "Username: " QDRANT_USERNAME
        echo -e "${YELLOW}Enter password for Qdrant Dashboard access:${NC}"
        htpasswd -c /etc/nginx/.htpasswd-qdrant "$QDRANT_USERNAME"
        chmod 644 /etc/nginx/.htpasswd-qdrant
        echo -e "${GREEN}✅ Qdrant authentication configured${NC}"
    fi
else
    echo -e "${YELLOW}Enter username for Qdrant Dashboard access:${NC}"
    read -p "Username: " QDRANT_USERNAME
    echo -e "${YELLOW}Enter password for Qdrant Dashboard access:${NC}"
    htpasswd -c /etc/nginx/.htpasswd-qdrant "$QDRANT_USERNAME"
    chmod 644 /etc/nginx/.htpasswd-qdrant
    echo -e "${GREEN}✅ Qdrant authentication configured${NC}"
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           Authentication Setup Complete!                 ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}📝 Summary:${NC}"
echo -e "   Neo4j htpasswd: ${GREEN}/etc/nginx/.htpasswd-neo4j${NC}"
echo -e "   Qdrant htpasswd: ${GREEN}/etc/nginx/.htpasswd-qdrant${NC}"
echo ""

echo -e "${YELLOW}🔒 Security Recommendations:${NC}"
echo ""
echo -e "${YELLOW}1. Neo4j Database Authentication:${NC}"
echo -e "   - Ensure NEO4J_PASSWORD is set in .env.self-learning"
echo -e "   - After nginx basic auth, Neo4j will prompt for database credentials"
echo -e "   - Default username: ${GREEN}neo4j${NC}"
echo -e "   - Password: As set in ${GREEN}NEO4J_PASSWORD${NC} environment variable"
echo ""

echo -e "${YELLOW}2. Change Neo4j password if using defaults:${NC}"
echo -e "   ${GREEN}# Check current password in .env.self-learning${NC}"
echo -e "   ${GREEN}grep NEO4J_PASSWORD /root/projects/crawldata/MCP-Servers/self-learning-agent/.env.self-learning${NC}"
echo ""

echo -e "${YELLOW}3. To add more users to htpasswd files:${NC}"
echo -e "   ${GREEN}# Neo4j (note: -c flag creates new file, omit it to append)${NC}"
echo -e "   ${GREEN}sudo htpasswd /etc/nginx/.htpasswd-neo4j another-user${NC}"
echo ""
echo -e "   ${GREEN}# Qdrant${NC}"
echo -e "   ${GREEN}sudo htpasswd /etc/nginx/.htpasswd-qdrant another-user${NC}"
echo ""

echo -e "${YELLOW}4. To remove a user:${NC}"
echo -e "   ${GREEN}sudo htpasswd -D /etc/nginx/.htpasswd-neo4j username${NC}"
echo ""

echo -e "${GREEN}✅ Next steps:${NC}"
echo -e "   1. Run: ${GREEN}sudo ./nginx/deploy-neo4j.sh${NC}"
echo -e "   2. Run: ${GREEN}sudo ./nginx/deploy-qdrant.sh${NC}"
echo -e "   3. Configure DNS A records in Cloudflare"
echo -e "   4. Access services at:"
echo -e "      - ${GREEN}https://neo4j.fishmakeweb.id.vn${NC}"
echo -e "      - ${GREEN}https://qdrant.fishmakeweb.id.vn/dashboard${NC}"
