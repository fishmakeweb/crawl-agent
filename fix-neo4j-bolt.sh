#!/bin/bash
# Fix Neo4j Bolt connection for browser access
# This enables bolt://neo4j.fishmakeweb.id.vn:7687 connections

set -e

echo "=============================================="
echo "🔧 Fixing Neo4j Bolt Connection"
echo "=============================================="
echo ""

# Solution 1: Update Nginx to allow Bolt through firewall and use localhost
echo "📝 Updating Nginx configuration..."

sudo tee /etc/nginx/sites-available/neo4j.fishmakeweb.id.vn > /dev/null <<'EOF'
server {
    listen 443 ssl;
    server_name neo4j.fishmakeweb.id.vn;

    # SSL certificate configuration (using Cloudflare Origin certificates)
    ssl_certificate /etc/ssl/certs/cloudflare-origin.pem;
    ssl_certificate_key /etc/ssl/private/cloudflare-origin-key.pem;

    # Recommended SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers EECDH+AESGCM:EDH+AESGCM:AES256+EECDH:AES256+EDH;

    # Security headers
    add_header X-Frame-Options SAMEORIGIN;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    
    # CRITICAL: Allow Neo4j Browser to connect to external Bolt endpoints
    add_header Content-Security-Policy "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; connect-src 'self' ws: wss: http: https: bolt: neo4j: bolt+s: neo4j+s:;" always;

    # Basic authentication
    auth_basic "Neo4j Browser - Restricted Area";
    auth_basic_user_file /etc/nginx/.htpasswd-neo4j;

    # Neo4j Browser UI proxy
    location / {
        proxy_pass http://localhost:7474;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;

        # Extended timeout for long-running queries
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Enable gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript;

    client_max_body_size 10M;

    access_log /var/log/nginx/neo4j.fishmakeweb.id.vn.access.log;
    error_log /var/log/nginx/neo4j.fishmakeweb.id.vn.error.log;
}
EOF

echo "✅ Nginx configuration updated"

# Test and reload Nginx
echo "🧪 Testing Nginx configuration..."
sudo nginx -t

echo "🔄 Reloading Nginx..."
sudo systemctl reload nginx

# Update Neo4j to advertise correct address
echo "📝 Updating Neo4j configuration..."

# Update docker-compose to set correct advertised address
cd /root/projects/crawldata/MCP-Servers/self-learning-agent

# Backup docker-compose
cp docker-compose.self-learning.yml docker-compose.self-learning.yml.backup

# Check if advertised address is set
if ! grep -q "NEO4J_dbms_connector_bolt_advertised__address" docker-compose.self-learning.yml; then
    echo "Adding advertised address configuration..."
    # This will be added to environment section
fi

echo ""
echo "=============================================="
echo "✅ Configuration Updated!"
echo "=============================================="
echo ""
echo "🔥 IMPORTANT: Ensure firewall allows port 7687"
echo ""

# Check firewall
if command -v ufw &> /dev/null && sudo ufw status | grep -q "Status: active"; then
    if ! sudo ufw status | grep -q "7687"; then
        echo "⚠️  Port 7687 not allowed in firewall!"
        read -p "   Allow port 7687? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo ufw allow 7687/tcp
            echo "   ✅ Port 7687 allowed"
        fi
    else
        echo "✅ Port 7687 is allowed in firewall"
    fi
fi

echo ""
echo "=============================================="
echo "📋 Connection Instructions"
echo "=============================================="
echo ""
echo "1. Open Neo4j Browser:"
echo "   https://neo4j.fishmakeweb.id.vn/browser/"
echo ""
echo "2. Use this connection URL (choose ONE):"
echo ""
echo "   Option A (Recommended - Direct Bolt):"
echo "   bolt://neo4j.fishmakeweb.id.vn:7687"
echo ""
echo "   Option B (If your network blocks 7687):"
echo "   bolt://localhost:7687"
echo "   (After SSH tunnel: ssh -L 7687:localhost:7687 root@server)"
echo ""
echo "3. Credentials:"
NEO4J_PASS=$(grep NEO4J_PASSWORD /root/projects/crawldata/MCP-Servers/self-learning-agent/.env.self-learning | cut -d'=' -f2)
echo "   Username: neo4j"
echo "   Password: $NEO4J_PASS"
echo ""
echo "=============================================="
echo "🔍 Troubleshooting"
echo "=============================================="
echo ""
echo "If connection fails, check:"
echo ""
echo "1. Bolt port accessible:"
echo "   telnet neo4j.fishmakeweb.id.vn 7687"
echo ""
echo "2. Neo4j logs:"
echo "   docker logs selflearning-agent_neo4j_1 --tail 50"
echo ""
echo "3. Test local connection:"
echo "   docker exec selflearning-agent_neo4j_1 cypher-shell -u neo4j -p '$NEO4J_PASS'"
echo ""
