#!/bin/bash
# Automated Neo4j Public Access Setup
# Makes Neo4j accessible at neo4j.fishmakeweb.id.vn

set -e

DOMAIN="neo4j.fishmakeweb.id.vn"
NEO4J_PORT=7474

echo "=============================================="
echo "🚀 Neo4j Public Access Setup"
echo "=============================================="
echo ""
echo "Domain: $DOMAIN"
echo "Neo4j Port: $NEO4J_PORT"
echo ""

# Check if Neo4j is running
if ! docker ps | grep -q neo4j; then
    echo "❌ Neo4j container is not running!"
    echo "   Start it with: docker-compose -f docker-compose.self-learning.yml up -d neo4j"
    exit 1
fi

echo "✅ Neo4j container is running"

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "📦 Installing Nginx..."
    sudo apt update
    sudo apt install -y nginx
else
    echo "✅ Nginx is installed"
fi

# Create Nginx configuration
echo "📝 Creating Nginx configuration..."
sudo tee /etc/nginx/sites-available/$DOMAIN > /dev/null <<'EOF'
server {
    listen 80;
    server_name neo4j.fishmakeweb.id.vn;

    # Increase buffer sizes for large Neo4j responses
    client_max_body_size 100M;
    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_busy_buffers_size 256k;

    # Main Neo4j browser interface
    location / {
        proxy_pass http://localhost:7474;
        proxy_http_version 1.1;
        
        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts for long-running queries
        proxy_connect_timeout 600;
        proxy_send_timeout 600;
        proxy_read_timeout 600;
        send_timeout 600;
    }

    # Access logs
    access_log /var/log/nginx/neo4j-access.log;
    error_log /var/log/nginx/neo4j-error.log;
}

# Stream proxy for Bolt protocol (port 7687)
# This is needed for Neo4j Browser to connect
stream {
    upstream neo4j_bolt {
        server localhost:7687;
    }

    server {
        listen 7687;
        proxy_pass neo4j_bolt;
        proxy_timeout 600s;
        proxy_connect_timeout 600s;
    }
}
EOF

echo "✅ Nginx configuration created"

# Enable the site
echo "🔗 Enabling site..."
sudo ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/

# Test Nginx configuration
echo "🧪 Testing Nginx configuration..."
if sudo nginx -t; then
    echo "✅ Nginx configuration is valid"
else
    echo "❌ Nginx configuration has errors!"
    exit 1
fi

# Reload Nginx
echo "🔄 Reloading Nginx..."
sudo systemctl reload nginx

echo ""
echo "=============================================="
echo "✅ Neo4j HTTP Access Configured!"
echo "=============================================="
echo ""
echo "Test access:"
echo "  curl http://$DOMAIN"
echo ""
echo "Browser access:"
echo "  http://$DOMAIN"
echo ""

# Ask about SSL
read -p "Do you want to setup SSL/HTTPS with Let's Encrypt? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Install certbot if needed
    if ! command -v certbot &> /dev/null; then
        echo "📦 Installing Certbot..."
        sudo apt install -y certbot python3-certbot-nginx
    fi
    
    echo "🔐 Setting up SSL certificate..."
    echo "   Make sure DNS for $DOMAIN points to this server!"
    echo ""
    
    sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@fishmakeweb.id.vn || {
        echo "⚠️  SSL setup failed. You can run it manually later:"
        echo "   sudo certbot --nginx -d $DOMAIN"
    }
    
    echo ""
    echo "=============================================="
    echo "✅ Setup Complete with SSL!"
    echo "=============================================="
    echo ""
    echo "Access Neo4j:"
    echo "  https://$DOMAIN"
    echo ""
    echo "Connection details:"
    echo "  Username: neo4j"
    echo "  Password: (check .env.self-learning NEO4J_PASSWORD)"
    echo "  Bolt URL: bolt://$DOMAIN:7687"
    echo ""
else
    echo ""
    echo "=============================================="
    echo "✅ Setup Complete (HTTP only)"
    echo "=============================================="
    echo ""
    echo "Access Neo4j:"
    echo "  http://$DOMAIN"
    echo ""
    echo "⚠️  WARNING: Using HTTP without encryption!"
    echo "   To add SSL later, run:"
    echo "   sudo certbot --nginx -d $DOMAIN"
    echo ""
fi

# Show Neo4j password
echo "=============================================="
echo "Neo4j Credentials"
echo "=============================================="
NEO4J_PASS=$(grep NEO4J_PASSWORD /root/projects/crawldata/MCP-Servers/self-learning-agent/.env.self-learning | cut -d'=' -f2)
echo "Username: neo4j"
echo "Password: $NEO4J_PASS"
echo ""
echo "💡 Update this password in Neo4j Browser after first login!"
echo ""

# Firewall check
echo "=============================================="
echo "🔥 Firewall Check"
echo "=============================================="
if command -v ufw &> /dev/null && sudo ufw status | grep -q "Status: active"; then
    echo "UFW is active. Checking rules..."
    if sudo ufw status | grep -q "80"; then
        echo "✅ Port 80 is allowed"
    else
        echo "⚠️  Port 80 not in firewall rules"
        read -p "Allow HTTP (port 80)? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo ufw allow 80/tcp
            echo "✅ Port 80 allowed"
        fi
    fi
    
    if sudo ufw status | grep -q "443"; then
        echo "✅ Port 443 is allowed"
    else
        echo "⚠️  Port 443 not in firewall rules"
        read -p "Allow HTTPS (port 443)? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo ufw allow 443/tcp
            echo "✅ Port 443 allowed"
        fi
    fi
    
    # Optional: Allow Bolt directly (not recommended)
    echo ""
    read -p "Allow direct Bolt access (port 7687)? Not recommended! (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo ufw allow 7687/tcp
        echo "✅ Port 7687 allowed (direct Bolt access)"
    fi
else
    echo "ℹ️  UFW not active or not installed"
fi

echo ""
echo "=============================================="
echo "🎉 Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Open http://$DOMAIN (or https:// if SSL enabled)"
echo "2. Login with credentials shown above"
echo "3. Change default password"
echo "4. Start exploring your graph!"
echo ""
