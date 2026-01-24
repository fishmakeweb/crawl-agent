# Neo4j Public Access Setup Guide

## Current Status
- ✅ Neo4j container running
- ✅ Ports 7474 (HTTP), 7687 (Bolt) exposed
- ✅ Listening on all interfaces (0.0.0.0)
- ❌ Not accessible from neo4j.fishmakeweb.id.vn

## Problem Analysis
The domain `neo4j.fishmakeweb.id.vn` is not resolving to Neo4j because:
1. **No reverse proxy** configured for this subdomain
2. Firewall might be blocking ports
3. Neo4j needs additional security configuration for external access

## Solution

### Option 1: Nginx Reverse Proxy (Recommended)

#### Step 1: Install Nginx (if not installed)
```bash
sudo apt update
sudo apt install nginx -y
```

#### Step 2: Create Neo4j proxy configuration
```bash
sudo nano /etc/nginx/sites-available/neo4j.fishmakeweb.id.vn
```

**Configuration:**
```nginx
server {
    listen 80;
    server_name neo4j.fishmakeweb.id.vn;

    # Redirect HTTP to HTTPS (after SSL is set up)
    # return 301 https://$server_name$request_uri;

    # For now, proxy to Neo4j HTTP
    location / {
        proxy_pass http://localhost:7474;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support for Neo4j Browser
    location /ws {
        proxy_pass http://localhost:7474/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### Step 3: Enable the site
```bash
sudo ln -s /etc/nginx/sites-available/neo4j.fishmakeweb.id.vn /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### Step 4: Setup SSL with Let's Encrypt (HTTPS)
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d neo4j.fishmakeweb.id.vn
```

### Option 2: Direct Port Access (Not Recommended - Security Risk)

#### Open firewall ports
```bash
# UFW
sudo ufw allow 7474/tcp
sudo ufw allow 7687/tcp

# Or iptables
sudo iptables -A INPUT -p tcp --dport 7474 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 7687 -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

⚠️ **WARNING**: This exposes Neo4j directly to the internet without encryption!

### Option 3: Docker Compose with Traefik (Advanced)

Add Traefik to your docker-compose:

```yaml
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--api.insecure=true"
      - "--providers.docker=true"
      - "--entrypoints.web.address=:80"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.letsencrypt.acme.email=admin@fishmakeweb.id.vn"
      - "--certificatesresolvers.letsencrypt.acme.storage=/letsencrypt/acme.json"
      - "--certificatesresolvers.letsencrypt.acme.httpchallenge.entrypoint=web"
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./letsencrypt:/letsencrypt
    networks:
      - selflearning-network

  neo4j:
    image: neo4j:5.15-community
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.neo4j.rule=Host(`neo4j.fishmakeweb.id.vn`)"
      - "traefik.http.routers.neo4j.entrypoints=websecure"
      - "traefik.http.routers.neo4j.tls.certresolver=letsencrypt"
      - "traefik.http.services.neo4j.loadbalancer.server.port=7474"
    # ... rest of neo4j config
```

## Security Recommendations

### 1. Update Neo4j Environment Variables
Add to docker-compose.self-learning.yml:

```yaml
environment:
  # Enable remote access
  - NEO4J_dbms_connector_bolt_listen__address=0.0.0.0:7687
  - NEO4J_dbms_connector_http_listen__address=0.0.0.0:7474
  
  # Security settings
  - NEO4J_dbms_security_auth__enabled=true
  - NEO4J_dbms_security_procedures_unrestricted=apoc.*
  
  # CORS for browser access
  - NEO4J_dbms_security_allow__csv__import__from__file__urls=true
  - NEO4J_browser_remote__content__hostname__whitelist=*
  - NEO4J_browser_post__connect__cmd=config; config connectionTimeout: 30000;
```

### 2. Use Strong Password
Ensure NEO4J_PASSWORD in .env is strong:
```bash
NEO4J_PASSWORD=$(openssl rand -base64 32)
```

### 3. IP Whitelisting (Nginx)
Add to nginx config:
```nginx
# Only allow specific IPs
allow 123.45.67.89;  # Your office IP
deny all;
```

## Verification Steps

### 1. Check DNS
```bash
nslookup neo4j.fishmakeweb.id.vn
# Should point to your server IP
```

### 2. Test Local Access
```bash
curl http://localhost:7474
# Should return Neo4j info
```

### 3. Test Remote Access (after setup)
```bash
curl http://neo4j.fishmakeweb.id.vn
# Should return Neo4j browser
```

### 4. Browser Access
Open: http://neo4j.fishmakeweb.id.vn
- Username: neo4j
- Password: (from NEO4J_PASSWORD env var)
- Connection URL: bolt://neo4j.fishmakeweb.id.vn:7687

## Troubleshooting

### Domain not resolving
```bash
# Check DNS
dig neo4j.fishmakeweb.id.vn

# Add to /etc/hosts for testing
echo "YOUR_SERVER_IP neo4j.fishmakeweb.id.vn" | sudo tee -a /etc/hosts
```

### Connection refused
```bash
# Check Neo4j logs
docker logs selflearning-agent_neo4j_1 --tail 100

# Check if port is open
telnet neo4j.fishmakeweb.id.vn 7474
```

### SSL issues
```bash
# Check certificate
sudo certbot certificates

# Renew if needed
sudo certbot renew --dry-run
```

## Quick Setup Script

Run this to set up Nginx proxy:

```bash
#!/bin/bash
# Quick Neo4j public access setup

# Install nginx
sudo apt update && sudo apt install -y nginx certbot python3-certbot-nginx

# Create config
sudo tee /etc/nginx/sites-available/neo4j.fishmakeweb.id.vn > /dev/null <<'EOF'
server {
    listen 80;
    server_name neo4j.fishmakeweb.id.vn;

    location / {
        proxy_pass http://localhost:7474;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/neo4j.fishmakeweb.id.vn /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Setup SSL (interactive)
sudo certbot --nginx -d neo4j.fishmakeweb.id.vn

echo "✅ Neo4j should now be accessible at https://neo4j.fishmakeweb.id.vn"
```
