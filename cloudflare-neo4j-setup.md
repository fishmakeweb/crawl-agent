# Neo4j with Cloudflare Setup Guide

## Problem
Cloudflare only proxies HTTP/HTTPS (ports 80/443) and blocks the Neo4j Bolt protocol (port 7687).

## Solution: DNS-only subdomain for Bolt

### 1. Cloudflare DNS Configuration

Add these DNS records in your Cloudflare dashboard:

| Type | Name | Content | Proxy Status | TTL |
|------|------|---------|-------------|-----|
| A | neo4j | YOUR_SERVER_IP | ✅ Proxied (🟠) | Auto |
| A | neo4j-bolt | YOUR_SERVER_IP | ❌ DNS only (☁️) | Auto |

**Important**: `neo4j-bolt` MUST be **DNS only** (gray cloud), not proxied!

### 2. Restart Neo4j

```bash
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
docker-compose -f docker-compose.self-learning.yml restart neo4j
```

### 3. Open Firewall Port

```bash
# Check if port 7687 is open
sudo ufw status | grep 7687

# If not open, allow it
sudo ufw allow 7687/tcp
sudo ufw reload
```

### 4. Connect from Browser

1. **Open Neo4j Browser**: https://neo4j.fishmakeweb.id.vn/browser/

2. **Connection Settings**:
   - **Connect URL**: `bolt://neo4j-bolt.fishmakeweb.id.vn:7687`
   - **Username**: `neo4j`
   - **Password**: (from `.env.self-learning` NEO4J_PASSWORD)

### 5. Verify Setup

```bash
# Test DNS resolution
nslookup neo4j-bolt.fishmakeweb.id.vn
# Should return your server IP

# Test port accessibility
telnet neo4j-bolt.fishmakeweb.id.vn 7687
# Should connect successfully

# Check Neo4j logs
docker logs selflearning-agent_neo4j_1 --tail 20
```

## Architecture

```
User Browser
    │
    ├─── HTTPS ───→ Cloudflare (proxied) ───→ neo4j.fishmakeweb.id.vn:443
    │                                              │
    │                                              ↓
    │                                          Nginx (reverse proxy)
    │                                              │
    │                                              ↓
    │                                          Neo4j HTTP :7474
    │
    └─── Bolt ────→ DNS only (no proxy) ───→ neo4j-bolt.fishmakeweb.id.vn:7687
                                                   │
                                                   ↓
                                               Neo4j Bolt :7687
```

## Alternative Solutions

### Option 1: SSH Tunnel (Most Secure)
```bash
# On your local machine
ssh -L 7687:localhost:7687 root@YOUR_SERVER_IP

# Then connect to: bolt://localhost:7687
```

### Option 2: Cloudflare Spectrum (Paid)
- Cloudflare Spectrum can proxy TCP/UDP ports
- Requires Enterprise plan
- Not recommended for small projects

### Option 3: Use VPN
- Set up WireGuard/OpenVPN on server
- Connect via VPN
- Access Neo4j via private IP

## Troubleshooting

### Connection Timeout
```bash
# Check if DNS resolves correctly
dig neo4j-bolt.fishmakeweb.id.vn

# Verify port is open
nc -zv neo4j-bolt.fishmakeweb.id.vn 7687

# Check firewall
sudo ufw status verbose
```

### "Proxy Status" is Orange (Proxied)
- Go to Cloudflare DNS settings
- Find `neo4j-bolt` record
- Click the orange cloud to turn it gray
- Wait 1-2 minutes for DNS propagation

### Still Can't Connect
```bash
# Check Neo4j is listening
docker exec selflearning-agent_neo4j_1 ss -tlnp | grep 7687

# Check advertised address
docker exec selflearning-agent_neo4j_1 cat /var/lib/neo4j/conf/neo4j.conf | grep advertised

# Restart with fresh config
docker-compose -f docker-compose.self-learning.yml down
docker-compose -f docker-compose.self-learning.yml up -d
```

## Security Recommendations

1. **Change default password** immediately after first login
2. **Restrict IP access** (optional):
   ```bash
   # In /etc/nginx/sites-available/neo4j.fishmakeweb.id.vn
   # Add under server block:
   allow YOUR_OFFICE_IP;
   deny all;
   ```
3. **Use strong password**:
   ```bash
   # Generate strong password
   openssl rand -base64 32
   # Update in .env.self-learning
   ```
4. **Enable audit logging** (Neo4j Enterprise only)
5. **Regular backups**:
   ```bash
   docker exec selflearning-agent_neo4j_1 neo4j-admin dump --database=neo4j --to=/data/backup.dump
   ```

## Quick Setup Script

```bash
#!/bin/bash
# Quick setup for Neo4j with Cloudflare

echo "🔧 Checking Cloudflare DNS configuration..."
echo ""
echo "Please ensure you have created this DNS record in Cloudflare:"
echo ""
echo "  Type: A"
echo "  Name: neo4j-bolt"
echo "  Content: $(curl -s ifconfig.me)"
echo "  Proxy: DNS only (gray cloud ☁️)"
echo ""
read -p "Press Enter when DNS record is created..."

echo ""
echo "🔥 Opening firewall port 7687..."
sudo ufw allow 7687/tcp
sudo ufw reload

echo ""
echo "🔄 Restarting Neo4j..."
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
docker-compose -f docker-compose.self-learning.yml restart neo4j

echo ""
echo "⏳ Waiting for Neo4j to start..."
sleep 10

echo ""
echo "✅ Setup complete!"
echo ""
echo "Connect to:"
echo "  Browser: https://neo4j.fishmakeweb.id.vn/browser/"
echo "  Bolt URL: bolt://neo4j-bolt.fishmakeweb.id.vn:7687"
echo ""
```
