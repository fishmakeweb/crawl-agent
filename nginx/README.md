# Training Agent Nginx Deployment

This directory contains nginx configuration and deployment scripts for the Training Agent.

## Quick Deployment

```bash
# Make deployment script executable
chmod +x nginx/deploy.sh

# Run deployment (requires root)
sudo nginx/deploy.sh
```

## Manual Deployment

If you prefer to deploy manually:

```bash
# 1. Copy nginx configuration
sudo cp nginx/train.fishmakeweb.id.vn.conf /etc/nginx/sites-available/train.fishmakeweb.id.vn

# 2. Enable the site
sudo ln -s /etc/nginx/sites-available/train.fishmakeweb.id.vn /etc/nginx/sites-enabled/

# 3. Test nginx configuration
sudo nginx -t

# 4. Reload nginx
sudo systemctl reload nginx

# 5. Ensure training server is running
docker ps | grep training-server

# 6. Start training server if not running
cd /root/projects/crawldata/MCP-Servers
docker run -d --name training-server \
  --restart unless-stopped \
  -p 8091:8091 \
  --env-file self-learning-agent/.env \
  -v $(pwd)/self-learning-agent/knowledge_db:/app/knowledge_db \
  self-learning-agent-training:latest

# 7. Verify deployment
curl -k https://train.fishmakeweb.id.vn/health
```

## Configuration Details

- **Domain**: train.fishmakeweb.id.vn
- **Port**: 8091 (internal)
- **SSL**: Cloudflare Origin Certificates
- **Timeouts**: 300s for long-running crawl operations
- **Max Body Size**: 50M for large crawl requests
- **CORS**: Enabled for API access
- **WebSocket**: Supported at `/ws` endpoint

## Logs

View nginx logs:
```bash
sudo tail -f /var/log/nginx/train.fishmakeweb.id.vn.access.log
sudo tail -f /var/log/nginx/train.fishmakeweb.id.vn.error.log
```

View docker logs:
```bash
docker logs -f training-server
```

## Troubleshooting

### Check nginx status
```bash
sudo systemctl status nginx
sudo nginx -t
```

### Check training server status
```bash
docker ps | grep training-server
docker logs training-server
```

### Check port availability
```bash
sudo netstat -tlnp | grep 8091
```

### Restart services
```bash
# Restart nginx
sudo systemctl restart nginx

# Restart training server
docker restart training-server
```

## Production Checklist

- [x] Nginx configuration created
- [x] SSL certificates configured (Cloudflare Origin)
- [x] CORS headers enabled
- [x] Extended timeouts for crawl operations
- [x] WebSocket support enabled
- [x] Gzip compression enabled
- [x] Security headers configured
- [x] Logging configured
- [x] Health check endpoint configured
- [x] Docker container with restart policy
- [x] Environment variables configured

## API Endpoints

Once deployed, the following endpoints will be available:

- `GET /health` - Health check
- `POST /crawl` - Submit crawl job
- `POST /feedback` - Submit user feedback
- `GET /metrics` - Get performance metrics
- `WS /ws` - WebSocket for real-time updates

## Monitoring

Check service health:
```bash
# Quick health check
curl -k https://train.fishmakeweb.id.vn/health | jq .

# Detailed metrics
curl -k https://train.fishmakeweb.id.vn/metrics | jq .
```

Monitor in real-time:
```bash
# Watch nginx access logs
sudo tail -f /var/log/nginx/train.fishmakeweb.id.vn.access.log

# Watch training server logs
docker logs -f training-server
```
