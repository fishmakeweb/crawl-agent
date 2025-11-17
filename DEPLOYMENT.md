# Deployment Guide: Training Agent to train.fishmakeweb.id.vn

## Prerequisites

✅ Docker image built: `self-learning-agent-training:latest`  
✅ Training server running locally on port 8091  
✅ Cloudflare Origin certificates installed at:
  - `/etc/ssl/certs/cloudflare-origin.pem`
  - `/etc/ssl/private/cloudflare-origin-key.pem`

## Quick Deploy

```bash
cd /root/projects/crawldata/MCP-Servers/self-learning-agent/nginx
chmod +x deploy.sh
sudo ./deploy.sh
```

## Files Created

1. **`nginx/train.fishmakeweb.id.vn.conf`** - Nginx configuration
2. **`nginx/deploy.sh`** - Automated deployment script
3. **`nginx/README.md`** - Nginx deployment documentation
4. **`ui/.env.production`** - Production environment variables for UI
5. **`ui/.env.example`** - Updated example with production URL

## Deployment Steps (Manual)

### 1. Deploy Nginx Configuration

```bash
# Copy configuration to nginx sites-available
sudo cp /root/projects/crawldata/MCP-Servers/self-learning-agent/nginx/train.fishmakeweb.id.vn.conf \
     /etc/nginx/sites-available/train.fishmakeweb.id.vn

# Enable the site
sudo ln -s /etc/nginx/sites-available/train.fishmakeweb.id.vn \
     /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 2. Ensure Training Server is Running

```bash
# Check if running
docker ps | grep training-server

# If not running, start it
cd /root/projects/crawldata/MCP-Servers
docker run -d --name training-server \
  --restart unless-stopped \
  -p 8091:8091 \
  --env-file self-learning-agent/.env \
  -v $(pwd)/self-learning-agent/knowledge_db:/app/knowledge_db \
  self-learning-agent-training:latest
```

### 3. Verify Deployment

```bash
# Check health endpoint
curl -k https://train.fishmakeweb.id.vn/health

# With formatted output
curl -s -k https://train.fishmakeweb.id.vn/health | jq .
```

## Configuration Details

### Nginx Configuration Highlights

- **SSL**: Cloudflare Origin Certificates
- **Backend**: `localhost:8091`
- **Timeouts**: 300s (for long-running crawl operations)
- **Max Body Size**: 50M (for large crawl requests)
- **CORS**: Enabled for cross-origin API access
- **WebSocket**: Supported at `/ws` endpoint
- **Compression**: Gzip enabled for API responses
- **Security Headers**: HSTS, X-Frame-Options, CSP headers configured

### Docker Container Configuration

- **Port Mapping**: 8091:8091
- **Restart Policy**: `unless-stopped`
- **Volume**: Knowledge database persisted at `./knowledge_db`
- **Environment**: Loaded from `.env` file

## API Endpoints Available

Once deployed at `https://train.fishmakeweb.id.vn`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and metrics |
| `/crawl` | POST | Submit crawl job for training |
| `/feedback` | POST | Submit user feedback |
| `/metrics` | GET | Get performance metrics |
| `/ws` | WS | WebSocket for real-time updates |

## Monitoring & Logs

### Nginx Logs
```bash
# Access logs
sudo tail -f /var/log/nginx/train.fishmakeweb.id.vn.access.log

# Error logs
sudo tail -f /var/log/nginx/train.fishmakeweb.id.vn.error.log
```

### Docker Logs
```bash
# Follow logs
docker logs -f training-server

# Last 100 lines
docker logs --tail 100 training-server
```

### Health Monitoring
```bash
# Watch health status
watch -n 5 'curl -s -k https://train.fishmakeweb.id.vn/health | jq .'
```

## Troubleshooting

### Issue: Nginx configuration test fails
```bash
# Check syntax
sudo nginx -t

# Check if port 443 is available
sudo netstat -tlnp | grep 443
```

### Issue: Cannot reach backend
```bash
# Check if training server is running
docker ps | grep training-server

# Check if port 8091 is listening
sudo netstat -tlnp | grep 8091

# Check docker logs
docker logs training-server
```

### Issue: SSL certificate errors
```bash
# Verify certificates exist
ls -la /etc/ssl/certs/cloudflare-origin.pem
ls -la /etc/ssl/private/cloudflare-origin-key.pem

# Check certificate validity
openssl x509 -in /etc/ssl/certs/cloudflare-origin.pem -text -noout
```

### Issue: 502 Bad Gateway
```bash
# Backend not responding
docker logs training-server

# Check backend health directly
curl http://localhost:8091/health

# Restart training server
docker restart training-server
```

## Rollback

If you need to rollback the deployment:

```bash
# Disable nginx site
sudo rm /etc/nginx/sites-enabled/train.fishmakeweb.id.vn

# Reload nginx
sudo systemctl reload nginx

# Stop training server
docker stop training-server
docker rm training-server
```

## Next Steps

1. **Configure Cloudflare**: Ensure DNS record points to your server
2. **Update UI**: Build and deploy UI with production environment variables
3. **Set up monitoring**: Configure alerting for service health
4. **Database setup**: Configure Qdrant, Neo4j, Redis if needed
5. **Backup strategy**: Set up regular backups of `knowledge_db` volume

## Production Checklist

- [ ] Nginx configuration deployed
- [ ] SSL certificates verified
- [ ] Training server running with restart policy
- [ ] Health check responding
- [ ] Logs configured and accessible
- [ ] DNS configured in Cloudflare
- [ ] UI updated with production API URL
- [ ] Monitoring alerts configured
- [ ] Backup strategy in place
- [ ] Documentation updated

## Support

For issues or questions:
- Check logs: nginx and docker
- Verify configuration: `nginx -t`
- Test backend directly: `curl http://localhost:8091/health`
- Review this guide's troubleshooting section
