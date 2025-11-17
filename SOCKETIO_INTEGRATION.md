# Socket.IO Integration for Training Agent

## Changes Made

### 1. Dependencies (`requirements.txt`)
✅ Added Socket.IO packages:
- `python-socketio>=5.10.0`
- `python-engineio>=4.8.0`

### 2. Training Agent (`training_agent.py`)
✅ Integrated Socket.IO server:
- Imported `socketio` module
- Created `AsyncServer` with ASGI mode
- Created `socket_app` wrapping FastAPI app
- Added Socket.IO event handlers:
  - `connect` - Handle client connections
  - `disconnect` - Handle disconnections
  - `ping/pong` - Heartbeat mechanism
- Updated broadcast functions to emit to both Socket.IO and native WebSocket clients
- Updated Dockerfile CMD to use `socket_app`

### 3. Nginx Configuration (`nginx/train.fishmakeweb.id.vn.conf`)
✅ Added Socket.IO routing:
- New `/socket.io/` location block with WebSocket upgrade
- Kept `/ws` endpoint for backward compatibility
- Extended timeouts (86400s for persistent connections)
- Disabled buffering for real-time updates

### 4. Dockerfile (`Dockerfile.training`)
✅ Updated CMD to run `socket_app` instead of `app`

## Rebuild & Deploy Commands

```bash
# 1. Stop current container
docker stop training-server
docker rm training-server

# 2. Rebuild Docker image
cd /root/projects/crawldata/MCP-Servers
docker build -t self-learning-agent-training:latest -f self-learning-agent/Dockerfile.training .

# 3. Run new container
docker run -d --name training-server \
  --restart unless-stopped \
  -p 8091:8091 \
  --env-file self-learning-agent/.env \
  -v $(pwd)/self-learning-agent/knowledge_db:/app/knowledge_db \
  self-learning-agent-training:latest

# 4. Update nginx configuration
sudo cp self-learning-agent/nginx/train.fishmakeweb.id.vn.conf /etc/nginx/sites-available/train.fishmakeweb.id.vn
sudo nginx -t
sudo systemctl reload nginx

# 5. Verify Socket.IO connection
# Check logs for Socket.IO client connections
docker logs -f training-server

# 6. Test from browser console
# Open https://train.fishmakeweb.id.vn and check WebSocket connection
```

## Quick Deploy Script

```bash
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
chmod +x nginx/redeploy-socketio.sh
sudo nginx/redeploy-socketio.sh
```

## Socket.IO Endpoints

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `/socket.io/` | Socket.IO | Real-time updates (primary) |
| `/ws` | WebSocket | Native WebSocket (backward compat) |
| `/health` | HTTP | Health check |

## Socket.IO Events

### Server → Client Events

| Event | Payload | Description |
|-------|---------|-------------|
| `connected` | `{status, sid}` | Connection confirmed |
| `pong` | `{}` | Response to ping |
| `job_completed` | `{type, job_id, success, items_count}` | Crawl job finished |
| `feedback_received` | `{type, job_id, quality_rating}` | User feedback processed |

### Client → Server Events

| Event | Payload | Description |
|-------|---------|-------------|
| `ping` | `{}` | Heartbeat check |

## Frontend Integration

The existing frontend code using `socket.io-client` will now work correctly:

```typescript
// ui/src/services/websocket.ts
const socket = io(WS_URL, {
  transports: ['websocket'],
  reconnection: true,
});

socket.on('job_completed', (data) => {
  // Handle job completion
});

socket.on('feedback_received', (data) => {
  // Handle feedback
});
```

## Testing

### Test Socket.IO Connection

```bash
# Check if Socket.IO is responding
curl -v https://train.fishmakeweb.id.vn/socket.io/?EIO=4&transport=polling

# Expected: 200 OK with Socket.IO session ID
```

### Monitor Connections

```bash
# Watch docker logs for connection events
docker logs -f training-server | grep "Socket.IO"

# Expected output:
# 🔌 Socket.IO client connected: <sid>
# 🔌 Socket.IO client disconnected: <sid>
```

### Browser Console Test

```javascript
// Open browser console on https://train.fishmakeweb.id.vn
const socket = io('https://train.fishmakeweb.id.vn', {
  transports: ['websocket']
});

socket.on('connected', (data) => {
  console.log('Connected:', data);
});

socket.emit('ping');
socket.on('pong', () => {
  console.log('Pong received!');
});
```

## Troubleshooting

### Issue: Still getting 403 Forbidden

**Check:**
1. Docker container rebuilt with new code
2. Nginx config updated and reloaded
3. Socket.IO imports working in container

**Fix:**
```bash
docker logs training-server | grep -i error
docker exec training-server python -c "import socketio; print('OK')"
```

### Issue: WebSocket upgrade failing

**Check nginx logs:**
```bash
sudo tail -f /var/log/nginx/train.fishmakeweb.id.vn.error.log
```

**Verify proxy settings:**
```bash
sudo nginx -T | grep -A 20 "location /socket.io"
```

### Issue: Cannot import socketio

**Verify dependencies installed:**
```bash
docker exec training-server pip list | grep socketio
```

**Expected:**
```
python-engineio    4.8.0
python-socketio    5.10.0
```

## Rollback

If issues occur, rollback to previous version:

```bash
# Use previous image
docker stop training-server
docker rm training-server
# Run previous version (without Socket.IO)
```

## Next Steps

After successful deployment:

1. ✅ Verify Socket.IO connection in browser
2. ✅ Test job submission and real-time updates
3. ✅ Monitor for any connection issues
4. ✅ Update UI deployment with production URL
5. ✅ Set up monitoring/alerting for WebSocket health
