"""
Training Agent Entry Point
Active learning with Agent-Lightning integration
"""
import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import uuid
import socketio
import logging
from datetime import datetime

from config import Config
from gemini_client import GeminiClient
from agents.shared_crawler_agent import SharedCrawlerAgent
from algorithms.self_improving_algorithm import SelfImprovingCrawlerAlgorithm
from knowledge.hybrid_knowledge_store import HybridKnowledgeStore
from knowledge.rl_controller import RLResourceController


class SocketIOLogHandler(logging.Handler):
    """Custom handler to emit logs via Socket.IO"""

    def __init__(self, socketio_server):
        super().__init__()
        self.sio = socketio_server
        self.job_id = None  # Set before each crawl

    def emit(self, record):
        """Emit log record via Socket.IO"""
        try:
            log_entry = {
                'level': record.levelname,
                'message': self.format(record),
                'logger': record.name,
                'timestamp': datetime.now().isoformat(),
                'job_id': self.job_id
            }

            # Send asynchronously
            asyncio.create_task(
                self.sio.emit('crawl_log', log_entry)
            )
        except Exception:
            self.handleError(record)

# Initialize
app = FastAPI(title="Training Crawler Agent", version="1.0.0")

# CORS - Must be added BEFORE Socket.IO wrapping
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=False,
    engineio_logger=False
)
socket_app = socketio.ASGIApp(sio, app)

# Load config
config = Config()
config.MODE = "training"
config.validate()

# Initialize components
gemini_client = GeminiClient(config.gemini)
rl_controller = RLResourceController(gemini_client, config.training)
knowledge_store = HybridKnowledgeStore(gemini_client, rl_controller, config.knowledge_store)

# Load previous training resources for incremental learning
import os
import glob
import re
previous_resources = None
if os.path.exists("/app/frozen_resources"):
    existing_files = glob.glob("/app/frozen_resources/training_resources_v*.json")
    if existing_files:
        # Find the latest version
        max_version = 0
        latest_file = None
        for filepath in existing_files:
            match = re.search(r'training_resources_v(\d+)\.json', filepath)
            if match:
                version = int(match.group(1))
                if version > max_version:
                    max_version = version
                    latest_file = filepath
        
        if latest_file:
            try:
                with open(latest_file, 'r') as f:
                    previous_resources = json.load(f)
                print(f"📚 Loaded previous training resources from v{max_version}")
                print(f"   - Domain patterns: {len(previous_resources.get('domain_patterns', {}))} domains")
                print(f"   - Performance history: {len(previous_resources.get('performance_history', []))} cycles")
                
                # Pre-populate knowledge store with previous learnings
                if previous_resources.get('domain_patterns'):
                    for domain, patterns in previous_resources['domain_patterns'].items():
                        # Note: This assumes knowledge_store has a method to import patterns
                        # You may need to adjust based on your HybridKnowledgeStore implementation
                        pass  # Will be loaded during algorithm initialization
            except Exception as e:
                print(f"⚠️  Failed to load previous resources: {e}")
                previous_resources = None

# Initialize algorithm
algorithm = SelfImprovingCrawlerAlgorithm(
    gemini_client=gemini_client,
    knowledge_store=knowledge_store,
    update_frequency=config.training.UPDATE_FREQUENCY
)

# Initialize agent in training mode
agent = SharedCrawlerAgent(gemini_client, mode="training")

# Active connections for WebSocket
active_connections: List[WebSocket] = []

# Job queue
job_queue = {}
feedback_queue = {}

# Configure logging for real-time streaming
log_handler = SocketIOLogHandler(sio)
log_handler.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(message)s')
log_handler.setFormatter(log_formatter)

# Attach to relevant loggers
logging.getLogger('crawl4ai').addHandler(log_handler)
logging.getLogger('crawl4ai_wrapper').addHandler(log_handler)

print(f"🎓 Training Agent started on port 8091")
print(f"   Update frequency: Every {config.training.UPDATE_FREQUENCY} rollouts")


# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    """Handle Socket.IO client connection"""
    print(f"🔌 Socket.IO client connected: {sid}")
    await sio.emit('connected', {'status': 'connected', 'sid': sid})

@sio.event
async def disconnect(sid):
    """Handle Socket.IO client disconnection"""
    print(f"🔌 Socket.IO client disconnected: {sid}")

@sio.event
async def ping(sid):
    """Handle ping from client"""
    await sio.emit('pong', room=sid)

@sio.event
async def subscribe_logs(sid, data):
    """Client subscribes to logs for specific job"""
    job_id = data.get('job_id')
    print(f"📡 Client {sid} subscribed to logs for job {job_id}")

@sio.event
async def unsubscribe_logs(sid, data):
    """Client unsubscribes from logs"""
    job_id = data.get('job_id')
    print(f"📡 Client {sid} unsubscribed from logs for job {job_id}")


# Request models
class TrainCrawlRequest(BaseModel):
    url: str
    user_description: str
    extraction_schema: Optional[Dict[str, Any]] = None
    feedback_from_previous: Optional[str] = None


class FeedbackRequest(BaseModel):
    job_id: str
    feedback: str


class CrawlResponse(BaseModel):
    job_id: str
    success: bool
    data: list
    metadata: Dict[str, Any]
    base_reward: float
    error: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "training",
        "update_cycle": algorithm.current_cycle,
        "pending_rollouts": len(algorithm.pending_rollouts),
        "max_rollouts": algorithm.update_frequency,
        "gemini_stats": gemini_client.get_stats(),
        "knowledge_metrics": knowledge_store.get_metrics(),
        "rl_policy": rl_controller.get_policy_summary()
    }


@app.post("/train-crawl", response_model=CrawlResponse)
async def train_crawl(request: TrainCrawlRequest):
    """Execute training crawl with active learning"""
    job_id = str(uuid.uuid4())

    try:
        task = {
            "url": request.url,
            "user_description": request.user_description,
            "extraction_schema": request.extraction_schema or {},
            "feedback_from_previous": request.feedback_from_previous
        }

        # Set job_id context for logging
        log_handler.job_id = job_id

        # Emit crawl start event
        await sio.emit('crawl_started', {
            'job_id': job_id,
            'url': request.url,
            'description': request.user_description
        })

        # Execute with current resources
        result = await agent.execute_crawl(task)

        # Clear job_id context
        log_handler.job_id = None

        # Debug logging
        print(f"📊 Crawl result keys: {result.keys()}")
        print(f"📊 Data field type: {type(result.get('data'))}")
        print(f"📊 Data length: {len(result.get('data', []))}")
        if result.get('data'):
            print(f"📊 First item: {result['data'][0] if result['data'] else 'None'}")

        # Calculate base reward
        base_reward = 0.8 if result["success"] else 0.2

        # Store job for feedback
        job_queue[job_id] = {
            "task": task,
            "result": result,
            "base_reward": base_reward,
            "awaiting_feedback": True
        }

        # Track rollout in algorithm
        algorithm.pending_rollouts.append(job_id)

        # Emit Socket.IO update for pending rollouts
        await sio.emit('pending_rollouts_updated', {
            'pending_count': len(algorithm.pending_rollouts),
            'update_frequency': algorithm.update_frequency,
            'cycle': algorithm.current_cycle
        })

        # Broadcast to WebSocket clients
        await broadcast_job_completed(job_id, result)

        response_data = result.get("data", [])
        print(f"📤 Returning {len(response_data)} items in HTTP response")
        print(f"📊 Pending rollouts: {len(algorithm.pending_rollouts)}/{algorithm.update_frequency}")

        return CrawlResponse(
            job_id=job_id,
            success=result["success"],
            data=response_data,
            metadata=result.get("metadata", {}),
            base_reward=base_reward,
            error=result.get("error")
        )

    except Exception as e:
        # Clear job_id context on error
        log_handler.job_id = None
        raise HTTPException(status_code=500, detail=str(e))


async def trigger_learning_update():
    """Trigger learning cycle when N rollouts complete"""
    print(f"\n{'='*60}")
    print(f"🔄 Triggering learning update (cycle {algorithm.current_cycle})...")
    print(f"{'='*60}")

    # Collect rollout data from job_queue
    rollout_data = []
    for rollout_id in algorithm.pending_rollouts:
        if rollout_id in job_queue:
            job = job_queue[rollout_id]
            rollout_data.append({
                'id': rollout_id,
                'task': job['task'],
                'result': job['result'],
                'reward': job['base_reward'],
                'metadata': {'user_feedback': feedback_queue.get(rollout_id, {}).get('original')}
            })

    # Call algorithm's interactive learning method
    new_resources = await algorithm.learn_from_interactive_rollouts(rollout_data)

    # Update cycle and clear pending
    algorithm.current_cycle += 1
    algorithm.pending_rollouts = []
    algorithm.feedback_queue = []

    # Auto-save resources to frozen_resources folder
    try:
        import os
        import glob
        import re
        from datetime import datetime
        
        # Create frozen_resources directory
        os.makedirs("/app/frozen_resources", exist_ok=True)
        
        # Find the highest existing version number
        existing_files = glob.glob("/app/frozen_resources/training_resources_v*.json")
        max_version = 0
        for filepath in existing_files:
            match = re.search(r'training_resources_v(\d+)\.json', filepath)
            if match:
                version = int(match.group(1))
                if version > max_version:
                    max_version = version
        
        # Use next version number (don't overwrite existing)
        next_version = max_version + 1
        
        # Merge with previous resources for incremental learning
        merged_domain_patterns = {}
        merged_performance_history = []
        
        # Load latest version to merge with
        if max_version > 0:
            try:
                latest_file = f"/app/frozen_resources/training_resources_v{max_version}.json"
                if os.path.exists(latest_file):
                    with open(latest_file, 'r') as f:
                        prev_data = json.load(f)
                        merged_domain_patterns = prev_data.get('domain_patterns', {})
                        merged_performance_history = prev_data.get('performance_history', [])
                        print(f"📚 Merging with v{max_version}: {len(merged_domain_patterns)} domains, {len(merged_performance_history)} history entries")
            except Exception as e:
                print(f"⚠️  Failed to load previous version for merging: {e}")
        
        # Merge new patterns with previous ones
        current_patterns = knowledge_store.get_domain_patterns()
        for domain, patterns in current_patterns.items():
            if domain in merged_domain_patterns:
                # Merge patterns for existing domain
                merged_domain_patterns[domain].extend(patterns)
            else:
                # New domain
                merged_domain_patterns[domain] = patterns
        
        # Merge performance history
        merged_performance_history.extend(algorithm.performance_history)
        
        # Prepare resources for export
        resources = {
            "version": next_version,
            "frozen_at": datetime.now().isoformat(),
            "previous_version": max_version if max_version > 0 else None,
            "extraction_prompt": algorithm._get_default_prompt(),
            "crawl_config": {
                "timeout": 30,
                "wait_for": "networkidle",
                "screenshot": False,
                "max_pages": 50,
                "headless": True
            },
            "domain_patterns": merged_domain_patterns,
            "performance_history": merged_performance_history,
            "total_cycles": algorithm.current_cycle + (prev_data.get('total_cycles', 0) if max_version > 0 else 0),
            "performance_metrics": new_resources.get('performance_metrics', {}),
            "incremental_learning": {
                "base_version": max_version if max_version > 0 else None,
                "new_domains_added": len([d for d in current_patterns.keys() if d not in (prev_data.get('domain_patterns', {}) if max_version > 0 else {})]),
                "new_patterns_count": sum(len(p) for p in current_patterns.values())
            }
        }
        
        # Save versioned file with next version number
        filename = f"/app/frozen_resources/training_resources_v{next_version}.json"
        with open(filename, 'w') as f:
            json.dump(resources, f, indent=2)
        
        # Also save as latest.json for production agent
        latest_filename = "/app/frozen_resources/latest.json"
        with open(latest_filename, 'w') as f:
            json.dump(resources, f, indent=2)
        
        print(f"💾 Auto-saved resources to {filename} and {latest_filename} (version {next_version})")
    except Exception as e:
        print(f"⚠️  Failed to auto-save resources: {e}")

    # Broadcast update complete
    await sio.emit('learning_cycle_complete', {
        'cycle': algorithm.current_cycle,
        'resources_updated': True,
        'performance_metrics': new_resources.get('performance_metrics', {})
    })

    print(f"✅ Learning update complete (cycle {algorithm.current_cycle})")
    print(f"{'='*60}\n")


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a crawl job"""
    if request.job_id not in job_queue:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        # Interpret feedback
        job = job_queue[request.job_id]

        interpretation = await gemini_client.interpret_feedback(
            request.feedback,
            context={
                "url": job["task"]["url"],
                "schema": job["task"].get("extraction_schema", {}),  # Expected schema (template)
                "data": job["result"].get("data", []),                # Actual extracted data
                "errors": job["result"].get("error", None)
            }
        )

        # Check if clarification needed
        if interpretation.get("clarification_needed", False):
            return {
                "status": "clarification_needed",
                "question": interpretation.get("clarification_question"),
                "confidence": interpretation.get("confidence", 0.0)
            }

        # Store feedback
        feedback_queue[request.job_id] = {
            "original": request.feedback,
            "interpreted": interpretation,
            "timestamp": asyncio.get_event_loop().time()
        }

        # Track feedback in algorithm
        algorithm.feedback_queue.append({
            'job_id': request.job_id,
            'interpretation': interpretation,
            'timestamp': datetime.now().isoformat()
        })

        job_queue[request.job_id]["awaiting_feedback"] = False

        # Emit Socket.IO update for pending rollouts
        await sio.emit('pending_rollouts_updated', {
            'pending_count': len(algorithm.pending_rollouts),
            'update_frequency': algorithm.update_frequency,
            'cycle': algorithm.current_cycle
        })

        # Broadcast feedback received
        await broadcast_feedback_received(request.job_id, interpretation)

        # Check if N-rollout threshold reached
        print(f"📊 Feedback received. Pending rollouts: {len(algorithm.pending_rollouts)}/{algorithm.update_frequency}")
        if len(algorithm.pending_rollouts) >= algorithm.update_frequency:
            # Trigger learning cycle asynchronously
            asyncio.create_task(trigger_learning_update())

        return {
            "status": "accepted",
            "interpretation": interpretation,
            "quality_rating": interpretation.get("quality_rating", 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()

            # Echo heartbeat
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        active_connections.remove(websocket)


@app.get("/stats")
async def get_stats():
    """Get training agent statistics"""
    return {
        "mode": "training",
        "update_cycle": algorithm.current_cycle,
        "pending_rollouts": len(algorithm.pending_rollouts),
        "pending_feedback": len([j for j in job_queue.values() if j.get("awaiting_feedback")]),
        "total_jobs": len(job_queue),
        "gemini_stats": gemini_client.get_stats(),
        "knowledge_metrics": knowledge_store.get_metrics(),
        "rl_policy": rl_controller.get_policy_summary(),
        "performance_history": algorithm.performance_history[-10:]  # Last 10 cycles
    }


@app.get("/knowledge/patterns")
async def get_patterns():
    """Get learned patterns by domain"""
    return knowledge_store.get_domain_patterns()


@app.post("/knowledge/consolidate")
async def trigger_consolidation():
    """Manually trigger pattern consolidation"""
    try:
        merged_count = await knowledge_store.consolidate_patterns()
        return {
            "status": "success",
            "patterns_merged": merged_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rl/policy")
async def get_rl_policy():
    """Get RL controller policy"""
    return rl_controller.get_policy_summary()


@app.post("/rl/trigger")
async def trigger_rl_decision():
    """Manually trigger RL controller decision"""
    try:
        metrics = knowledge_store.get_metrics()
        action_name, params = await rl_controller.decide_action(metrics)

        return {
            "action": action_name,
            "parameters": params,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/resources")
async def export_resources():
    """Export learned resources to frozen_resources folder for production"""
    try:
        import os
        from datetime import datetime
        
        # Get current resources from knowledge store
        resources = {
            "version": algorithm.current_cycle,
            "frozen_at": datetime.now().isoformat(),
            "extraction_prompt": algorithm._get_default_prompt(),
            "crawl_config": {
                "timeout": 30,
                "wait_for": "networkidle",
                "screenshot": False,
                "max_pages": 50,
                "headless": True
            },
            "domain_patterns": knowledge_store.get_domain_patterns(),
            "performance_history": algorithm.performance_history,
            "total_cycles": algorithm.current_cycle
        }
        
        # Create frozen_resources directory if it doesn't exist
        os.makedirs("frozen_resources", exist_ok=True)
        
        # Save to file
        filename = f"frozen_resources/training_resources_v{algorithm.current_cycle}.json"
        with open(filename, 'w') as f:
            json.dump(resources, f, indent=2)
        
        print(f"💾 Exported resources to {filename}")
        
        return {
            "status": "success",
            "filename": filename,
            "version": algorithm.current_cycle,
            "domain_patterns_count": len(resources["domain_patterns"]),
            "total_cycles": algorithm.current_cycle
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def broadcast_job_completed(job_id: str, result: dict):
    """Broadcast job completion to all Socket.IO clients"""
    message = {
        "type": "job_completed",
        "job_id": job_id,
        "success": result["success"],
        "items_count": len(result.get("data", []))
    }
    await sio.emit('job_completed', message)
    
    # Also broadcast to native WebSocket clients for backward compatibility
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except:
            pass


async def broadcast_feedback_received(job_id: str, interpretation: dict):
    """Broadcast feedback received to all Socket.IO clients"""
    message = {
        "type": "feedback_received",
        "job_id": job_id,
        "quality_rating": interpretation.get("quality_rating", 3)
    }
    await sio.emit('feedback_received', message)
    
    # Also broadcast to native WebSocket clients for backward compatibility
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except:
            pass


# Background task: Start RL controller monitoring
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    # Start RL controller in background
    asyncio.create_task(
        rl_controller.start_monitoring(
            knowledge_store,
            interval_hours=1
        )
    )
    print("✅ RL Controller monitoring started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    rl_controller.stop_monitoring()
    knowledge_store.close()
    print("👋 Training Agent shut down gracefully")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8091)
