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
import redis
import os
import glob
import re
import time
from datetime import datetime

from config import Config
from gemini_client import GeminiClient
from agents.shared_crawler_agent import SharedCrawlerAgent
from algorithms.self_improving_algorithm import SelfImprovingCrawlerAlgorithm
from knowledge.hybrid_knowledge_store import HybridKnowledgeStore
from knowledge.rl_controller import RLResourceController
from training_queue_manager import TrainingQueueManager
from training_buffer import TrainingBuffer


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

# Initialize Redis for training queue and buffer
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis-cache"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=False  # Keep bytes for performance
)

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize queue manager and buffer
queue_manager = TrainingQueueManager(redis_client)
training_buffer = TrainingBuffer(redis_client)

print(f"✅ Training queue initialized at version {queue_manager.get_current_version()}")

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


# Background Training Queue Worker
async def training_queue_worker():
    """
    Background worker that processes training queue.
    Only ONE instance processes at a time using distributed lock.
    """
    logger = logging.getLogger(__name__)
    logger.info("🔄 Training queue worker starting...")
    
    while True:
        try:
            # Get distributed lock (only one worker active across all containers)
            lock = redis_client.lock(
                "training:lock:queue_processor",
                timeout=300,  # 5 min max per job
                blocking_timeout=1
            )
            
            if lock.acquire(blocking=False):
                try:
                    # Pop job from queue (blocking with timeout)
                    job_data = redis_client.brpop("training:queue:pending", timeout=5)
                    
                    if job_data:
                        job = json.loads(job_data[1])
                        logger.info(f"🎯 Processing training job {job['job_id']}")
                        
                        # Mark as active
                        queue_manager.mark_job_active(job)
                        
                        # Emit progress
                        await sio.emit('training_started', {
                            'job_id': job['job_id'],
                            'admin_id': job['admin_id']
                        })
                        
                        # Process training
                        try:
                            await process_training_job(job)
                            queue_manager.mark_job_completed(job['job_id'])
                            
                            # Emit completion
                            await sio.emit('training_completed', {
                                'job_id': job['job_id'],
                                'admin_id': job['admin_id'],
                                'status': 'ready_to_commit'
                            })
                            
                            # Emit buffer ready
                            await sio.emit('buffer_ready', {
                                'job_id': job['job_id'],
                                'admin_id': job['admin_id'],
                                'status': 'ready_to_commit'
                            })
                            
                            # Emit queue update
                            queue_status = queue_manager.get_queue_status()
                            await sio.emit('queue_updated', {
                                'event': 'job_completed',
                                'job_id': job['job_id'],
                                'queue_status': queue_status
                            })
                            
                        except Exception as e:
                            logger.error(f"❌ Training job {job['job_id']} failed: {e}", exc_info=True)
                            queue_manager.mark_job_failed(job['job_id'], str(e))
                            
                            # Emit failure
                            await sio.emit('training_failed', {
                                'job_id': job['job_id'],
                                'admin_id': job['admin_id'],
                                'error': str(e)
                            })
                            
                            # Emit queue update
                            queue_status = queue_manager.get_queue_status()
                            await sio.emit('queue_updated', {
                                'event': 'job_failed',
                                'job_id': job['job_id'],
                                'error': str(e),
                                'queue_status': queue_status
                            })
                        
                finally:
                    lock.release()
            
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"❌ Error in queue worker: {e}", exc_info=True)
            await asyncio.sleep(1)


async def process_training_job(job: Dict):
    """
    Execute training and buffer results in Redis.
    """
    job_id = job["job_id"]
    admin_id = job["admin_id"]
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Executing training for job {job_id}")
    
    try:
        # Emit training progress started
        await sio.emit('training_progress', {
            'job_id': job_id,
            'admin_id': admin_id,
            'stage': 'crawling',
            'message': f'Starting crawl for {job["url"]}',
            'progress': 10
        })
        
        # 1. Execute crawl
        task = {
            "url": job["url"],
            "user_description": job["prompt"],
            "extraction_schema": job.get("schema", {}),
            "feedback_from_previous": job.get("feedback_from_previous")
        }
        
        result = await agent.execute_crawl(task)
        
        # Emit crawl completed
        await sio.emit('training_progress', {
            'job_id': job_id,
            'admin_id': admin_id,
            'stage': 'crawl_completed',
            'message': f'Crawl completed. Success: {result["success"]}',
            'progress': 40,
            'items_extracted': len(result.get("data", []))
        })
        
        # 2. Calculate reward
        reward = 0.8 if result["success"] else 0.2
        
        # Emit learning started
        await sio.emit('training_progress', {
            'job_id': job_id,
            'admin_id': admin_id,
            'stage': 'learning',
            'message': 'Analyzing patterns and learning...',
            'progress': 60
        })
        
        # 3. Learn patterns (in memory)
        rollout_data = [{
            'id': job_id,
            'task': task,
            'result': result,
            'reward': reward,
            'metadata': {
                'user_feedback': job.get("feedback_from_previous"),
                'admin_id': admin_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }]
        
        learned_resources = await algorithm.learn_from_interactive_rollouts(rollout_data)
        
        # Emit patterns learned
        await sio.emit('training_progress', {
            'job_id': job_id,
            'admin_id': admin_id,
            'stage': 'patterns_learned',
            'message': 'Patterns extracted and analyzed',
            'progress': 80,
            'patterns_count': sum(len(p) for p in learned_resources.get("domain_patterns", {}).values())
        })
        
        # 4. Buffer crawl result
        training_buffer.store_result(admin_id, job_id, result)
        
        # 5. Buffer patterns in Redis
        domain_patterns = learned_resources.get("domain_patterns", {})
        training_buffer.store_patterns(admin_id, job_id, domain_patterns)
        
        # 6. Buffer metrics
        metrics = {
            "reward": reward,
            "success": result["success"],
            "items_extracted": len(result.get("data", [])),
            "execution_time_ms": result.get("execution_time_ms", 0),
            "base_reward": reward
        }
        training_buffer.store_metrics(admin_id, job_id, metrics)
        
        # 7. Buffer history
        history_entry = {
            "cycle": algorithm.current_cycle,
            "reward": reward,
            "timestamp": datetime.utcnow().isoformat()
        }
        training_buffer.add_history_entry(admin_id, job_id, history_entry)
        
        # 8. Mark buffer ready for commit
        timestamp = datetime.utcnow().isoformat()
        training_buffer.set_metadata(admin_id, job_id, {
            "status": "ready_to_commit",
            "job_id": job_id,
            "admin_id": admin_id,
            "completed_at": timestamp,
            "timestamp": timestamp,
            "url": job["url"],
            "description": job.get("prompt", ""),
            "domains": list(domain_patterns.keys()),
            "patterns_count": sum(len(patterns) for patterns in domain_patterns.values()),
            "ttl_hours": 24
        })
        
        logger.info(f"✅ Training job {job_id} completed and buffered")
        
        # Emit training complete with buffer created
        await sio.emit('training_progress', {
            'job_id': job_id,
            'admin_id': admin_id,
            'stage': 'completed',
            'message': 'Training completed and buffered for review',
            'progress': 100
        })
        
        await sio.emit('buffer_created', {
            'job_id': job_id,
            'admin_id': admin_id,
            'patterns_count': sum(len(patterns) for patterns in domain_patterns.values()),
            'domains': list(domain_patterns.keys()),
            'reward': reward
        })
        
    except Exception as e:
        # Mark buffer as failed
        training_buffer.set_metadata(admin_id, job_id, {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        })
        raise


# App Startup Event
@app.on_event("startup")
async def startup():
    """Start background queue worker on app startup"""
    # Clean up any stale locks from previous containers
    try:
        redis_client.delete("training:lock:queue_processor")
        print("🧹 Cleaned up stale queue processor lock")
    except Exception as e:
        print(f"⚠️ Could not clean lock: {e}")
    
    asyncio.create_task(training_queue_worker())
    print("✅ Training queue worker started")


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


class CommitRequest(BaseModel):
    admin_id: str = "admin"
    feedback: Optional[str] = None


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


@app.post("/train-crawl")
async def train_crawl(request: TrainCrawlRequest, admin_id: str = "admin"):
    """
    Submit training crawl to queue (non-blocking).
    Returns immediately with job_id and queue position.
    """
    job_id = str(uuid.uuid4())

    try:
        # Create training job
        training_job = {
            "job_id": job_id,
            "admin_id": admin_id,
            "url": request.url,
            "prompt": request.user_description,
            "schema": request.extraction_schema or {},
            "feedback_from_previous": request.feedback_from_previous,
            "submitted_at": datetime.utcnow().isoformat()
        }
        
        # Enqueue (returns position in queue)
        position = queue_manager.enqueue_training(training_job)
        
        # Emit to Socket.IO
        await sio.emit('training_queued', {
            'job_id': job_id,
            'position': position,
            'admin_id': admin_id
        })
        
        # Emit queue status update
        queue_status = queue_manager.get_queue_status()
        await sio.emit('queue_updated', {
            'event': 'job_queued',
            'job_id': job_id,
            'queue_status': queue_status
        })
        
        print(f"📥 Training job {job_id} queued at position {position}")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "position": position,
            "message": f"Training job queued at position {position}. Results will be available when processing completes.",
            "queue_status": queue_manager.get_queue_status()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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


@app.post("/commit-training/{job_id}")
async def commit_training(job_id: str, request: CommitRequest):
    """
    Commit buffered training results to a new version.
    Requires distributed lock to ensure atomicity.
    Optional feedback can be included.
    """
    try:
        # Extract admin_id and feedback from request body
        admin_id = request.admin_id
        feedback = request.feedback
        
        # Check if buffer exists
        if not training_buffer.buffer_exists(admin_id, job_id):
            raise HTTPException(status_code=404, detail=f"Training buffer not found for job {job_id}")
        
        # Check buffer status
        status = training_buffer.get_buffer_status(admin_id, job_id)
        if status != "ready_to_commit":
            raise HTTPException(status_code=400, detail=f"Buffer status is '{status}', must be 'ready_to_commit'")
        
        # Acquire version increment lock
        lock = redis_client.lock(
            "training:lock:version_increment",
            timeout=30,
            blocking_timeout=5
        )
        
        with lock:
            # Load buffer data
            buffer_data = training_buffer.get_buffer_data(admin_id, job_id)
            buffered_patterns = buffer_data["patterns"]
            buffered_metrics = buffer_data["metrics"]
            buffered_history = buffer_data["history"]
            
            # Add to pending commits
            pending_commit = {
                "job_id": job_id,
                "admin_id": admin_id,
                "feedback": feedback,
                "patterns": buffered_patterns,
                "metrics": buffered_metrics,
                "history": buffered_history,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            redis_client.rpush("training:pending_commits", json.dumps(pending_commit))
            pending_count = redis_client.llen("training:pending_commits")
            
            # Clean up buffer immediately
            training_buffer.clear_buffer(admin_id, job_id)
            
            logger.info(f"📦 Added job {job_id} to pending commits ({pending_count}/5)")
            
            # Emit commit progress
            await sio.emit('commit_progress', {
                'job_id': job_id,
                'admin_id': admin_id,
                'pending_count': pending_count,
                'threshold': 5,
                'commits_needed': max(0, 5 - pending_count)
            })
            
            # Check if we have 5 pending commits
            if pending_count >= 5:
                try:
                    logger.info(f"🎯 5 commits reached! Creating new version...")
                    
                    # Get current version
                    current_version = queue_manager.get_current_version()
                    next_version = current_version + 1
                    
                    # Get all pending commits
                    pending_commits_raw = redis_client.lrange("training:pending_commits", 0, -1)
                    pending_commits = [json.loads(c) for c in pending_commits_raw]
                    
                    # Load previous version for merging
                    previous_patterns = {}
                    prev_data = {}
                    if current_version > 0:
                        previous_file = f"/app/frozen_resources/training_resources_v{current_version}.json"
                        if os.path.exists(previous_file):
                            with open(previous_file, 'r') as f:
                                prev_data = json.load(f)
                                previous_patterns = prev_data.get("domain_patterns", {})
                    
                    # Merge all pending commits
                    merged_patterns = dict(previous_patterns)
                    merged_history = prev_data.get("performance_history", [])
                    all_feedbacks = []
                    all_job_ids = []
                    
                    for commit in pending_commits:
                        # Merge patterns
                        for domain, patterns in commit["patterns"].items():
                            if domain in merged_patterns:
                                merged_patterns[domain].extend(patterns)
                            else:
                                merged_patterns[domain] = patterns
                        
                        # Merge history
                        merged_history.extend(commit["history"])
                        
                        # Collect feedback and job IDs
                        if commit.get("feedback"):
                            all_feedbacks.append({
                                "job_id": commit["job_id"],
                                "admin_id": commit["admin_id"],
                                "feedback": commit["feedback"]
                            })
                        all_job_ids.append(commit["job_id"])
                    
                    # Create new version object
                    new_version_data = {
                        "version": next_version,
                        "previous_version": current_version if current_version > 0 else None,
                        "frozen_at": datetime.utcnow().isoformat(),
                        "committed_jobs": all_job_ids,  # List of all job IDs from batch
                        "admin_feedbacks": all_feedbacks,  # All feedback from commits
                        "commit_count": len(pending_commits),
                        "extraction_prompt": algorithm._get_default_prompt(),
                        "crawl_config": {
                            "timeout": 30,
                            "wait_for": "networkidle",
                            "screenshot": False,
                            "max_pages": 50,
                            "headless": True
                        },
                        "domain_patterns": merged_patterns,
                        "performance_history": merged_history,
                        "total_cycles": algorithm.current_cycle + prev_data.get("total_cycles", 0),
                        "incremental_learning": {
                            "base_version": current_version,
                            "total_patterns": sum(len(p) for p in merged_patterns.values()),
                            "total_domains": len(merged_patterns)
                        }
                    }
                    
                    # Acquire file write lock
                    file_lock = redis_client.lock(
                        "training:lock:file_write",
                        timeout=10,
                        blocking_timeout=5
                    )
                    
                    with file_lock:
                        # Write to file system
                        version_file = f"/app/frozen_resources/training_resources_v{next_version}.json"
                        latest_file = "/app/frozen_resources/latest.json"
                        
                        os.makedirs("/app/frozen_resources", exist_ok=True)
                        
                        with open(version_file, 'w') as f:
                            json.dump(new_version_data, f, indent=2)
                        
                        with open(latest_file, 'w') as f:
                            json.dump(new_version_data, f, indent=2)
                        
                        # Store version metadata in Redis
                        redis_client.hset(f"training:versions:{next_version}:metadata", mapping={
                            "version": str(next_version),
                            "timestamp": new_version_data["frozen_at"],
                            "commit_count": str(len(pending_commits)),
                            "total_domains": str(len(merged_patterns)),
                            "total_patterns": str(new_version_data["incremental_learning"]["total_patterns"])
                        })
                        
                        # Atomically increment version counter (COMMIT POINT)
                        queue_manager.increment_version()
                        
                        # Clear pending commits
                        redis_client.delete("training:pending_commits")
                        
                        logger.info(f"✅ Created version {next_version} from {len(pending_commits)} commits")
                        
                        # Emit success event
                        await sio.emit('version_committed', {
                            'version': next_version,
                            'job_ids': all_job_ids,
                            'commit_count': len(pending_commits)
                        })
                        
                        # Also emit version_created for UI compatibility
                        await sio.emit('version_created', {
                            'version': next_version,
                            'commit_count': len(pending_commits),
                            'patterns_count': new_version_data["incremental_learning"]["total_patterns"],
                            'domains_count': new_version_data["incremental_learning"]["total_domains"],
                            'timestamp': new_version_data["frozen_at"]
                        })
                        
                        return {
                            "status": "version_created",
                            "version": next_version,
                            "message": f"✅ Created version {next_version} from 5 commits",
                            "file": version_file,
                            "commit_count": len(pending_commits),
                            "metrics": new_version_data["incremental_learning"]
                        }
                        
                except Exception as e:
                    logger.error(f"❌ Failed to create version: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Version creation failed: {str(e)}")
            else:
                # Not enough commits yet, just return pending status
                return {
                    "status": "pending",
                    "message": f"✅ Commit added to staging ({pending_count}/5)",
                    "pending_count": pending_count,
                    "commits_needed": 5 - pending_count
                }
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Commit training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pending-commits/status")
async def get_pending_commits_status():
    """Get status of pending commits awaiting version creation"""
    try:
        pending_count = redis_client.llen("training:pending_commits")
        return {
            "pending_count": pending_count,
            "commits_needed": max(0, 5 - pending_count),
            "threshold": 5,
            "ready_for_version": pending_count >= 5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/status")
async def get_queue_status():
    """Get training queue status"""
    try:
        status = queue_manager.get_queue_status()
        pending_jobs = queue_manager.get_pending_jobs()
        active_jobs = queue_manager.get_active_jobs()
        
        return {
            "summary": status,
            "pending_jobs": pending_jobs[:10],  # First 10
            "active_jobs": active_jobs,
            "pending_count": len(pending_jobs),
            "active_count": len(active_jobs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/pending")
async def get_pending_jobs():
    """Get all pending training jobs"""
    try:
        jobs = queue_manager.get_pending_jobs()
        return {"pending_jobs": jobs, "count": len(jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/buffers/list")
async def list_buffers(admin_id: Optional[str] = None):
    """List all training buffers"""
    try:
        if admin_id:
            buffers = training_buffer.list_all_buffers(admin_id)
        else:
            buffers = training_buffer.list_all_buffers()
        
        return {
            "buffers": buffers,
            "count": len(buffers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/buffers/pending")
async def list_pending_buffers(admin_id: Optional[str] = None):
    """List buffers ready to commit"""
    try:
        buffers = training_buffer.list_pending_buffers(admin_id)
        return {
            "pending_buffers": buffers,
            "count": len(buffers)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/buffer/{job_id}")
async def get_buffer_summary(job_id: str, admin_id: str):
    """Get full buffer data for a specific job"""
    try:
        buffer_data = training_buffer.get_buffer_data(admin_id, job_id)
        if not buffer_data:
            raise HTTPException(status_code=404, detail="Buffer not found")
        
        # Add job_id and admin_id to the response
        buffer_data["job_id"] = job_id
        buffer_data["admin_id"] = admin_id
        
        # Convert patterns dict to list for frontend
        if "patterns" in buffer_data and isinstance(buffer_data["patterns"], dict):
            buffer_data["patterns"] = list(buffer_data["patterns"].values())
        
        # Extract common fields from metadata for easier access
        if "metadata" in buffer_data:
            buffer_data["url"] = buffer_data["metadata"].get("url", "")
            buffer_data["description"] = buffer_data["metadata"].get("description", "")
            buffer_data["timestamp"] = buffer_data["metadata"].get("timestamp", "")
        
        return buffer_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/buffer/{job_id}")
async def discard_buffer(job_id: str, admin_id: str):
    """Discard training buffer (rollback)"""
    try:
        if not training_buffer.buffer_exists(admin_id, job_id):
            raise HTTPException(status_code=404, detail="Buffer not found")
        
        training_buffer.clear_buffer(admin_id, job_id)
        
        await sio.emit('buffer_discarded', {
            'job_id': job_id,
            'admin_id': admin_id
        })
        
        return {
            "status": "discarded",
            "job_id": job_id,
            "message": "Training buffer discarded successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/versions/history")
async def get_version_history():
    """Get training version history"""
    try:
        current = queue_manager.get_current_version()
        
        versions = []
        for v in range(1, current + 1):
            metadata_raw = redis_client.hgetall(f"training:versions:{v}:metadata")
            if metadata_raw:
                metadata = {
                    k.decode() if isinstance(k, bytes) else k: 
                    v.decode() if isinstance(v, bytes) else v
                    for k, v in metadata_raw.items()
                }
                versions.append(metadata)
        
        return {
            "current_version": current,
            "versions": versions,
            "total_versions": len(versions)
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
