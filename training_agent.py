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

from config import Config
from gemini_client import GeminiClient
from agents.shared_crawler_agent import SharedCrawlerAgent
from algorithms.self_improving_algorithm import SelfImprovingCrawlerAlgorithm
from knowledge.hybrid_knowledge_store import HybridKnowledgeStore
from knowledge.rl_controller import RLResourceController

# Initialize
app = FastAPI(title="Training Crawler Agent", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
config = Config()
config.MODE = "training"
config.validate()

# Initialize components
gemini_client = GeminiClient(config.gemini)
rl_controller = RLResourceController(gemini_client, config.training)
knowledge_store = HybridKnowledgeStore(gemini_client, rl_controller, config.knowledge_store)

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

print(f"🎓 Training Agent started on port 8001")
print(f"   Update frequency: Every {config.training.UPDATE_FREQUENCY} rollouts")


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

        # Execute with current resources
        result = await agent.execute_crawl(task)

        # Calculate base reward
        base_reward = 0.8 if result["success"] else 0.2

        # Store job for feedback
        job_queue[job_id] = {
            "task": task,
            "result": result,
            "base_reward": base_reward,
            "awaiting_feedback": True
        }

        # Broadcast to WebSocket clients
        await broadcast_job_completed(job_id, result)

        return CrawlResponse(
            job_id=job_id,
            success=result["success"],
            data=result.get("data", []),
            metadata=result.get("metadata", {}),
            base_reward=base_reward,
            error=result.get("error")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                "fields": job["task"].get("extraction_schema", {}),
                "data": job["result"].get("data", [])
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

        job_queue[request.job_id]["awaiting_feedback"] = False

        # Broadcast feedback received
        await broadcast_feedback_received(request.job_id, interpretation)

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


async def broadcast_job_completed(job_id: str, result: dict):
    """Broadcast job completion to all WebSocket clients"""
    message = json.dumps({
        "type": "job_completed",
        "job_id": job_id,
        "success": result["success"],
        "items_count": len(result.get("data", []))
    })

    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            pass


async def broadcast_feedback_received(job_id: str, interpretation: dict):
    """Broadcast feedback received to all WebSocket clients"""
    message = json.dumps({
        "type": "feedback_received",
        "job_id": job_id,
        "quality_rating": interpretation.get("quality_rating", 3)
    })

    for connection in active_connections:
        try:
            await connection.send_text(message)
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
