"""
Training Queue Manager with Redis
Handles concurrent training requests safely with distributed coordination
"""

import redis
import json
import glob
import re
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrainingQueueManager:
    """
    Manages training queue with Redis for concurrent multi-admin training.
    
    Features:
    - FIFO queue for fair job processing
    - Atomic version counter (no race conditions)
    - Distributed locking for multi-container deployments
    - Queue position tracking
    - Job status monitoring
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.initialize_version()
    
    def initialize_version(self):
        """
        Initialize version counter from existing files.
        This ensures continuity after restarts.
        """
        if not self.redis.exists("training:version:current"):
            # Scan file system for max version
            try:
                files = glob.glob("/app/frozen_resources/training_resources_v*.json")
                max_version = 0
                
                for filepath in files:
                    match = re.search(r'training_resources_v(\d+)\.json', filepath)
                    if match:
                        version = int(match.group(1))
                        if version > max_version:
                            max_version = version
                
                self.redis.set("training:version:current", max_version)
                logger.info(f"✅ Initialized version counter to {max_version} from file system")
            except Exception as e:
                logger.warning(f"Failed to scan for existing versions: {e}. Starting at version 0")
                self.redis.set("training:version:current", 0)
        else:
            current = int(self.redis.get("training:version:current") or 0)
            logger.info(f"✅ Version counter already initialized at {current}")
    
    def enqueue_training(self, job: Dict) -> int:
        """
        Add training job to queue (FIFO).
        
        Args:
            job: Training job dict with job_id, admin_id, url, prompt, schema
        
        Returns:
            Queue position (1 = first in line)
        """
        job["queued_at"] = datetime.utcnow().isoformat()
        job["status"] = "pending"
        
        # Push to pending queue (left push = FIFO with right pop)
        self.redis.lpush(
            "training:queue:pending",
            json.dumps(job)
        )
        
        # Get position (1-indexed)
        position = self.redis.llen("training:queue:pending")
        
        logger.info(f"📥 Enqueued training job {job['job_id']} at position {position}")
        return position
    
    def get_queue_position(self, job_id: str) -> Optional[int]:
        """
        Get job position in queue.
        
        Returns:
            Position (1 = next in line) or None if not in queue
        """
        queue = self.redis.lrange("training:queue:pending", 0, -1)
        
        # Search from right (FIFO - right pop)
        for idx, item in enumerate(reversed(queue)):
            job = json.loads(item)
            if job["job_id"] == job_id:
                return idx + 1  # 1-indexed
        
        return None
    
    def get_queue_status(self) -> Dict:
        """
        Get overall queue status.
        
        Returns:
            Dict with pending, active, completed counts and current version
        """
        pending_count = self.redis.llen("training:queue:pending")
        active_count = self.redis.hlen("training:queue:active")
        completed_count = self.redis.zcard("training:queue:completed")
        current_version = int(self.redis.get("training:version:current") or 0)
        
        return {
            "pending_count": pending_count,
            "active_count": active_count,
            "completed_count": completed_count,
            "current_version": current_version,
            "total_processed": active_count + completed_count
        }
    
    def get_current_version(self) -> int:
        """
        Get current version atomically.
        
        Returns:
            Current version number
        """
        return int(self.redis.get("training:version:current") or 0)
    
    def increment_version(self) -> int:
        """
        Atomically increment version counter.
        Safe for concurrent access.
        
        Returns:
            New version number
        """
        new_version = int(self.redis.incr("training:version:current"))
        logger.info(f"🔼 Version incremented to {new_version}")
        return new_version
    
    def mark_job_active(self, job: Dict):
        """Mark job as actively processing"""
        job["started_at"] = datetime.utcnow().isoformat()
        job["status"] = "active"
        
        self.redis.hset(
            "training:queue:active",
            job["job_id"],
            json.dumps(job)
        )
        
        logger.info(f"🔄 Job {job['job_id']} marked as active")
    
    def mark_job_completed(self, job_id: str):
        """Mark job as completed"""
        # Remove from active
        self.redis.hdel("training:queue:active", job_id)
        
        # Add to completed with timestamp
        self.redis.zadd(
            "training:queue:completed",
            {job_id: datetime.utcnow().timestamp()}
        )
        
        logger.info(f"✅ Job {job_id} marked as completed")
    
    def mark_job_failed(self, job_id: str, error: str):
        """Mark job as failed"""
        # Remove from active
        self.redis.hdel("training:queue:active", job_id)
        
        # Store failure info
        self.redis.hset(
            "training:queue:failed",
            job_id,
            json.dumps({
                "failed_at": datetime.utcnow().isoformat(),
                "error": error
            })
        )
        
        logger.error(f"❌ Job {job_id} marked as failed: {error}")
    
    def get_pending_jobs(self) -> List[Dict]:
        """Get all pending jobs"""
        queue = self.redis.lrange("training:queue:pending", 0, -1)
        return [json.loads(item) for item in queue]
    
    def get_active_jobs(self) -> List[Dict]:
        """Get all active jobs"""
        active = self.redis.hgetall("training:queue:active")
        return [json.loads(v) for v in active.values()]
    
    def get_job_info(self, job_id: str) -> Optional[Dict]:
        """Get job info from any queue"""
        # Check pending
        for item in self.redis.lrange("training:queue:pending", 0, -1):
            job = json.loads(item)
            if job["job_id"] == job_id:
                return {**job, "queue": "pending"}
        
        # Check active
        active_data = self.redis.hget("training:queue:active", job_id)
        if active_data:
            return {**json.loads(active_data), "queue": "active"}
        
        # Check completed
        if self.redis.zscore("training:queue:completed", job_id) is not None:
            timestamp = self.redis.zscore("training:queue:completed", job_id)
            return {
                "job_id": job_id,
                "queue": "completed",
                "completed_at": datetime.fromtimestamp(timestamp).isoformat()
            }
        
        # Check failed
        failed_data = self.redis.hget("training:queue:failed", job_id)
        if failed_data:
            return {**json.loads(failed_data), "queue": "failed"}
        
        return None
    
    def clear_old_completed(self, max_age_hours: int = 24):
        """Clear completed jobs older than max_age_hours"""
        cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        removed = self.redis.zremrangebyscore(
            "training:queue:completed",
            0,
            cutoff
        )
        
        if removed > 0:
            logger.info(f"🧹 Cleared {removed} old completed jobs")
        
        return removed
