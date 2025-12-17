"""
Training Buffer Management
Stores intermediate training results in Redis before commit
"""

import redis
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TrainingBuffer:
    """
    Manages buffered training results in Redis.
    
    Buffers allow:
    - Storing training results before commit
    - Admin review of results before version creation
    - Discarding bad training runs
    - Rollback capability
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def store_patterns(self, admin_id: str, job_id: str, patterns: Dict[str, List]):
        """
        Store learned domain patterns in buffer.
        
        Args:
            admin_id: Admin who submitted the training
            job_id: Training job identifier
            patterns: Dict of {domain: [pattern_list]}
        """
        key = f"training:buffer:{admin_id}:{job_id}:patterns"
        
        # Serialize each domain's pattern list
        serialized = {
            domain: json.dumps(pattern_list)
            for domain, pattern_list in patterns.items()
        }
        
        if serialized:
            self.redis.hset(key, mapping=serialized)
            
            # Set expiry (24 hours)
            self.redis.expire(key, 86400)
            
            logger.info(f"📦 Stored patterns for job {job_id}: {len(patterns)} domains")
    
    def store_result(self, admin_id: str, job_id: str, result: Dict[str, Any]):
        """
        Store crawl result data in buffer.
        
        Args:
            admin_id: Admin who submitted the training
            job_id: Training job identifier
            result: Crawl result dict with success, data, error, etc.
        """
        key = f"training:buffer:{admin_id}:{job_id}:result"
        self.redis.set(key, json.dumps(result))
        self.redis.expire(key, 86400)
        logger.info(f"💾 Stored result for job {job_id}")
    
    def store_metrics(self, admin_id: str, job_id: str, metrics: Dict[str, Any]):
        """
        Store performance metrics in buffer.
        
        Args:
            admin_id: Admin who submitted the training
            job_id: Training job identifier
            metrics: Dict of performance metrics
        """
        key = f"training:buffer:{admin_id}:{job_id}:metrics"
        
        # Convert all values to strings for Redis
        serialized = {
            k: json.dumps(v) if not isinstance(v, (str, int, float)) else str(v)
            for k, v in metrics.items()
        }
        
        self.redis.hset(key, mapping=serialized)
        self.redis.expire(key, 86400)
        
        logger.info(f"📊 Stored metrics for job {job_id}")
    
    def add_history_entry(self, admin_id: str, job_id: str, entry: Dict):
        """
        Add performance history entry to buffer.
        
        Args:
            admin_id: Admin who submitted the training
            job_id: Training job identifier
            entry: History entry dict (cycle, reward, timestamp, etc.)
        """
        key = f"training:buffer:{admin_id}:{job_id}:history"
        
        self.redis.lpush(key, json.dumps(entry))
        self.redis.expire(key, 86400)
        
        logger.debug(f"📈 Added history entry for job {job_id}")
    
    def set_metadata(self, admin_id: str, job_id: str, metadata: Dict):
        """
        Set buffer metadata (status, timestamps, etc.).
        
        Args:
            admin_id: Admin who submitted the training
            job_id: Training job identifier
            metadata: Metadata dict (status, completed_at, etc.)
        """
        key = f"training:buffer:{admin_id}:{job_id}:metadata"
        
        metadata["updated_at"] = datetime.utcnow().isoformat()
        
        # Serialize complex values
        serialized = {
            k: json.dumps(v) if not isinstance(v, (str, int, float)) else str(v)
            for k, v in metadata.items()
        }
        
        self.redis.hset(key, mapping=serialized)
        self.redis.expire(key, 86400)
        
        logger.info(f"ℹ️ Set metadata for job {job_id}: status={metadata.get('status')}")
    
    def get_buffer_data(self, admin_id: str, job_id: str) -> Dict:
        """
        Get all buffer data for a training job.
        
        Args:
            admin_id: Admin who submitted the training
            job_id: Training job identifier
        
        Returns:
            Dict with patterns, metrics, history, metadata, result
        """
        base_key = f"training:buffer:{admin_id}:{job_id}"
        
        # Get crawl result
        result_raw = self.redis.get(f"{base_key}:result")
        result = {}
        if result_raw:
            result = json.loads(result_raw.decode() if isinstance(result_raw, bytes) else result_raw)
        
        # Get patterns
        patterns_raw = self.redis.hgetall(f"{base_key}:patterns")
        patterns = {}
        if patterns_raw:
            patterns = {
                k.decode() if isinstance(k, bytes) else k: json.loads(v.decode() if isinstance(v, bytes) else v)
                for k, v in patterns_raw.items()
            }
        
        # Get metrics
        metrics_raw = self.redis.hgetall(f"{base_key}:metrics")
        metrics = {}
        if metrics_raw:
            metrics = {
                k.decode() if isinstance(k, bytes) else k: self._deserialize_value(v)
                for k, v in metrics_raw.items()
            }
        
        # Get history
        history_raw = self.redis.lrange(f"{base_key}:history", 0, -1)
        history = []
        if history_raw:
            history = [
                json.loads(h.decode() if isinstance(h, bytes) else h)
                for h in history_raw
            ]
        
        # Get metadata
        metadata_raw = self.redis.hgetall(f"{base_key}:metadata")
        metadata = {}
        if metadata_raw:
            metadata = {
                k.decode() if isinstance(k, bytes) else k: self._deserialize_value(v)
                for k, v in metadata_raw.items()
            }
        
        return {
            "result": result,
            "patterns": patterns,
            "metrics": metrics,
            "history": history,
            "metadata": metadata
        }
    
    def _deserialize_value(self, value: Any) -> Any:
        """Helper to deserialize Redis values"""
        if isinstance(value, bytes):
            value = value.decode()
        
        # Try to parse as JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    
    def buffer_exists(self, admin_id: str, job_id: str) -> bool:
        """Check if buffer exists for a job"""
        key = f"training:buffer:{admin_id}:{job_id}:metadata"
        return self.redis.exists(key) > 0
    
    def get_buffer_status(self, admin_id: str, job_id: str) -> Optional[str]:
        """Get buffer status"""
        key = f"training:buffer:{admin_id}:{job_id}:metadata"
        status = self.redis.hget(key, "status")
        
        if status:
            return status.decode() if isinstance(status, bytes) else status
        return None
    
    def clear_buffer(self, admin_id: str, job_id: str):
        """
        Clear all buffer data for a training job.
        Used after commit or when discarding results.
        """
        base_key = f"training:buffer:{admin_id}:{job_id}"
        
        deleted = self.redis.delete(
            f"{base_key}:patterns",
            f"{base_key}:metrics",
            f"{base_key}:history",
            f"{base_key}:metadata"
        )
        
        logger.info(f"🗑️ Cleared buffer for job {job_id} ({deleted} keys deleted)")
        return deleted
    
    def list_pending_buffers(self, admin_id: Optional[str] = None) -> List[Dict]:
        """
        List all pending buffers ready for commit.
        
        Args:
            admin_id: Optional filter by admin (None = all admins)
        
        Returns:
            List of buffer metadata dicts
        """
        pattern = f"training:buffer:{admin_id or '*'}:*:metadata"
        
        buffers = []
        for key in self.redis.scan_iter(pattern):
            metadata_raw = self.redis.hgetall(key)
            
            if not metadata_raw:
                continue
            
            # Deserialize metadata
            metadata = {
                k.decode() if isinstance(k, bytes) else k: self._deserialize_value(v)
                for k, v in metadata_raw.items()
            }
            
            # Only include ready_to_commit buffers
            if metadata.get("status") == "ready_to_commit":
                buffers.append(metadata)
        
        logger.info(f"📋 Found {len(buffers)} pending buffers")
        return buffers
    
    def list_all_buffers(self, admin_id: Optional[str] = None) -> List[Dict]:
        """
        List all buffers (any status).
        
        Args:
            admin_id: Optional filter by admin
        
        Returns:
            List of buffer metadata dicts
        """
        pattern = f"training:buffer:{admin_id or '*'}:*:metadata"
        
        buffers = []
        for key in self.redis.scan_iter(pattern):
            metadata_raw = self.redis.hgetall(key)
            
            if not metadata_raw:
                continue
            
            metadata = {
                k.decode() if isinstance(k, bytes) else k: self._deserialize_value(v)
                for k, v in metadata_raw.items()
            }
            
            buffers.append(metadata)
        
        return buffers
    
    def get_buffer_summary(self, admin_id: str, job_id: str) -> Optional[Dict]:
        """
        Get summary of buffer contents (counts, not full data).
        
        Returns:
            Summary dict or None if buffer doesn't exist
        """
        if not self.buffer_exists(admin_id, job_id):
            return None
        
        base_key = f"training:buffer:{admin_id}:{job_id}"
        
        pattern_count = self.redis.hlen(f"{base_key}:patterns")
        history_count = self.redis.llen(f"{base_key}:history")
        
        metadata_raw = self.redis.hgetall(f"{base_key}:metadata")
        metadata = {
            k.decode() if isinstance(k, bytes) else k: self._deserialize_value(v)
            for k, v in metadata_raw.items()
        } if metadata_raw else {}
        
        return {
            "job_id": job_id,
            "admin_id": admin_id,
            "domain_count": pattern_count,
            "history_entries": history_count,
            "status": metadata.get("status"),
            "completed_at": metadata.get("completed_at"),
            "updated_at": metadata.get("updated_at")
        }
