"""
Training Hub - Centralized Socket.IO Event Management

This hub consolidates all real-time event broadcasting for the training system,
following the SignalR/Hub pattern for organized event management and room-based targeting.

Rooms:
- admin_{admin_id}: Admin-specific notifications
- training_{job_id}: Job-specific progress updates  
- training_global: System-wide updates
- dashboard: Dashboard client updates
"""

import logging
from typing import Dict, Any, Optional
import socketio

logger = logging.getLogger(__name__)


class TrainingHub:
    """
    Centralized hub for training system Socket.IO events.
    Manages connections, rooms, and event broadcasting with proper targeting.
    """
    
    def __init__(self, sio: socketio.AsyncServer):
        """
        Initialize the training hub with a Socket.IO server instance.
        
        Args:
            sio: AsyncServer instance for Socket.IO communication
        """
        self.sio = sio
        self._register_handlers()
        logger.info("✅ TrainingHub initialized")
    
    def _register_handlers(self):
        """Register Socket.IO event handlers for connection and room management"""
        
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            logger.info(f"🔌 Client connected: {sid}")
            await self.sio.emit('connected', {'sid': sid}, room=sid)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            logger.info(f"🔌 Client disconnected: {sid}")
        
        @self.sio.on('join_room')
        async def handle_join_room(sid, room_name):
            """Handle client joining a room"""
            await self.sio.enter_room(sid, room_name)
            logger.info(f"👥 Client {sid} joined room: {room_name}")
            await self.sio.emit('room_joined', {'room': room_name}, room=sid)
        
        @self.sio.on('leave_room')
        async def handle_leave_room(sid, room_name):
            """Handle client leaving a room"""
            await self.sio.leave_room(sid, room_name)
            logger.info(f"👋 Client {sid} left room: {room_name}")
            await self.sio.emit('room_left', {'room': room_name}, room=sid)
        
        @self.sio.on('subscribe_dashboard')
        async def handle_subscribe_dashboard(sid):
            """Subscribe client to dashboard updates"""
            await self.sio.enter_room(sid, 'dashboard')
            logger.info(f"📊 Client {sid} subscribed to dashboard")
            await self.sio.emit('dashboard_subscribed', {}, room=sid)
        
        @self.sio.on('join_admin_workspace')
        async def handle_join_admin_workspace(sid, admin_id):
            """Join admin-specific workspace"""
            room = f"admin_{admin_id}"
            await self.sio.enter_room(sid, room)
            logger.info(f"👤 Client {sid} joined admin workspace: {admin_id}")
            await self.sio.emit('admin_workspace_joined', {'admin_id': admin_id}, room=sid)
    
    # ========== Queue Events ==========
    
    async def emit_training_queued(self, job_id: str, position: int, admin_id: str):
        """
        Emit event when training job is queued.
        Targets: admin workspace, training_global
        """
        payload = {
            'job_id': job_id,
            'position': position,
            'admin_id': admin_id
        }
        
        # Send to admin and global
        await self.sio.emit('training_queued', payload, room=f"admin_{admin_id}")
        await self.sio.emit('training_queued', payload, room='training_global')
        logger.info(f"📥 Emitted training_queued: {job_id} at position {position}")
    
    async def emit_training_started(self, job_id: str, admin_id: str):
        """
        Emit event when training job starts.
        Targets: admin workspace, job session, training_global
        """
        payload = {
            'job_id': job_id,
            'admin_id': admin_id,
            'status': 'started'
        }
        
        await self.sio.emit('training_started', payload, room=f"admin_{admin_id}")
        await self.sio.emit('training_started', payload, room=f"training_{job_id}")
        await self.sio.emit('training_started', payload, room='training_global')
        logger.info(f"▶️ Emitted training_started: {job_id}")
    
    async def emit_queue_updated(self, queue_status: Dict[str, Any], target_room: Optional[str] = None):
        """
        Emit queue status update.
        Targets: dashboard, training_global, or specific room
        """
        payload = {
            'event': 'queue_status_updated',
            'queue_status': queue_status
        }
        
        if target_room:
            await self.sio.emit('queue_updated', payload, room=target_room)
        else:
            # Broadcast to dashboard and global
            await self.sio.emit('queue_updated', payload, room='dashboard')
            await self.sio.emit('queue_updated', payload, room='training_global')
        
        logger.debug(f"🔄 Emitted queue_updated: {queue_status.get('pending_count', 0)} pending")
    
    # ========== Training Progress Events ==========
    
    async def emit_training_progress(
        self, 
        job_id: str, 
        stage: str, 
        percentage: int, 
        message: str,
        admin_id: Optional[str] = None
    ):
        """
        Emit training progress update.
        Targets: job session, admin workspace (if provided), dashboard
        """
        payload = {
            'job_id': job_id,
            'stage': stage,
            'percentage': percentage,
            'message': message
        }
        
        # Always send to job session and dashboard
        await self.sio.emit('training_progress', payload, room=f"training_{job_id}")
        await self.sio.emit('training_progress', payload, room='dashboard')
        
        # Send to admin if provided
        if admin_id:
            await self.sio.emit('training_progress', payload, room=f"admin_{admin_id}")
        
        logger.info(f"📊 Emitted training_progress: {job_id} - {stage} ({percentage}%)")
    
    async def emit_crawl_complete(self, job_id: str, data: Dict[str, Any], admin_id: Optional[str] = None):
        """
        Emit crawl completion event.
        Targets: job session, admin workspace, dashboard
        """
        payload = {
            'job_id': job_id,
            'stage': 'crawl_complete',
            'data': data
        }
        
        await self.sio.emit('crawl_complete', payload, room=f"training_{job_id}")
        await self.sio.emit('crawl_complete', payload, room='dashboard')
        
        if admin_id:
            await self.sio.emit('crawl_complete', payload, room=f"admin_{admin_id}")
        
        logger.info(f"🕷️ Emitted crawl_complete: {job_id}")
    
    async def emit_learning_complete(
        self, 
        job_id: str, 
        patterns_learned: int,
        admin_id: Optional[str] = None
    ):
        """
        Emit learning completion event.
        Targets: job session, admin workspace, dashboard
        """
        payload = {
            'job_id': job_id,
            'stage': 'learning_complete',
            'patterns_learned': patterns_learned
        }
        
        await self.sio.emit('learning_complete', payload, room=f"training_{job_id}")
        await self.sio.emit('learning_complete', payload, room='dashboard')
        
        if admin_id:
            await self.sio.emit('learning_complete', payload, room=f"admin_{admin_id}")
        
        logger.info(f"🧠 Emitted learning_complete: {job_id} - {patterns_learned} patterns")
    
    # ========== Buffer Events ==========
    
    async def emit_buffer_created(
        self, 
        job_id: str, 
        admin_id: str, 
        buffer_data: Dict[str, Any]
    ):
        """
        Emit buffer created event.
        Targets: admin workspace, dashboard
        """
        payload = {
            'job_id': job_id,
            'admin_id': admin_id,
            'buffer_id': buffer_data.get('buffer_id'),
            'patterns_count': buffer_data.get('patterns_count', 0),
            'ttl_hours': buffer_data.get('ttl_hours', 24)
        }
        
        await self.sio.emit('buffer_created', payload, room=f"admin_{admin_id}")
        await self.sio.emit('buffer_created', payload, room='dashboard')
        
        logger.info(f"💾 Emitted buffer_created: {job_id} for admin {admin_id}")
    
    async def emit_buffer_ready(
        self, 
        job_id: str, 
        admin_id: str, 
        patterns_count: int
    ):
        """
        Emit buffer ready for review event.
        Targets: admin workspace, dashboard
        """
        payload = {
            'job_id': job_id,
            'admin_id': admin_id,
            'patterns_count': patterns_count,
            'status': 'ready_to_commit'
        }
        
        await self.sio.emit('buffer_ready', payload, room=f"admin_{admin_id}")
        await self.sio.emit('buffer_ready', payload, room='dashboard')
        
        logger.info(f"✅ Emitted buffer_ready: {job_id} with {patterns_count} patterns")
    
    # ========== Commit & Version Events ==========
    
    async def emit_commit_progress(
        self, 
        pending_count: int, 
        threshold: int,
        admin_id: Optional[str] = None
    ):
        """
        Emit commit progress update.
        Targets: dashboard, training_global, admin workspace (if provided)
        """
        payload = {
            'pending_count': pending_count,
            'threshold': threshold,
            'commits_needed': threshold - pending_count
        }
        
        await self.sio.emit('commit_progress', payload, room='dashboard')
        await self.sio.emit('commit_progress', payload, room='training_global')
        
        if admin_id:
            await self.sio.emit('commit_progress', payload, room=f"admin_{admin_id}")
        
        logger.info(f"📦 Emitted commit_progress: {pending_count}/{threshold}")
    
    async def emit_version_created(
        self, 
        version: int, 
        patterns_count: int, 
        domains_count: int
    ):
        """
        Emit version creation event.
        Targets: dashboard, training_global (broadcast to all)
        """
        payload = {
            'version': version,
            'patterns_count': patterns_count,
            'domains_count': domains_count,
            'timestamp': None  # Will be set by caller if needed
        }
        
        # Broadcast to everyone
        await self.sio.emit('version_created', payload, room='dashboard')
        await self.sio.emit('version_created', payload, room='training_global')
        
        logger.info(f"🎉 Emitted version_created: v{version} with {patterns_count} patterns")
    
    # ========== Completion Events ==========
    
    async def emit_training_completed(
        self, 
        job_id: str, 
        admin_id: str, 
        result: Dict[str, Any]
    ):
        """
        Emit training completion event.
        Targets: admin workspace, job session, dashboard
        """
        payload = {
            'job_id': job_id,
            'admin_id': admin_id,
            'status': 'completed',
            'result': result
        }
        
        await self.sio.emit('training_completed', payload, room=f"admin_{admin_id}")
        await self.sio.emit('training_completed', payload, room=f"training_{job_id}")
        await self.sio.emit('training_completed', payload, room='dashboard')
        
        logger.info(f"✅ Emitted training_completed: {job_id}")
    
    async def emit_training_failed(
        self, 
        job_id: str, 
        admin_id: str, 
        error: str
    ):
        """
        Emit training failure event.
        Targets: admin workspace, job session, dashboard
        """
        payload = {
            'job_id': job_id,
            'admin_id': admin_id,
            'status': 'failed',
            'error': error
        }
        
        await self.sio.emit('training_failed', payload, room=f"admin_{admin_id}")
        await self.sio.emit('training_failed', payload, room=f"training_{job_id}")
        await self.sio.emit('training_failed', payload, room='dashboard')
        
        logger.error(f"❌ Emitted training_failed: {job_id} - {error}")
    
    async def emit_buffer_discarded(
        self,
        job_id: str,
        admin_id: str
    ):
        """
        Emit buffer discarded event.
        Targets: admin workspace, dashboard
        """
        payload = {
            'job_id': job_id,
            'admin_id': admin_id,
            'status': 'discarded'
        }
        
        await self.sio.emit('buffer_discarded', payload, room=f"admin_{admin_id}")
        await self.sio.emit('buffer_discarded', payload, room='dashboard')
        
        logger.info(f"🗑️ Emitted buffer_discarded: {job_id}")
    
    # ========== Utility Methods ==========
    
    async def get_room_clients(self, room_name: str) -> int:
        """Get number of clients in a room"""
        try:
            room = self.sio.manager.rooms.get(room_name, set())
            return len(room)
        except Exception as e:
            logger.error(f"Error getting room clients: {e}")
            return 0
    
    async def broadcast_system_message(self, message: str, level: str = "info"):
        """Broadcast system-wide message to all clients"""
        payload = {
            'message': message,
            'level': level,
            'type': 'system'
        }
        
        await self.sio.emit('system_message', payload)
        logger.info(f"📢 Broadcasted system message: {message}")
