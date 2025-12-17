// WebSocket client for real-time updates from training agent

import { io, Socket } from 'socket.io-client';
import type { WebSocketMessage } from '../types';

const WS_URL = process.env.REACT_APP_TRAINING_SERVICE_URL || 'http://localhost:8001';

// Constants
const RECONNECTION_DELAY = 1000;
const MAX_RECONNECTION_DELAY = 5000;
const MAX_RECONNECT_ATTEMPTS = 5;

class WebSocketService {
  private socket: Socket | null = null;
  private messageHandlers: Map<string, Set<(data: any) => void>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = MAX_RECONNECT_ATTEMPTS;

  connect() {
    if (this.socket?.connected) {
      return;
    }

    this.socket = io(WS_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: RECONNECTION_DELAY,
      reconnectionDelayMax: MAX_RECONNECTION_DELAY,
    });

    this.socket.on('connect', () => {
      if (process.env.NODE_ENV === 'development') {
        console.log('WebSocket connected');
      }
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', () => {
      if (process.env.NODE_ENV === 'development') {
        console.log('WebSocket disconnected');
      }
    });

    this.socket.on('message', (message: WebSocketMessage) => {
      this.handleMessage(message);
    });

    // Listen for new Socket.IO events
    this.socket.on('crawl_log', (log: any) => {
      this.handleMessage({ type: 'crawl_log', ...log });
    });

    this.socket.on('crawl_started', (data: any) => {
      this.handleMessage({ type: 'crawl_started', ...data });
    });

    this.socket.on('pending_rollouts_updated', (data: any) => {
      this.handleMessage({ type: 'pending_rollouts_updated', ...data });
    });

    this.socket.on('learning_cycle_complete', (data: any) => {
      this.handleMessage({ type: 'learning_cycle_complete', ...data });
    });

    // Queue management events
    this.socket.on('training_queued', (data: any) => {
      this.handleMessage({ type: 'training_queued', ...data });
    });

    this.socket.on('training_started', (data: any) => {
      this.handleMessage({ type: 'training_started', ...data });
    });

    this.socket.on('training_completed', (data: any) => {
      this.handleMessage({ type: 'training_completed', ...data });
    });

    this.socket.on('training_failed', (data: any) => {
      this.handleMessage({ type: 'training_failed', ...data });
    });

    this.socket.on('version_committed', (data: any) => {
      this.handleMessage({ type: 'version_committed', ...data });
    });

    this.socket.on('buffer_discarded', (data: any) => {
      this.handleMessage({ type: 'buffer_discarded', ...data });
    });

    this.socket.on('connect_error', (error) => {
      if (process.env.NODE_ENV === 'development') {
        console.error('WebSocket connection error:', error);
      }
      this.reconnectAttempts++;
      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        if (process.env.NODE_ENV === 'development') {
          console.error('Max reconnection attempts reached');
        }
      }
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.messageHandlers.clear();
    this.reconnectAttempts = 0;
  }

  on(eventType: string, handler: (data: any) => void) {
    if (!this.messageHandlers.has(eventType)) {
      this.messageHandlers.set(eventType, new Set());
    }
    this.messageHandlers.get(eventType)!.add(handler);
  }

  off(eventType: string, handler: (data: any) => void) {
    const handlers = this.messageHandlers.get(eventType);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  private handleMessage(message: WebSocketMessage) {
    const handlers = this.messageHandlers.get(message.type);
    if (handlers) {
      handlers.forEach((handler) => handler(message));
    }

    // Also trigger 'all' handlers
    const allHandlers = this.messageHandlers.get('all');
    if (allHandlers) {
      allHandlers.forEach((handler) => handler(message));
    }
  }

  isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

export const wsService = new WebSocketService();
export default wsService;
