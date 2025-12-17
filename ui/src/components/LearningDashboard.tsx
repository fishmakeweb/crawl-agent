// Dashboard component for visualizing learning progress and statistics

import React, { useEffect, useState, useCallback } from 'react';
import type { TrainingStats } from '../types';
import { trainingApi } from '../services/api';
import wsService from '../services/websocket';

export const LearningDashboard: React.FC = () => {
  const [stats, setStats] = useState<TrainingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [queueStats, setQueueStats] = useState({ pending: 0, active: 0, completed: 0 });
  const [commitProgress, setCommitProgress] = useState({ count: 0, threshold: 5 });
  const [bufferStats, setBufferStats] = useState({ pending: 0, total: 0 });
  const [latestVersion, setLatestVersion] = useState<number | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const loadStats = useCallback(async () => {
    try {
      const data = await trainingApi.getStats();
      setStats(data);
      setError(null);
      setLoading(false);
    } catch (err) {
      if (err instanceof Error) {
        setError('Failed to load statistics');
        if (process.env.NODE_ENV === 'development') {
          console.error(err);
        }
      }
      setLoading(false);
    }
  }, []);

  const loadInitialData = useCallback(async () => {
    try {
      // Load queue status
      const queueStatus = await trainingApi.getQueueStatus();
      setQueueStats({
        pending: queueStatus.summary?.pending_count || 0,
        active: queueStatus.summary?.active_count || 0,
        completed: queueStatus.summary?.completed_count || 0
      });

      // Load commit progress
      const commitStatus = await trainingApi.getPendingCommitsStatus();
      setCommitProgress({
        count: commitStatus.pending_count || 0,
        threshold: commitStatus.threshold || 5
      });

      // Load buffer stats
      const [allBuffers, pendingBuffers] = await Promise.all([
        trainingApi.listBuffers(),
        trainingApi.getPendingBuffers()
      ]);
      setBufferStats({
        pending: pendingBuffers.length,
        total: allBuffers.length
      });

      // Load latest version
      const versionData = await trainingApi.getVersionHistory();
      setLatestVersion(versionData.current_version);
    } catch (err) {
      if (process.env.NODE_ENV === 'development') {
        console.error('Failed to load initial data:', err);
      }
    }
  }, []);

  useEffect(() => {
    // Load initial stats and real-time data
    loadStats();
    loadInitialData();

    // Real-time event handlers
    const handleQueueUpdate = (data: Record<string, unknown>) => {
      if (data.queue_status) {
        const queueStatus = data.queue_status as Record<string, unknown>;
        setQueueStats({
          pending: (queueStatus.pending_count as number) || 0,
          active: (queueStatus.active_count as number) || 0,
          completed: (queueStatus.completed_count as number) || 0
        });
      }
      setLastUpdate(new Date());
    };

    const handleCommitProgress = (data: Record<string, unknown>) => {
      setCommitProgress({
        count: (data.pending_count as number) || 0,
        threshold: (data.threshold as number) || 5
      });
      setLastUpdate(new Date());
    };

    const handleBufferCreated = () => {
      setBufferStats(prev => ({
        pending: prev.pending + 1,
        total: prev.total + 1
      }));
      setLastUpdate(new Date());
    };

    const handleBufferReady = () => {
      // Buffer is ready for review (already counted in created)
      setLastUpdate(new Date());
    };

    const handleTrainingCompleted = () => {
      // Refresh stats when training completes
      loadStats();
      setLastUpdate(new Date());
    };

    const handleVersionCreated = (data: Record<string, unknown>) => {
      // Refresh stats when new version is created
      loadStats();
      setCommitProgress({ count: 0, threshold: 5 }); // Reset commit progress
      // Update latest version from event
      if (data.version) {
        setLatestVersion(data.version as number);
      }
      // Reset buffer pending count when version created (buffers were committed)
      setBufferStats(prev => ({
        pending: 0,
        total: prev.total
      }));
      setLastUpdate(new Date());
    };

    // Subscribe to real-time events
    wsService.on('queue_updated', handleQueueUpdate);
    wsService.on('commit_progress', handleCommitProgress);
    wsService.on('buffer_created', handleBufferCreated);
    wsService.on('buffer_ready', handleBufferReady);
    wsService.on('training_completed', handleTrainingCompleted);
    wsService.on('version_created', handleVersionCreated);

    return () => {
      // Cleanup event listeners
      wsService.off('queue_updated', handleQueueUpdate);
      wsService.off('commit_progress', handleCommitProgress);
      wsService.off('buffer_created', handleBufferCreated);
      wsService.off('buffer_ready', handleBufferReady);
      wsService.off('training_completed', handleTrainingCompleted);
      wsService.off('version_created', handleVersionCreated);
    };
  }, [loadStats, loadInitialData]);

  if (loading) {
    return <div className="dashboard-loading">Loading statistics...</div>;
  }

  if (error) {
    return <div className="dashboard-error">{error}</div>;
  }

  if (!stats) {
    return <div className="dashboard-empty">No statistics available</div>;
  }

  const { mode, knowledge_metrics } = stats;

  return (
    <div className="learning-dashboard">
      <div className="dashboard-header">
        <h2>🔴 Real-Time Learning Dashboard</h2>
        <span className={`mode-badge ${mode}`}>{mode.toUpperCase()} MODE</span>
      </div>

      <div className="stats-grid">
        <div className="stat-card realtime">
          <h4>🔄 Training Queue</h4>
          <div className="stat-value">{queueStats.pending}</div>
          <div className="stat-label">Pending Jobs</div>
          <div className="stat-detail">
            Active: {queueStats.active} | Completed: {queueStats.completed}
          </div>
        </div>

        <div className="stat-card realtime">
          <h4>📦 Commit Progress</h4>
          <div className="stat-value">{commitProgress.count}/{commitProgress.threshold}</div>
          <div className="stat-label">Pending Commits</div>
          <div className="rollout-progress">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${(commitProgress.count / commitProgress.threshold) * 100}%` }}
              />
            </div>
            <div className="stat-detail">
              {commitProgress.threshold - commitProgress.count} more needed for version
            </div>
          </div>
        </div>

        <div className="stat-card realtime">
          <h4>💾 Buffer Status</h4>
          <div className="stat-value">{bufferStats.pending}/{bufferStats.total}</div>
          <div className="stat-label">Pending Review</div>
          <div className="stat-detail">
            {bufferStats.pending} buffers awaiting admin review
          </div>
        </div>

        <div className="stat-card">
          <h4>📚 Knowledge Store</h4>
          <div className="stat-value">{knowledge_metrics.total_patterns}</div>
          <div className="stat-label">Learned Patterns</div>
          <div className="stat-detail">
            Vector Size: {knowledge_metrics.vector_size_mb.toFixed(2)} MB
          </div>
          <div className="stat-detail">
            Graph Nodes: {knowledge_metrics.graph_nodes}
          </div>
          <div className="stat-detail">
            Cache Hit Rate: {(knowledge_metrics.cache_hit_rate * 100).toFixed(1)}%
          </div>
          {latestVersion && (
            <div className="stat-detail">
              Latest Version: v{latestVersion}
            </div>
          )}
        </div>
      </div>

      <div className="dashboard-actions">
        <button 
          className="export-button"
          onClick={async () => {
            try {
              const result = await trainingApi.exportResources();
              alert(`✅ Resources exported successfully!\n\nFile: ${result.filename}\nVersion: ${result.version}\nDomains: ${result.domain_patterns_count}\nCycles: ${result.total_cycles}`);
            } catch (err) {
              alert(`❌ Export failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
            }
          }}
        >
          📦 Export Resources
        </button>
      </div>

      <div className="dashboard-footer">
        <div className="realtime-indicator">
          <span className="live-dot"></span>
          <span>Live Updates</span>
        </div>
        <p>Last updated: {lastUpdate.toLocaleTimeString()}</p>
      </div>
    </div>
  );
};
