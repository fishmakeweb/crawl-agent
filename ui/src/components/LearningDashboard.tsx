// Dashboard component for visualizing learning progress and statistics

import React, { useEffect, useState, useCallback } from 'react';
import type { TrainingStats, LearningInsights } from '../types';
import { trainingApi } from '../services/api';
import wsService from '../services/websocket';
import './LearningDashboard.css';

export const LearningDashboard: React.FC = () => {
  const [stats, setStats] = useState<TrainingStats | null>(null);
  const [insights, setInsights] = useState<LearningInsights | null>(null);
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

      // Load learning insights
      const insightsData = await trainingApi.getLearningInsights();
      if (process.env.NODE_ENV === 'development') {
        console.log('Insights loaded:', insightsData);
      }
      setInsights(insightsData);
    } catch (err) {
      if (process.env.NODE_ENV === 'development') {
        console.error('Failed to load initial data:', err);
      }
    }
  }, []);

  useEffect(() => {
    // Join dashboard room for targeted updates
    wsService.subscribeDashboard();

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

      {/* AI Learning Insights Section */}
      {insights && insights.summary && (
        <div className="insights-section">
          <h3 className="section-title">🧠 AI Learning Insights</h3>
          
          {/* Show message if no learning data yet */}
          {insights.summary.total_patterns === 0 ? (
            <div className="no-insights">
              <p>🎓 No learning data yet. Submit training jobs to start learning!</p>
            </div>
          ) : (
            <>
          {/* Domain Expertise */}
          {insights.domain_expertise && insights.domain_expertise.length > 0 && (
            <div className="expertise-container">
              <h4>🎓 Domain Expertise (Top 10)</h4>
              <div className="expertise-table">
                <div className="table-header">
                  <div className="col-domain">Domain</div>
                  <div className="col-patterns">Patterns</div>
                  <div className="col-success">Success Rate</div>
                  <div className="col-usage">Usage</div>
                  <div className="col-confidence">Confidence</div>
                </div>
                {insights.domain_expertise.slice(0, 10).map((domain, idx) => (
                  <div key={idx} className="table-row">
                    <div className="col-domain">
                      <span className="domain-rank">#{idx + 1}</span>
                      <span className="domain-name">{domain.domain}</span>
                    </div>
                    <div className="col-patterns">{domain.pattern_count}</div>
                    <div className="col-success">
                      <div className="success-badge" style={{
                        backgroundColor: domain.avg_success_rate >= 0.7 ? '#dcfce7' : domain.avg_success_rate >= 0.5 ? '#fef3c7' : '#fee2e2',
                        color: domain.avg_success_rate >= 0.7 ? '#166534' : domain.avg_success_rate >= 0.5 ? '#92400e' : '#991b1b'
                      }}>
                        {(domain.avg_success_rate * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="col-usage">{domain.total_usage}×</div>
                    <div className="col-confidence">
                      <span className={`confidence-badge ${domain.confidence}`}>
                        {domain.confidence === 'high' ? '🟢 High' : domain.confidence === 'medium' ? '🟡 Medium' : '🔴 Low'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Domain Distribution Chart */}
          {insights.domain_distribution && insights.domain_distribution.length > 0 && (
            <div className="domain-chart-container">
              <h4>🌐 Pattern Distribution by Domain</h4>
              <div className="domain-chart">
                {insights.domain_distribution.map((domain, idx) => {
                  const maxPatterns = Math.max(...(insights.domain_distribution || []).map(d => d.patterns));
                  const percentage = (domain.patterns / maxPatterns) * 100;
                  const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];
                  
                  return (
                    <div key={idx} className="domain-bar-item">
                      <div className="domain-bar-label">
                        <span className="domain-bar-name">{domain.domain}</span>
                        <span className="domain-bar-count">{domain.patterns} patterns</span>
                      </div>
                      <div className="domain-bar-track">
                        <div 
                          className="domain-bar-fill"
                          style={{
                            width: `${percentage}%`,
                            backgroundColor: colors[idx % colors.length]
                          }}
                        >
                          <span className="domain-bar-percent">{(domain.success_rate * 100).toFixed(0)}% success</span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Success Rate Distribution */}
          {insights.success_distribution && (
            <div className="success-dist-container">
              <h4>📊 Learning Quality Distribution</h4>
              <div className="success-dist-chart">
                <div className="success-dist-item">
                  <div className="success-dist-bar" style={{
                    width: `${insights.success_distribution.excellent}`,
                    maxWidth: '100%',
                    backgroundColor: '#10b981'
                  }}>
                    <span className="success-dist-label">Excellent (80%+)</span>
                    <span className="success-dist-value">{insights.success_distribution.excellent}</span>
                  </div>
                </div>
                <div className="success-dist-item">
                  <div className="success-dist-bar" style={{
                    width: `${insights.success_distribution.good}`,
                    maxWidth: '100%',
                    backgroundColor: '#3b82f6'
                  }}>
                    <span className="success-dist-label">Good (60-80%)</span>
                    <span className="success-dist-value">{insights.success_distribution.good}</span>
                  </div>
                </div>
                <div className="success-dist-item">
                  <div className="success-dist-bar" style={{
                    width: `${insights.success_distribution.moderate}`,
                    maxWidth: '100%',
                    backgroundColor: '#f59e0b'
                  }}>
                    <span className="success-dist-label">Moderate (40-60%)</span>
                    <span className="success-dist-value">{insights.success_distribution.moderate}</span>
                  </div>
                </div>
                <div className="success-dist-item">
                  <div className="success-dist-bar" style={{
                    width: `${insights.success_distribution.poor}`,
                    maxWidth: '100%',
                    backgroundColor: '#ef4444'
                  }}>
                    <span className="success-dist-label">Needs Improvement (&lt;40%)</span>
                    <span className="success-dist-value">{insights.success_distribution.poor}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Learning Performance Trend */}
          {insights.recent_performance.length > 0 && (
            <div className="performance-container">
              <h4>📈 Recent Learning Performance</h4>
              <div className="performance-chart">
                {insights.recent_performance.map((perf, idx) => {
                  const maxReward = Math.max(...insights.recent_performance.map(p => p.avg_reward));
                  const heightPercent = (perf.avg_reward / maxReward) * 100;
                  
                  return (
                    <div key={idx} className="performance-bar-container">
                      <div 
                        className="performance-bar"
                        style={{ height: `${heightPercent}%` }}
                        title={`Cycle ${perf.cycle}: ${(perf.avg_reward * 100).toFixed(1)}%`}
                      />
                      <div className="performance-label">C{perf.cycle}</div>
                    </div>
                  );
                })}
              </div>
              <div className="performance-legend">
                <span>Last {insights.recent_performance.length} learning cycles</span>
                <span className="performance-note">Higher bars = Better performance</span>
              </div>
            </div>
          )}

          {/* Storage Metrics */}
          {insights.storage_metrics && (
            <div className="storage-container">
              <h4>💾 Knowledge Store Metrics</h4>
              <div className="storage-grid">
                <div className="storage-card">
                  <div className="storage-icon">📦</div>
                  <div className="storage-info">
                    <div className="storage-label">Vector Size</div>
                    <div className="storage-value">{insights.storage_metrics.vector_size_mb} MB</div>
                  </div>
                </div>
                <div className="storage-card">
                  <div className="storage-icon">🔗</div>
                  <div className="storage-info">
                    <div className="storage-label">Graph Nodes</div>
                    <div className="storage-value">{insights.storage_metrics.graph_nodes}</div>
                  </div>
                </div>
                <div className="storage-card">
                  <div className="storage-icon">↔️</div>
                  <div className="storage-info">
                    <div className="storage-label">Relationships</div>
                    <div className="storage-value">{insights.storage_metrics.graph_relationships}</div>
                  </div>
                </div>
                <div className="storage-card">
                  <div className="storage-icon">🎯</div>
                  <div className="storage-info">
                    <div className="storage-label">Stored Patterns</div>
                    <div className="storage-value">{insights.storage_metrics.total_stored_patterns}</div>
                  </div>
                </div>
                <div className="storage-card">
                  <div className="storage-icon">📊</div>
                  <div className="storage-info">
                    <div className="storage-label">Redundancy</div>
                    <div className="storage-value">{insights.storage_metrics.pattern_redundancy}%</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Pattern Types */}
          {insights.pattern_types && Object.keys(insights.pattern_types).length > 0 && (
            <div className="pattern-types-container">
              <h4>🔍 Pattern Types Learned</h4>
              <div className="pattern-types-grid">
                {Object.entries(insights.pattern_types).map(([type, count]) => (
                  <div key={type} className="pattern-type-card">
                    <div className="pattern-type-name">{type}</div>
                    <div className="pattern-type-count">{count}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
            </>
          )}
        </div>
      )}

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
