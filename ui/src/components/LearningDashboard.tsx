// Dashboard component for visualizing learning progress and statistics

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { TrainingStats, PendingRolloutsUpdate } from '../types';
import { trainingApi } from '../services/api';
import wsService from '../services/websocket';

// Constants
const STATS_REFRESH_INTERVAL = 5000;

export const LearningDashboard: React.FC = () => {
  const [stats, setStats] = useState<TrainingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingCount, setPendingCount] = useState<number>(0);
  const [maxRollouts, setMaxRollouts] = useState<number>(5);
  const [currentCycle, setCurrentCycle] = useState<number>(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const isLoadingRef = useRef(false);

  const loadStats = useCallback(async () => {
    // Prevent overlapping requests
    if (isLoadingRef.current) {
      return;
    }

    isLoadingRef.current = true;

    // Cancel previous request if still pending
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();

    try {
      const data = await trainingApi.getStats(abortControllerRef.current.signal);
      setStats(data);
      setError(null);
    } catch (err) {
      // Don't show error if request was aborted
      if (err instanceof Error && err.name !== 'AbortError' && err.name !== 'CanceledError') {
        setError('Failed to load statistics');
        if (process.env.NODE_ENV === 'development') {
          console.error(err);
        }
      }
    } finally {
      setLoading(false);
      isLoadingRef.current = false;
    }
  }, []);

  useEffect(() => {
    loadStats();

    // Poll for updates, ensuring previous request completes first
    const interval = setInterval(() => {
      loadStats();
    }, STATS_REFRESH_INTERVAL);

    // Listen for real-time rollout updates
    const handleRolloutUpdate = (data: any) => {
      const update = data as PendingRolloutsUpdate;
      setPendingCount(update.pending_count);
      setMaxRollouts(update.update_frequency);
      setCurrentCycle(update.cycle);
    };

    wsService.on('pending_rollouts_updated', handleRolloutUpdate);

    return () => {
      clearInterval(interval);
      wsService.off('pending_rollouts_updated', handleRolloutUpdate);
      // Cancel any pending requests on unmount
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [loadStats]);

  if (loading) {
    return <div className="dashboard-loading">Loading statistics...</div>;
  }

  if (error) {
    return <div className="dashboard-error">{error}</div>;
  }

  if (!stats) {
    return <div className="dashboard-empty">No statistics available</div>;
  }

  const { mode, update_cycle, pending_rollouts, pending_feedback, total_jobs, gemini_stats, knowledge_metrics, performance_history } = stats;

  return (
    <div className="learning-dashboard">
      <div className="dashboard-header">
        <h2>Learning Dashboard</h2>
        <span className={`mode-badge ${mode}`}>{mode.toUpperCase()} MODE</span>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <h4>Training Progress</h4>
          <div className="stat-value">{currentCycle || update_cycle}</div>
          <div className="stat-label">Update Cycles</div>
          <div className="rollout-progress">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${((pendingCount || pending_rollouts) / (maxRollouts || 5)) * 100}%` }}
              />
            </div>
            <div className="stat-detail">
              Pending Rollouts: {pendingCount || pending_rollouts} / {maxRollouts || 5}
            </div>
          </div>
        </div>

        <div className="stat-card">
          <h4>Jobs & Feedback</h4>
          <div className="stat-value">{total_jobs}</div>
          <div className="stat-label">Total Jobs</div>
          <div className="stat-detail">
            Awaiting Feedback: {pending_feedback}
          </div>
        </div>

        <div className="stat-card">
          <h4>Gemini API Usage</h4>
          <div className="stat-value">{gemini_stats.gemini_calls}</div>
          <div className="stat-label">Total API Calls</div>
          <div className="stat-detail">
            Cache Hit Rate: {(gemini_stats.cache_hit_rate * 100).toFixed(1)}%
          </div>
          <div className="stat-detail">
            Local LLM Calls: {gemini_stats.local_llm_calls}
          </div>
        </div>

        <div className="stat-card">
          <h4>Cost Optimization</h4>
          <div className="stat-value">
            ${gemini_stats.estimated_cost_usd.toFixed(2)}
          </div>
          <div className="stat-label">Estimated Cost</div>
          <div className="stat-detail savings">
            Saved: ${gemini_stats.estimated_savings_usd.toFixed(2)}
          </div>
          <div className="stat-detail">
            Batched Requests: {gemini_stats.batched_requests}
          </div>
        </div>

        <div className="stat-card">
          <h4>Knowledge Store</h4>
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
        </div>
      </div>

      {performance_history && performance_history.length > 0 && (
        <div className="performance-chart">
          <h3>Performance Over Time</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performance_history}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="cycle"
                label={{ value: 'Update Cycle', position: 'insideBottom', offset: -5 }}
              />
              <YAxis
                label={{ value: 'Average Reward', angle: -90, position: 'insideLeft' }}
                domain={[0, 1]}
              />
              <Tooltip
                formatter={(value: number) => (value * 100).toFixed(1) + '%'}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="avg_reward"
                stroke="#4CAF50"
                strokeWidth={2}
                name="Average Reward"
              />
            </LineChart>
          </ResponsiveContainer>
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
        <p>Last updated: {new Date().toLocaleTimeString()}</p>
      </div>
    </div>
  );
};
