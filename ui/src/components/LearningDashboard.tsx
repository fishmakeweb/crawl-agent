// Dashboard component for visualizing learning progress and statistics

import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { TrainingStats } from '../types';
import { trainingApi } from '../services/api';

export const LearningDashboard: React.FC = () => {
  const [stats, setStats] = useState<TrainingStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadStats = async () => {
    try {
      const data = await trainingApi.getStats();
      setStats(data);
      setError(null);
    } catch (err) {
      setError('Failed to load statistics');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

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
          <div className="stat-value">{update_cycle}</div>
          <div className="stat-label">Update Cycles</div>
          <div className="stat-detail">
            Pending Rollouts: {pending_rollouts} / 5
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

      <div className="dashboard-footer">
        <p>Last updated: {new Date().toLocaleTimeString()}</p>
      </div>
    </div>
  );
};

export default LearningDashboard;
