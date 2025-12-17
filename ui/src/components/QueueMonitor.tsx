// Real-time queue monitoring component

import React, { useEffect, useState, useCallback } from 'react';
import { trainingApi } from '../services/api';
import type { QueueStatus } from '../types';

interface QueueMonitorProps {
  onRefresh?: () => void;
}

export const QueueMonitor: React.FC<QueueMonitorProps> = ({ onRefresh }) => {
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchQueueStatus = useCallback(async () => {
    try {
      setError(null);
      const status = await trainingApi.getQueueStatus();
      setQueueStatus(status);
      onRefresh?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch queue status');
    } finally {
      setLoading(false);
    }
  }, [onRefresh]);

  useEffect(() => {
    fetchQueueStatus();
    
    let interval: NodeJS.Timeout | null = null;
    if (autoRefresh) {
      interval = setInterval(fetchQueueStatus, 3000); // Refresh every 3 seconds
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, fetchQueueStatus]);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const formatJobId = (jobId: string) => {
    return jobId.substring(0, 8);
  };

  if (loading) {
    return <div className="queue-monitor loading">Loading queue status...</div>;
  }

  if (error) {
    return (
      <div className="queue-monitor error">
        <p>Error: {error}</p>
        <button onClick={fetchQueueStatus}>Retry</button>
      </div>
    );
  }

  if (!queueStatus) {
    return <div className="queue-monitor">No queue data available</div>;
  }

  return (
    <div className="queue-monitor">
      <div className="queue-header">
        <h2>Training Queue Status</h2>
        <div className="queue-controls">
          <label>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <button onClick={fetchQueueStatus}>Refresh Now</button>
        </div>
      </div>

      <div className="queue-stats">
        <div className="stat-card">
          <div className="stat-label">Active Jobs</div>
          <div className="stat-value">{queueStatus.active_jobs?.length || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Pending Jobs</div>
          <div className="stat-value">{queueStatus.summary?.pending_count || 0}</div>
        </div>
      </div>

      {queueStatus.active_jobs && queueStatus.active_jobs.length > 0 && (
        <div className="active-job-section">
          <h3>🔄 Currently Training</h3>
          {queueStatus.active_jobs.map((job) => (
            <div key={job.job_id} className="job-card active">
              <div className="job-header">
                <span className="job-id">Job: {formatJobId(job.job_id)}</span>
                <span className="job-status status-active">ACTIVE</span>
              </div>
              <div className="job-details">
                <div className="job-field">
                  <strong>Admin:</strong> {job.admin_id}
                </div>
                <div className="job-field">
                  <strong>URL:</strong> <a href={job.url} target="_blank" rel="noopener noreferrer">{job.url}</a>
                </div>
                <div className="job-field">
                  <strong>Description:</strong> {job.description}
                </div>
                <div className="job-field">
                  <strong>Status:</strong> {job.status}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {queueStatus.pending_jobs && queueStatus.pending_jobs.length > 0 && (
        <div className="pending-jobs-section">
          <h3>⏳ Queued Jobs ({queueStatus.pending_jobs.length})</h3>
          <div className="jobs-list">
            {queueStatus.pending_jobs.map((job, index) => (
              <div key={job.job_id} className="job-card pending">
                <div className="job-header">
                  <span className="job-id">Job: {formatJobId(job.job_id)}</span>
                  <span className="job-position">Position #{index + 1}</span>
                </div>
                <div className="job-details">
                  <div className="job-field">
                    <strong>Admin:</strong> {job.admin_id}
                  </div>
                  <div className="job-field">
                    <strong>URL:</strong> <a href={job.url} target="_blank" rel="noopener noreferrer">{job.url}</a>
                  </div>
                  <div className="job-field">
                    <strong>Description:</strong> {job.description}
                  </div>
                  <div className="job-field">
                    <strong>Queued:</strong> {formatTimestamp(job.timestamp)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {(!queueStatus.active_jobs || queueStatus.active_jobs.length === 0) && (!queueStatus.pending_jobs || queueStatus.pending_jobs.length === 0) && (
        <div className="empty-queue">
          <p>✅ Queue is empty - No training jobs in progress</p>
        </div>
      )}
    </div>
  );
};
