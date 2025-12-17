// Buffer review and commit/discard component

import React, { useEffect, useState } from 'react';
import { trainingApi } from '../services/api';
import type { BufferMetadata, BufferData } from '../types';
import './BufferReview.css';

interface BufferReviewProps {
  onCommit?: (jobId: string, version: number) => void;
  onDiscard?: (jobId: string) => void;
}

export const BufferReview: React.FC<BufferReviewProps> = ({ onCommit, onDiscard }) => {
  const [buffers, setBuffers] = useState<BufferMetadata[]>([]);
  const [selectedBuffer, setSelectedBuffer] = useState<BufferData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [adminId, setAdminId] = useState('admin'); // Default admin ID (matches backend)
  const [processing, setProcessing] = useState<string | null>(null);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [feedbackJobId, setFeedbackJobId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState('');
  const [pendingCommits, setPendingCommits] = useState({ pending_count: 0, commits_needed: 5 });

  const fetchBuffers = async () => {
    try {
      setError(null);
      const bufferList = await trainingApi.getPendingBuffers();
      setBuffers(bufferList);
      
      // Also fetch pending commits status
      const commitsStatus = await trainingApi.getPendingCommitsStatus();
      setPendingCommits(commitsStatus);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch buffers');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBuffers();
    
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchBuffers, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleViewBuffer = async (jobId: string) => {
    try {
      setError(null);
      // Find the buffer to get its admin_id
      const buffer = buffers.find(b => b.job_id === jobId);
      const bufferAdminId = buffer?.admin_id || adminId;
      const bufferData = await trainingApi.getBuffer(jobId, bufferAdminId);
      setSelectedBuffer(bufferData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch buffer details');
    }
  };

  const handleCommit = async (jobId: string, withFeedback?: string) => {
    if (!withFeedback && !window.confirm(`Commit training job ${jobId.substring(0, 8)} to create a new version?`)) {
      return;
    }

    setProcessing(jobId);
    try {
      setError(null);
      // Find the buffer to get its admin_id
      const buffer = buffers.find(b => b.job_id === jobId);
      const bufferAdminId = buffer?.admin_id || adminId;
      const result = await trainingApi.commitTraining(jobId, bufferAdminId, withFeedback);
      
      // Handle different response types
      if (result.status === 'version_created') {
        alert(`🎉 ${result.message}\n\n✅ New Version: v${result.version}\n📦 From ${result.commit_count} commits${withFeedback ? '\n📝 Feedback included' : ''}`);
        if (result.version) {
          onCommit?.(jobId, result.version);
        }
      } else {
        alert(`${result.message}\n\n⏳ ${result.commits_needed} more commits needed to create version${withFeedback ? '\n📝 Feedback saved' : ''}`);
      }
      
      // Refresh buffers and close detail view
      await fetchBuffers();
      if (selectedBuffer?.job_id === jobId) {
        setSelectedBuffer(null);
      }
      
      // Close feedback modal if open
      if (showFeedbackModal) {
        setShowFeedbackModal(false);
        setFeedback('');
        setFeedbackJobId(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to commit training');
    } finally {
      setProcessing(null);
    }
  };

  const handleCommitWithFeedback = (jobId: string) => {
    setFeedbackJobId(jobId);
    setShowFeedbackModal(true);
  };

  const handleFeedbackSubmit = () => {
    if (!feedbackJobId) return;
    if (!feedback.trim()) {
      alert('Please enter feedback before submitting');
      return;
    }
    handleCommit(feedbackJobId, feedback);
  };

  const handleDiscard = async (jobId: string) => {
    if (!window.confirm(`Discard training job ${jobId.substring(0, 8)}? This cannot be undone.`)) {
      return;
    }

    setProcessing(jobId);
    try {
      setError(null);
      const result = await trainingApi.discardBuffer(jobId, adminId);
      alert(`🗑️ ${result.message}`);
      onDiscard?.(jobId);
      
      // Refresh buffers and close detail view
      await fetchBuffers();
      if (selectedBuffer?.job_id === jobId) {
        setSelectedBuffer(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to discard buffer');
    } finally {
      setProcessing(null);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const formatJobId = (jobId: string) => {
    return jobId.substring(0, 8);
  };

  if (loading) {
    return <div className="buffer-review loading">Loading buffers...</div>;
  }

  if (error) {
    return (
      <div className="buffer-review error">
        <p>Error: {error}</p>
        <button onClick={fetchBuffers}>Retry</button>
      </div>
    );
  }

  return (
    <div className="buffer-review">
      <div className="buffer-header">
        <h2>Training Buffers ({buffers.length})</h2>
        <div className="pending-commits-indicator" style={{
          padding: '8px 16px',
          borderRadius: '6px',
          background: pendingCommits.pending_count >= 5 ? '#d4edda' : '#fff3cd',
          border: `2px solid ${pendingCommits.pending_count >= 5 ? '#28a745' : '#ffc107'}`,
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <span style={{ fontSize: '20px' }}>
            {pendingCommits.pending_count >= 10 ? '✅' : '⏳'}
          </span>
          <div>
            <strong style={{ display: 'block', fontSize: '14px' }}>
              Pending Commits: {pendingCommits.pending_count}/5
            </strong>
            <small style={{ color: '#666', fontSize: '12px' }}>
              {pendingCommits.commits_needed > 0 
                ? `${pendingCommits.commits_needed} more needed for new version`
                : 'Ready to create version!'}
            </small>
          </div>
        </div>
        <div className="admin-selector">
          <label htmlFor="admin-id">Admin ID:</label>
          <input
            id="admin-id"
            type="text"
            value={adminId}
            onChange={(e) => setAdminId(e.target.value)}
            placeholder="admin-1"
          />
        </div>
        <button onClick={fetchBuffers} className="refresh-btn">Refresh</button>
      </div>

      {buffers.length === 0 ? (
        <div className="empty-buffers">
          <p>📋 No pending buffers to review</p>
          <small>Training results will appear here after jobs complete</small>
        </div>
      ) : (
        <div className="buffers-container">
          <div className="buffers-list">
            {buffers.map((buffer) => (
              <div key={buffer.job_id} className="buffer-card">
                <div className="buffer-card-header">
                  <span className="buffer-job-id">Job: {formatJobId(buffer.job_id)}</span>
                  <span className="buffer-patterns-count">{buffer.patterns_count} patterns</span>
                </div>
                <div className="buffer-card-body">
                  <div className="buffer-field">
                    <strong>Admin:</strong> {buffer.admin_id}
                  </div>
                  <div className="buffer-field">
                    <strong>URL:</strong> <a href={buffer.url} target="_blank" rel="noopener noreferrer">{buffer.url}</a>
                  </div>
                  <div className="buffer-field">
                    <strong>Description:</strong> {buffer.description}
                  </div>
                  <div className="buffer-field">
                    <strong>Created:</strong> {formatTimestamp(buffer.timestamp)}
                  </div>
                  <div className="buffer-field">
                    <strong>Expires:</strong> in {buffer.ttl_hours}h
                  </div>
                </div>
                <div className="buffer-card-actions">
                  <button
                    onClick={() => handleViewBuffer(buffer.job_id)}
                    className="btn-view"
                    disabled={processing === buffer.job_id}
                  >
                    👁️ View Details
                  </button>
                  <button
                    onClick={() => handleCommit(buffer.job_id)}
                    className="btn-commit"
                    disabled={processing === buffer.job_id}
                  >
                    ✅ Commit
                  </button>
                  <button
                    onClick={() => handleCommitWithFeedback(buffer.job_id)}
                    className="btn-commit-feedback"
                    disabled={processing === buffer.job_id}
                  >
                    📝 Commit + Feedback
                  </button>
                  <button
                    onClick={() => handleDiscard(buffer.job_id)}
                    className="btn-discard"
                    disabled={processing === buffer.job_id}
                  >
                    🗑️ Discard
                  </button>
                </div>
              </div>
            ))}
          </div>

          {selectedBuffer && (
            <div className="buffer-details-modal" onClick={() => setSelectedBuffer(null)}>
              <div className="buffer-details-content" onClick={(e) => e.stopPropagation()}>
                <div className="buffer-details-header">
                  <h3>Buffer Details: {formatJobId(selectedBuffer.job_id)}</h3>
                  <button onClick={() => setSelectedBuffer(null)} className="close-btn">✕</button>
                </div>

                <div className="buffer-details-body">
                  <section className="detail-section">
                    <h4>Metadata</h4>
                    <div className="detail-field">
                      <strong>Job ID:</strong> {selectedBuffer.job_id}
                    </div>
                    <div className="detail-field">
                      <strong>Admin:</strong> {selectedBuffer.admin_id}
                    </div>
                    <div className="detail-field">
                      <strong>URL:</strong> <a href={selectedBuffer.url} target="_blank" rel="noopener noreferrer">{selectedBuffer.url}</a>
                    </div>
                    <div className="detail-field">
                      <strong>Description:</strong> {selectedBuffer.description}
                    </div>
                    <div className="detail-field">
                      <strong>Timestamp:</strong> {formatTimestamp(selectedBuffer.timestamp)}
                    </div>
                  </section>

                  {selectedBuffer.result && (
                    <section className="detail-section">
                      <h4>Crawl Result</h4>
                      <div className="detail-field">
                        <strong>Success:</strong> {selectedBuffer.result.success ? '✅ Yes' : '❌ No'}
                      </div>
                      {selectedBuffer.result.error && (
                        <div className="detail-field">
                          <strong>Error:</strong> <span style={{color: 'red'}}>{selectedBuffer.result.error}</span>
                        </div>
                      )}
                      <div className="detail-field">
                        <strong>Items Extracted:</strong> {selectedBuffer.result.data?.length || 0}
                      </div>
                      {selectedBuffer.result.data && selectedBuffer.result.data.length > 0 && (
                        <div className="detail-field">
                          <strong>Extracted Data:</strong>
                          <pre style={{maxHeight: '300px', overflow: 'auto', background: '#f5f5f5', padding: '10px', borderRadius: '4px'}}>
                            {JSON.stringify(selectedBuffer.result.data, null, 2)}
                          </pre>
                        </div>
                      )}
                    </section>
                  )}

                  <section className="detail-section">
                    <h4>Metrics</h4>
                    {selectedBuffer.metrics ? (
                      <div className="metrics-grid">
                        <div className="metric-card">
                          <div className="metric-label">Success</div>
                          <div className="metric-value">{selectedBuffer.metrics.success ? '✅' : '❌'}</div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Items Extracted</div>
                          <div className="metric-value">{selectedBuffer.metrics.items_extracted || 0}</div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Execution Time</div>
                          <div className="metric-value">{selectedBuffer.metrics.execution_time_ms || 0}ms</div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Base Reward</div>
                          <div className="metric-value">{(selectedBuffer.metrics.base_reward || 0).toFixed(2)}</div>
                        </div>
                      </div>
                    ) : (
                      <p>No metrics available</p>
                    )}
                  </section>

                  <section className="detail-section">
                    <h4>Patterns ({selectedBuffer.patterns?.length || 0})</h4>
                    {selectedBuffer.patterns && selectedBuffer.patterns.length > 0 ? (
                      <div className="patterns-list">
                        {selectedBuffer.patterns.slice(0, 5).map((pattern, idx) => (
                          <div key={idx} className="pattern-item">
                            <pre>{JSON.stringify(pattern, null, 2)}</pre>
                          </div>
                        ))}
                        {selectedBuffer.patterns.length > 5 && (
                          <div className="patterns-more">
                            ... and {selectedBuffer.patterns.length - 5} more patterns
                          </div>
                        )}
                      </div>
                    ) : (
                      <p>No patterns available</p>
                    )}
                  </section>
                </div>

                <div className="buffer-details-actions">
                  <button
                    onClick={() => handleCommit(selectedBuffer.job_id)}
                    className="btn-commit-large"
                    disabled={processing === selectedBuffer.job_id}
                  >
                    ✅ Commit
                  </button>
                  <button
                    onClick={() => handleCommitWithFeedback(selectedBuffer.job_id)}
                    className="btn-commit-feedback-large"
                    disabled={processing === selectedBuffer.job_id}
                  >
                    📝 Commit + Feedback
                  </button>
                  <button
                    onClick={() => handleDiscard(selectedBuffer.job_id)}
                    className="btn-discard-large"
                    disabled={processing === selectedBuffer.job_id}
                  >
                    🗑️ Discard
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Feedback Modal */}
      {showFeedbackModal && (
        <div className="buffer-details-modal" onClick={() => setShowFeedbackModal(false)}>
          <div className="buffer-details-content" style={{maxWidth: '600px'}} onClick={(e) => e.stopPropagation()}>
            <div className="buffer-details-header">
              <h3>📝 Add Admin Feedback</h3>
              <button onClick={() => setShowFeedbackModal(false)} className="close-btn">✕</button>
            </div>

            <div className="buffer-details-body">
              <p style={{marginBottom: '15px', color: '#666'}}>
                Provide feedback about this training result. This will be stored with the version for future reference.
              </p>
              <textarea
                value={feedback}
                onChange={(e) => setFeedback(e.target.value)}
                placeholder="Enter your feedback about the training quality, patterns learned, or suggestions for improvement..."
                style={{
                  width: '100%',
                  minHeight: '150px',
                  padding: '10px',
                  fontSize: '14px',
                  borderRadius: '4px',
                  border: '1px solid #ddd',
                  resize: 'vertical',
                  fontFamily: 'inherit'
                }}
              />
            </div>

            <div className="buffer-details-actions">
              <button
                onClick={handleFeedbackSubmit}
                className="btn-commit-large"
                disabled={!feedback.trim() || processing === feedbackJobId}
              >
                ✅ Commit with Feedback
              </button>
              <button
                onClick={() => {
                  setShowFeedbackModal(false);
                  setFeedback('');
                  setFeedbackJobId(null);
                }}
                className="btn-discard-large"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
