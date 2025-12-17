// Version history display component

import React, { useEffect, useState, useCallback } from 'react';
import { trainingApi } from '../services/api';
import type { VersionInfo } from '../types';
import './VersionHistory.css';

interface VersionHistoryProps {
  onRefresh?: () => void;
}

export const VersionHistory: React.FC<VersionHistoryProps> = ({ onRefresh }) => {
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchVersions = useCallback(async () => {
    try {
      setError(null);
      const versionData = await trainingApi.getVersionHistory();
      setVersions(versionData.versions);
      onRefresh?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch version history');
    } finally {
      setLoading(false);
    }
  }, [onRefresh]);

  useEffect(() => {
    fetchVersions();
  }, [fetchVersions]);

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const formatRelativeTime = (timestamp: string) => {
    const now = new Date();
    const then = new Date(timestamp);
    const diffMs = now.getTime() - then.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 30) return `${diffDays}d ago`;
    return formatTimestamp(timestamp);
  };

  if (loading) {
    return <div className="version-history loading">Loading version history...</div>;
  }

  if (error) {
    return (
      <div className="version-history error">
        <p>Error: {error}</p>
        <button onClick={fetchVersions}>Retry</button>
      </div>
    );
  }

  if (versions.length === 0) {
    return (
      <div className="version-history empty">
        <p>📚 No version history available yet</p>
        <small>Committed training sessions will appear here</small>
      </div>
    );
  }

  return (
    <div className="version-history">
      <div className="version-header">
        <h2>Training Version History ({versions.length})</h2>
        <button onClick={fetchVersions} className="refresh-btn">Refresh</button>
      </div>

      <div className="versions-timeline">
        {versions.map((version) => (
          <div key={version.version} className={`version-item ${version.is_latest ? 'latest' : ''}`}>
            <div className="version-marker">
              <div className="version-number">v{version.version}</div>
              {version.is_latest && <span className="latest-badge">LATEST</span>}
            </div>

            <div className="version-content">
              <div className="version-header-info">
                <div className="version-timestamp">
                  <span className="timestamp-full">{formatTimestamp(version.timestamp)}</span>
                  <span className="timestamp-relative">{formatRelativeTime(version.timestamp)}</span>
                </div>
              </div>

              <div className="version-details">
                <div className="version-field">
                  <strong>Admin:</strong> {version.admin_id}
                </div>
                <div className="version-field">
                  <strong>Patterns:</strong> {version.patterns_count} learned patterns
                </div>
                <div className="version-field">
                  <strong>File:</strong> <code>{version.file_path}</code>
                </div>
              </div>

              {version.is_latest && (
                <div className="version-status">
                  <span className="status-icon">🎯</span>
                  <span className="status-text">Currently in use by production agent</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="version-summary">
        <div className="summary-card">
          <div className="summary-label">Total Versions</div>
          <div className="summary-value">{versions.length}</div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Latest Version</div>
          <div className="summary-value">v{versions[0]?.version || 0}</div>
        </div>
        <div className="summary-card">
          <div className="summary-label">Total Patterns</div>
          <div className="summary-value">{versions[0]?.patterns_count || 0}</div>
        </div>
      </div>
    </div>
  );
};
