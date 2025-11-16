// Component for displaying crawl results with metadata

import React from 'react';
import type { CrawlResult } from '../types';

interface CrawlResultsProps {
  result: CrawlResult | null;
  onRetry?: () => void;
}

export const CrawlResults: React.FC<CrawlResultsProps> = ({ result, onRetry }) => {
  if (!result) {
    return (
      <div className="results-placeholder">
        <p>Submit a crawl job to see results here</p>
      </div>
    );
  }

  const { success, data, metadata, base_reward, error } = result;

  return (
    <div className={`crawl-results ${success ? 'success' : 'error'}`}>
      <div className="results-header">
        <h3>{success ? '✓ Crawl Successful' : '✗ Crawl Failed'}</h3>
        <div className="metadata">
          <span>
            <strong>Job ID:</strong> {result.job_id}
          </span>
          <span>
            <strong>Pages:</strong> {metadata.pages_collected}
          </span>
          <span>
            <strong>Time:</strong> {metadata.execution_time_ms}ms
          </span>
          <span>
            <strong>Domain:</strong> {metadata.domain}
          </span>
          <span className="reward-badge">
            <strong>Reward:</strong> {(base_reward * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
          {onRetry && (
            <button onClick={onRetry} className="btn-secondary">
              Retry
            </button>
          )}
        </div>
      )}

      {success && data && data.length > 0 && (
        <div className="extracted-data">
          <h4>Extracted Data ({data.length} items)</h4>
          <div className="data-preview">
            {data.slice(0, 5).map((item, idx) => (
              <div key={idx} className="data-item">
                <pre>{JSON.stringify(item, null, 2)}</pre>
              </div>
            ))}
            {data.length > 5 && (
              <p className="data-more">... and {data.length - 5} more items</p>
            )}
          </div>
        </div>
      )}

      {success && (!data || data.length === 0) && (
        <div className="no-data">
          <p>No data extracted. The crawl succeeded but found no matching content.</p>
        </div>
      )}

      <div className="quality-indicator">
        <div className="quality-bar">
          <div
            className="quality-fill"
            style={{ width: `${base_reward * 100}%` }}
          />
        </div>
        <p className="quality-label">
          Quality Score: {(base_reward * 100).toFixed(1)}%
          {base_reward < 0.5 && ' - Consider providing feedback'}
          {base_reward >= 0.5 && base_reward < 0.8 && ' - Good, but can improve'}
          {base_reward >= 0.8 && ' - Excellent!'}
        </p>
      </div>
    </div>
  );
};

export default CrawlResults;
