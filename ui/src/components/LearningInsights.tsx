import React, { useState, useEffect } from 'react';
import { trainingApi } from '../services/api';
import type { LearningInsights } from '../types';
import './LearningInsights.css';

const LearningInsightsComponent: React.FC = () => {
  const [insights, setInsights] = useState<LearningInsights | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadInsights();
    const interval = setInterval(loadInsights, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const loadInsights = async () => {
    try {
      setError(null);
      const data = await trainingApi.getLearningInsights();
      setInsights(data);
      setLoading(false);
    } catch (err) {
      console.error('Failed to load insights:', err);
      setError('Failed to load learning insights');
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high': return '#10b981';
      case 'medium': return '#f59e0b';
      case 'low': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getConfidenceLabel = (confidence: string) => {
    switch (confidence) {
      case 'high': return '🟢 High';
      case 'medium': return '🟡 Medium';
      case 'low': return '🔴 Low';
      default: return '⚪ Unknown';
    }
  };

  if (loading) {
    return <div className="insights-container">
      <div className="loading">Loading learning insights...</div>
    </div>;
  }

  if (error || !insights) {
    return <div className="insights-container">
      <div className="error">{error || 'No insights available'}</div>
    </div>;
  }

  return (
    <div className="insights-container">
      <h2>🧠 AI Learning Insights</h2>
      <p className="insights-subtitle">What the AI has learned through training</p>

      {/* Summary Cards */}
      <div className="insights-summary">
        <div className="summary-card">
          <div className="summary-icon">📚</div>
          <div className="summary-content">
            <div className="summary-value">{insights.summary.total_patterns}</div>
            <div className="summary-label">Total Patterns</div>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">🌐</div>
          <div className="summary-content">
            <div className="summary-value">{insights.summary.domains_learned}</div>
            <div className="summary-label">Domains Mastered</div>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">🎯</div>
          <div className="summary-content">
            <div className="summary-value">{(insights.summary.avg_success_rate * 100).toFixed(1)}%</div>
            <div className="summary-label">Avg Success Rate</div>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">🔄</div>
          <div className="summary-content">
            <div className="summary-value">{insights.summary.learning_cycles}</div>
            <div className="summary-label">Learning Cycles</div>
          </div>
        </div>
      </div>

      {/* Knowledge Quality */}
      {insights.knowledge_quality && (
        <div className="quality-section">
          <h3>📊 Knowledge Quality Distribution</h3>
          <div className="quality-bars">
            <div className="quality-bar">
              <div className="quality-label">
                <span>🟢 High Confidence</span>
                <span className="quality-count">{insights.knowledge_quality.high_confidence_domains}</span>
              </div>
              <div className="quality-progress">
                <div 
                  className="quality-fill high"
                  style={{
                    width: `${(insights.knowledge_quality.high_confidence_domains / insights.summary.domains_learned * 100) || 0}%`
                  }}
                />
              </div>
            </div>

            <div className="quality-bar">
              <div className="quality-label">
                <span>🟡 Medium Confidence</span>
                <span className="quality-count">{insights.knowledge_quality.medium_confidence_domains}</span>
              </div>
              <div className="quality-progress">
                <div 
                  className="quality-fill medium"
                  style={{
                    width: `${(insights.knowledge_quality.medium_confidence_domains / insights.summary.domains_learned * 100) || 0}%`
                  }}
                />
              </div>
            </div>

            <div className="quality-bar">
              <div className="quality-label">
                <span>🔴 Low Confidence</span>
                <span className="quality-count">{insights.knowledge_quality.low_confidence_domains}</span>
              </div>
              <div className="quality-progress">
                <div 
                  className="quality-fill low"
                  style={{
                    width: `${(insights.knowledge_quality.low_confidence_domains / insights.summary.domains_learned * 100) || 0}%`
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Domain Expertise */}
      <div className="expertise-section">
        <h3>🎓 Domain Expertise (Top 10)</h3>
        {insights.domain_expertise.length === 0 ? (
          <div className="no-data">No domains learned yet. Submit training jobs to begin learning!</div>
        ) : (
          <div className="expertise-table">
            <div className="table-header">
              <div className="col-domain">Domain</div>
              <div className="col-patterns">Patterns</div>
              <div className="col-success">Success Rate</div>
              <div className="col-usage">Usage</div>
              <div className="col-confidence">Confidence</div>
            </div>
            {insights.domain_expertise.map((domain, idx) => (
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
                  <span className="confidence-badge" style={{ color: getConfidenceColor(domain.confidence) }}>
                    {getConfidenceLabel(domain.confidence)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pattern Types Distribution */}
      {Object.keys(insights.pattern_types).length > 0 && (
        <div className="pattern-types-section">
          <h3>🔍 Pattern Types Learned</h3>
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

      {/* Learning Performance Trend */}
      {insights.recent_performance.length > 0 && (
        <div className="performance-section">
          <h3>📈 Recent Learning Performance</h3>
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
            <span>Showing last {insights.recent_performance.length} learning cycles</span>
            <span className="performance-note">Higher bars = Better learning performance</span>
          </div>
        </div>
      )}

      <div className="insights-footer">
        <button onClick={loadInsights} className="refresh-btn">
          🔄 Refresh Insights
        </button>
      </div>
    </div>
  );
};

export default LearningInsightsComponent;
