// Form component for submitting crawl training jobs

import React, { useState } from 'react';
import type { CrawlJob } from '../types';

interface CrawlJobFormProps {
  onSubmit: (job: CrawlJob) => void;
  disabled?: boolean;
  previousFeedback?: string;
}

export const CrawlJobForm: React.FC<CrawlJobFormProps> = ({
  onSubmit,
  disabled = false,
  previousFeedback,
}) => {
  const [url, setUrl] = useState('');
  const [description, setDescription] = useState('');
  const [requiredFields, setRequiredFields] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const job: CrawlJob = {
      url,
      user_description: description,
    };

    if (requiredFields.trim()) {
      job.extraction_schema = {
        required: requiredFields.split(',').map((f) => f.trim()).filter(Boolean),
      };
    }

    if (previousFeedback) {
      job.feedback_from_previous = previousFeedback;
    }

    onSubmit(job);
  };

  const isValid = url.trim() && description.trim();

  return (
    <form onSubmit={handleSubmit} className="crawl-job-form">
      <div className="form-group">
        <label htmlFor="url">Target URL *</label>
        <input
          id="url"
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com"
          required
          disabled={disabled}
          className="form-input"
        />
      </div>

      <div className="form-group">
        <label htmlFor="description">What to Extract (Natural Language) *</label>
        <textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Extract product names, prices, and descriptions from the product listing page..."
          required
          disabled={disabled}
          rows={4}
          className="form-textarea"
        />
        <small className="form-hint">
          Describe in natural language what data you want to extract
        </small>
      </div>

      <div className="form-group">
        <label htmlFor="required-fields">Required Fields (Optional)</label>
        <input
          id="required-fields"
          type="text"
          value={requiredFields}
          onChange={(e) => setRequiredFields(e.target.value)}
          placeholder="name, price, description"
          disabled={disabled}
          className="form-input"
        />
        <small className="form-hint">Comma-separated field names</small>
      </div>

      {previousFeedback && (
        <div className="feedback-notice">
          <strong>Re-crawling with previous feedback:</strong>
          <p>{previousFeedback}</p>
        </div>
      )}

      <button type="submit" disabled={!isValid || disabled} className="btn-primary">
        {disabled ? 'Crawling...' : 'Start Training Crawl'}
      </button>
    </form>
  );
};

export default CrawlJobForm;
