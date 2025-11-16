// Required feedback form component with natural language input

import React, { useState } from 'react';

interface FeedbackFormProps {
  jobId: string;
  onSubmit: (feedback: string) => void;
  disabled?: boolean;
}

export const FeedbackForm: React.FC<FeedbackFormProps> = ({
  jobId,
  onSubmit,
  disabled = false,
}) => {
  const [feedback, setFeedback] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (feedback.trim()) {
      onSubmit(feedback.trim());
      setFeedback('');
    }
  };

  const suggestionExamples = [
    'Great! All products were extracted correctly.',
    'Prices are missing the currency symbol.',
    'Some product descriptions are incomplete.',
    'Perfect extraction, but should also include product ratings.',
    'The pagination didn\'t work - only got first page.',
  ];

  return (
    <div className="feedback-form-container">
      <div className="feedback-header">
        <h4>Provide Feedback (Required)</h4>
        <p className="feedback-notice">
          Your feedback helps the agent learn and improve. Please describe what went
          well and what could be improved.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="feedback-form">
        <div className="form-group">
          <label htmlFor="feedback">Feedback in Natural Language *</label>
          <textarea
            id="feedback"
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="Describe what was good and what needs improvement..."
            required
            disabled={disabled}
            rows={5}
            className="form-textarea"
          />
        </div>

        <div className="feedback-suggestions">
          <p className="suggestions-label">Example feedback:</p>
          <div role="list">
            {suggestionExamples.map((example, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setFeedback(example)}
                disabled={disabled}
                className="suggestion-item"
                role="listitem"
              >
                {example}
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={!feedback.trim() || disabled}
          className="btn-primary"
        >
          {disabled ? 'Submitting...' : 'Submit Feedback'}
        </button>
      </form>
    </div>
  );
};
