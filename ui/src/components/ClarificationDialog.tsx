// Dialog for handling clarification questions when feedback confidence is low

import React, { useState } from 'react';

interface ClarificationDialogProps {
  question: string;
  confidence: number;
  onResponse: (response: string) => void;
  onCancel: () => void;
}

export const ClarificationDialog: React.FC<ClarificationDialogProps> = ({
  question,
  confidence,
  onResponse,
  onCancel,
}) => {
  const [response, setResponse] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (response.trim()) {
      onResponse(response.trim());
    }
  };

  return (
    <div className="dialog-overlay">
      <div className="dialog-content clarification-dialog">
        <div className="dialog-header">
          <h3>Clarification Needed</h3>
          <span className="confidence-badge">
            Confidence: {(confidence * 100).toFixed(0)}%
          </span>
        </div>

        <div className="dialog-body">
          <p className="clarification-notice">
            The agent needs clarification to better understand your feedback:
          </p>

          <div className="question-box">
            <p className="question-text">{question}</p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="clarification-response">Your Response</label>
              <textarea
                id="clarification-response"
                value={response}
                onChange={(e) => setResponse(e.target.value)}
                placeholder="Please clarify your feedback..."
                required
                rows={4}
                className="form-textarea"
                autoFocus
              />
            </div>

            <div className="dialog-actions">
              <button
                type="button"
                onClick={onCancel}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!response.trim()}
                className="btn-primary"
              >
                Submit Clarification
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ClarificationDialog;
