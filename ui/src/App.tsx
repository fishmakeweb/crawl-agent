// Main application component tying everything together

import React, { useState, useEffect, useRef } from 'react';
import { CrawlJobForm } from './components/CrawlJobForm';
import { CrawlResults } from './components/CrawlResults';
import { FeedbackForm } from './components/FeedbackForm';
import { ClarificationDialog } from './components/ClarificationDialog';
import { LearningDashboard } from './components/LearningDashboard';
import { ErrorBoundary } from './components/ErrorBoundary';
import { trainingApi } from './services/api';
import wsService from './services/websocket';
import type { CrawlJob, CrawlResult, FeedbackResponse, WebSocketMessage } from './types';
import './App.css';

// Constants
const WS_CHECK_INTERVAL = 1000;
const MAX_NOTIFICATIONS = 5;
const NOTIFICATION_TIMEOUT = 5000;

interface Notification {
  id: string;
  message: string;
}

export const App: React.FC = () => {
  const [currentResult, setCurrentResult] = useState<CrawlResult | null>(null);
  const [crawling, setCrawling] = useState(false);
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [feedbackRequired, setFeedbackRequired] = useState(false);
  const [clarificationNeeded, setClarificationNeeded] = useState<{
    question: string;
    confidence: number;
    originalFeedback: string;
  } | null>(null);
  const [activeTab, setActiveTab] = useState<'training' | 'dashboard'>('training');
  const [wsConnected, setWsConnected] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const notificationTimeoutsRef = useRef<Map<string, NodeJS.Timeout>>(new Map());

  useEffect(() => {
    // Initialize WebSocket
    wsService.connect();

    const handleMessage = (message: WebSocketMessage) => {
      if (process.env.NODE_ENV === 'development') {
        console.log('WebSocket message:', message);
      }

      if (message.type === 'job_completed') {
        addNotification(`Job ${message.job_id?.substring(0, 8)} completed`);
      } else if (message.type === 'feedback_received') {
        addNotification('Feedback processed successfully');
      } else if (message.type === 'update_cycle') {
        addNotification(`Learning cycle ${message.cycle} completed!`);
      } else if (message.type === 'error') {
        addNotification(`Error: ${message.message}`);
      }
    };

    wsService.on('all', handleMessage);

    // Check connection status
    const checkConnection = setInterval(() => {
      setWsConnected(wsService.isConnected());
    }, WS_CHECK_INTERVAL);

    // Copy ref to local variable for cleanup
    const timeoutsMap = notificationTimeoutsRef.current;

    return () => {
      wsService.off('all', handleMessage);
      clearInterval(checkConnection);
      wsService.disconnect();
      // Clean up all notification timeouts
      timeoutsMap.forEach((timeout) => clearTimeout(timeout));
      timeoutsMap.clear();
    };
  }, []);

  const addNotification = (message: string) => {
    const id = `${Date.now()}-${Math.random()}`;
    const notification: Notification = { id, message };

    setNotifications((prev) => {
      const updated = [...prev, notification];
      return updated.slice(-MAX_NOTIFICATIONS);
    });

    // Schedule removal of this specific notification
    const timeout = setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
      notificationTimeoutsRef.current.delete(id);
    }, NOTIFICATION_TIMEOUT);

    notificationTimeoutsRef.current.set(id, timeout);
  };

  const handleCrawlSubmit = async (job: CrawlJob) => {
    setCrawling(true);
    setCurrentResult(null);
    setFeedbackRequired(false);

    try {
      const result = await trainingApi.submitCrawl(job);
      setCurrentResult(result);
      setFeedbackRequired(true);
      addNotification('Crawl completed! Please provide feedback.');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      if (process.env.NODE_ENV === 'development') {
        console.error('Crawl failed:', error);
      }
      addNotification(`Crawl failed: ${errorMessage}`);
    } finally {
      setCrawling(false);
    }
  };

  const handleFeedbackSubmit = async (feedback: string) => {
    if (!currentResult || submittingFeedback) return;

    setSubmittingFeedback(true);

    try {
      const response: FeedbackResponse = await trainingApi.submitFeedback(
        currentResult.job_id,
        feedback
      );

      if (response.status === 'clarification_needed') {
        // Show clarification dialog
        setClarificationNeeded({
          question: response.question || 'Could you clarify your feedback?',
          confidence: response.confidence || 0,
          originalFeedback: feedback,
        });
      } else {
        // Feedback accepted
        setFeedbackRequired(false);
        addNotification('Feedback accepted! Agent is learning...');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      if (process.env.NODE_ENV === 'development') {
        console.error('Feedback submission failed:', error);
      }
      addNotification(`Feedback failed: ${errorMessage}`);
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const handleClarificationResponse = async (response: string) => {
    if (!currentResult || !clarificationNeeded || submittingFeedback) return;

    setSubmittingFeedback(true);

    try {
      // Combine original feedback with clarification
      const combinedFeedback = `${clarificationNeeded.originalFeedback}\n\nClarification: ${response}`;

      const feedbackResponse: FeedbackResponse = await trainingApi.submitFeedback(
        currentResult.job_id,
        combinedFeedback
      );

      if (feedbackResponse.status === 'clarification_needed') {
        // Still needs more clarification
        setClarificationNeeded({
          question: feedbackResponse.question || 'Could you clarify further?',
          confidence: feedbackResponse.confidence || 0,
          originalFeedback: combinedFeedback,
        });
      } else {
        // Feedback accepted
        setClarificationNeeded(null);
        setFeedbackRequired(false);
        addNotification('Feedback accepted after clarification!');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      if (process.env.NODE_ENV === 'development') {
        console.error('Clarification submission failed:', error);
      }
      addNotification(`Clarification failed: ${errorMessage}`);
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const handleClarificationCancel = () => {
    setClarificationNeeded(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Self-Learning Web Crawler - Training UI</h1>
        <div className="header-status">
          <span className={`ws-status ${wsConnected ? 'connected' : 'disconnected'}`}>
            {wsConnected ? '● Connected' : '○ Disconnected'}
          </span>
        </div>
      </header>

      <div className="notifications">
        {notifications.map((notification) => (
          <div key={notification.id} className="notification" role="alert">
            {notification.message}
          </div>
        ))}
      </div>

      <div className="tabs">
        <button
          className={`tab ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          Training Interface
        </button>
        <button
          className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          Learning Dashboard
        </button>
      </div>

      <main className="app-content">
        <ErrorBoundary>
          {activeTab === 'training' ? (
            <div className="training-interface">
              <div className="training-column">
                <section className="section">
                  <h2>Submit Crawl Job</h2>
                  <CrawlJobForm onSubmit={handleCrawlSubmit} disabled={crawling} />
                </section>
              </div>

              <div className="results-column">
                <section className="section">
                  <h2>Crawl Results</h2>
                  <CrawlResults result={currentResult} />
                </section>

                {feedbackRequired && currentResult && (
                  <section className="section feedback-section">
                    <FeedbackForm
                      jobId={currentResult.job_id}
                      onSubmit={handleFeedbackSubmit}
                      disabled={submittingFeedback}
                    />
                  </section>
                )}
              </div>
            </div>
          ) : (
            <LearningDashboard />
          )}
        </ErrorBoundary>
      </main>

      {clarificationNeeded && (
        <ClarificationDialog
          question={clarificationNeeded.question}
          confidence={clarificationNeeded.confidence}
          onResponse={handleClarificationResponse}
          onCancel={handleClarificationCancel}
        />
      )}

      <footer className="app-footer">
        <p>
          Self-Learning Agent powered by Microsoft Agent-Lightning + Gemini AI
        </p>
      </footer>
    </div>
  );
};

export default App;
