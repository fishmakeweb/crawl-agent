// API client for Training Agent

import axios, { AxiosError } from 'axios';
import type { CrawlJob, CrawlResult, FeedbackResponse, TrainingStats } from '../types';

const API_BASE_URL = process.env.REACT_APP_TRAINING_SERVICE_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000, // 5 minutes timeout
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (process.env.NODE_ENV === 'development') {
      console.error('API Error:', error);
    }

    // Handle specific error cases
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const data = error.response.data as { message?: string; detail?: string };

      switch (status) {
        case 400:
          throw new Error(data?.message || data?.detail || 'Invalid request');
        case 401:
          throw new Error('Unauthorized - please login again');
        case 403:
          throw new Error('Access forbidden');
        case 404:
          throw new Error('Resource not found');
        case 422:
          throw new Error(data?.message || data?.detail || 'Validation error');
        case 429:
          throw new Error('Too many requests - please try again later');
        case 500:
          throw new Error('Server error - please try again later');
        case 503:
          throw new Error('Service unavailable - please try again later');
        default:
          throw new Error(data?.message || data?.detail || `Request failed with status ${status}`);
      }
    } else if (error.request) {
      // Request made but no response received
      throw new Error('Network error - please check your connection');
    } else {
      // Error setting up the request
      throw new Error(error.message || 'Request failed');
    }
  }
);

export const trainingApi = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // Submit training crawl
  async submitCrawl(job: CrawlJob): Promise<CrawlResult> {
    const response = await api.post('/train-crawl', job);
    return response.data;
  },

  // Submit feedback
  async submitFeedback(jobId: string, feedback: string): Promise<FeedbackResponse> {
    const response = await api.post('/feedback', {
      job_id: jobId,
      feedback: feedback,
    });
    return response.data;
  },

  // Get training stats
  async getStats(signal?: AbortSignal): Promise<TrainingStats> {
    const response = await api.get('/stats', { signal });
    return response.data;
  },

  // Get learned patterns
  async getPatterns() {
    const response = await api.get('/knowledge/patterns');
    return response.data;
  },

  // Trigger consolidation
  async triggerConsolidation() {
    const response = await api.post('/knowledge/consolidate');
    return response.data;
  },

  // Get RL policy
  async getRLPolicy() {
    const response = await api.get('/rl/policy');
    return response.data;
  },
};

export default trainingApi;
