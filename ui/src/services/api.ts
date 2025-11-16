// API client for Training Agent

import axios from 'axios';
import type { CrawlJob, CrawlResult, FeedbackResponse, TrainingStats } from '../types';

const API_BASE_URL = process.env.REACT_APP_TRAINING_SERVICE_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

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
  async getStats(): Promise<TrainingStats> {
    const response = await api.get('/stats');
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
