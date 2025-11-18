// TypeScript type definitions for the Training UI

export interface CrawlJob {
  url: string;
  user_description: string;
  extraction_schema?: {
    required?: string[];
    [key: string]: any;
  };
  feedback_from_previous?: string;
}

export interface CrawlResult {
  job_id: string;
  success: boolean;
  data: any[];
  metadata: {
    execution_time_ms: number;
    pages_collected: number;
    domain: string;
  };
  base_reward: number;
  error?: string;
}

export interface FeedbackInterpretation {
  confidence: number;
  quality_rating: number;
  specific_issues: string[];
  desired_improvements: string[];
  clarification_needed: boolean;
  clarification_question?: string;
}

export interface FeedbackResponse {
  status: 'accepted' | 'clarification_needed';
  interpretation?: FeedbackInterpretation;
  question?: string;
  confidence?: number;
  quality_rating?: number;
}

export interface TrainingStats {
  mode: string;
  update_cycle: number;
  pending_rollouts: number;
  pending_feedback: number;
  total_jobs: number;
  gemini_stats: {
    gemini_calls: number;
    cache_hits: number;
    local_llm_calls: number;
    batched_requests: number;
    total_requests: number;
    cache_hit_rate: number;
    estimated_cost_usd: number;
    estimated_savings_usd: number;
  };
  knowledge_metrics: {
    total_patterns: number;
    vector_size_mb: number;
    graph_nodes: number;
    cache_hit_rate: number;
  };
  performance_history: Array<{
    cycle: number;
    avg_reward: number;
    timestamp: string;
  }>;
}

export interface WebSocketMessage {
  type: 'job_completed' | 'feedback_received' | 'update_cycle' | 'error' | 'crawl_log' | 'crawl_started' | 'pending_rollouts_updated' | 'learning_cycle_complete';
  job_id?: string;
  success?: boolean;
  items_count?: number;
  quality_rating?: number;
  cycle?: number;
  message?: string;
}

// Log entry from crawl4ai
export interface CrawlLogEntry {
  level: string;          // INFO, WARNING, ERROR, DEBUG
  message: string;        // [INIT]...., [PLAN]...., etc.
  logger: string;         // crawl4ai, crawl4ai_wrapper
  timestamp: string;      // ISO timestamp
  job_id: string | null;  // Current job ID context
}

// Pending rollouts update
export interface PendingRolloutsUpdate {
  pending_count: number;
  update_frequency: number;
  cycle: number;
}

// Learning cycle completion
export interface LearningCycleComplete {
  cycle: number;
  resources_updated: boolean;
  performance_metrics: {
    successful_patterns: number;
    failure_patterns: number;
    feedback_insights: number;
    avg_reward: number;
  };
}
