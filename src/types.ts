/**
 * Type definitions for the Prompt Learning MCP Server
 */

export interface PromptMetrics {
  success_rate: number;
  avg_latency_ms: number;
  token_efficiency: number;
  observation_count: number;
  last_updated: string;
}

export interface PromptRecord {
  id: string;
  prompt_text: string;
  contextualized_text: string;
  domain: string;
  task_type: string;
  metrics: PromptMetrics;
  created_at: string;
  tags: string[];
}

export interface RetrievePromptsArgs {
  query: string;
  domain?: string;
  top_k?: number;
  min_performance?: number;
}

export interface RetrievePromptsResult {
  results: Array<{
    prompt_id: string;
    prompt_text: string;
    similarity_score: number;
    metrics: PromptMetrics;
    domain: string;
  }>;
}

export interface RecordFeedbackArgs {
  prompt_id: string;
  prompt_text?: string;
  domain?: string;
  outcome: {
    success: boolean;
    latency_ms?: number;
    output_tokens?: number;
    quality_score?: number;
  };
  user_feedback?: {
    satisfaction?: number;
    comments?: string;
  };
}

export interface RecordFeedbackResult {
  status: string;
  prompt_id: string;
  updated_metrics: PromptMetrics;
}

export interface OptimizePromptArgs {
  prompt: string;
  domain?: string;
  max_iterations?: number;
  target_score?: number;
}

export interface OptimizePromptResult {
  original_prompt: string;
  optimized_prompt: string;
  improvements_made: string[];
  iterations: number;
  estimated_improvement: number;
  similar_prompts_used: number;
}

export interface SuggestImprovementsArgs {
  prompt: string;
  current_performance?: {
    success_rate?: number;
    avg_latency_ms?: number;
    token_efficiency?: number;
  };
  improvement_focus?: 'clarity' | 'specificity' | 'efficiency' | 'all';
}

export interface SuggestImprovementsResult {
  suggestions: Array<{
    type: string;
    description: string;
    example: string;
    expected_improvement: number;
  }>;
  based_on: {
    similar_prompts_analyzed: number;
    avg_performance_of_similar: number;
  };
}

export interface GetAnalyticsArgs {
  domain?: string;
  time_range?: '7d' | '30d' | '90d' | 'all';
  metrics?: string[];
}

export interface GetAnalyticsResult {
  summary: {
    total_prompts: number;
    avg_success_rate: number;
    improvement_trend: number;
  };
  by_domain: Record<string, { count: number; avg_success: number }>;
  top_patterns: Array<{ pattern: string; success_rate: number }>;
}

export interface EmbeddingResult {
  embedding: number[];
  tokens_used: number;
}

export interface QdrantPoint {
  id: string;
  vector: number[];
  payload: Record<string, unknown>;
}

export interface QdrantSearchResult {
  id: string;
  score: number;
  payload: Record<string, unknown>;
}
