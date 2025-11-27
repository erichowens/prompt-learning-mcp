#!/usr/bin/env node
/**
 * Prompt Learning MCP Server
 *
 * An MCP server that provides stateful prompt optimization with embedding-based learning.
 * Supports both cold-start (pattern-based) and warm-start (RAG-based) optimization.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import OpenAI from 'openai';
import { Redis } from 'ioredis';
import { v4 as uuidv4 } from 'uuid';

import { EmbeddingService } from './embeddings.js';
import { VectorDB } from './vectordb.js';
import { PromptOptimizer } from './optimizer.js';
import type {
  PromptMetrics,
  PromptRecord,
  RetrievePromptsArgs,
  RecordFeedbackArgs,
  OptimizePromptArgs,
  SuggestImprovementsArgs,
  GetAnalyticsArgs
} from './types.js';

// Configuration
const QDRANT_URL = process.env.VECTOR_DB_URL || 'http://localhost:6333';
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
  console.error('Error: OPENAI_API_KEY environment variable is required');
  process.exit(1);
}

// Initialize clients
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
let redis: Redis | null = null;

try {
  redis = new Redis(REDIS_URL, {
    maxRetriesPerRequest: 3,
    retryStrategy: (times: number) => {
      if (times > 3) return null;
      return Math.min(times * 100, 1000);
    }
  });
  redis.on('error', (err: Error) => {
    console.error('[Redis] Connection error (will continue without caching):', err.message);
    redis = null;
  });
} catch (e) {
  console.error('[Redis] Failed to connect (will continue without caching)');
  redis = null;
}

const embeddings = new EmbeddingService(openai, redis);
const vectorDb = new VectorDB(QDRANT_URL);
const optimizer = new PromptOptimizer(openai);

// EMA alpha for metric updates
const EMA_ALPHA = 0.3;

// Create MCP server
const server = new Server(
  {
    name: 'prompt-learning',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Tool definitions
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'retrieve_prompts',
        description: 'Find similar high-performing prompts from the learning database',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'The prompt to find similar examples for'
            },
            domain: {
              type: 'string',
              description: 'Domain filter (e.g., "code_review", "summarization")'
            },
            top_k: {
              type: 'integer',
              default: 5,
              description: 'Number of results to return'
            },
            min_performance: {
              type: 'number',
              default: 0.7,
              description: 'Minimum success_rate threshold (0-1)'
            }
          },
          required: ['query']
        }
      },
      {
        name: 'record_feedback',
        description: 'Record the outcome of a prompt execution for learning',
        inputSchema: {
          type: 'object',
          properties: {
            prompt_id: {
              type: 'string',
              description: 'ID of existing prompt, or "new" to create one'
            },
            prompt_text: {
              type: 'string',
              description: 'The prompt text (required if prompt_id is "new")'
            },
            domain: {
              type: 'string',
              description: 'Domain classification'
            },
            outcome: {
              type: 'object',
              properties: {
                success: { type: 'boolean' },
                latency_ms: { type: 'number' },
                output_tokens: { type: 'integer' },
                quality_score: { type: 'number' }
              },
              required: ['success']
            },
            user_feedback: {
              type: 'object',
              properties: {
                satisfaction: { type: 'number' },
                comments: { type: 'string' }
              }
            }
          },
          required: ['prompt_id', 'outcome']
        }
      },
      {
        name: 'optimize_prompt',
        description: 'Optimize a prompt using pattern-based and RAG-based techniques',
        inputSchema: {
          type: 'object',
          properties: {
            prompt: {
              type: 'string',
              description: 'The prompt to optimize'
            },
            domain: {
              type: 'string',
              description: 'Domain for optimization context'
            },
            max_iterations: {
              type: 'integer',
              default: 5,
              description: 'Maximum optimization iterations'
            },
            target_score: {
              type: 'number',
              default: 0.9,
              description: 'Target quality score to reach'
            }
          },
          required: ['prompt']
        }
      },
      {
        name: 'suggest_improvements',
        description: 'Get improvement suggestions for a prompt without full optimization',
        inputSchema: {
          type: 'object',
          properties: {
            prompt: {
              type: 'string',
              description: 'The prompt to analyze'
            },
            current_performance: {
              type: 'object',
              properties: {
                success_rate: { type: 'number' },
                avg_latency_ms: { type: 'number' },
                token_efficiency: { type: 'number' }
              }
            },
            improvement_focus: {
              type: 'string',
              enum: ['clarity', 'specificity', 'efficiency', 'all'],
              default: 'all'
            }
          },
          required: ['prompt']
        }
      },
      {
        name: 'get_analytics',
        description: 'Get performance analytics and trends',
        inputSchema: {
          type: 'object',
          properties: {
            domain: {
              type: 'string',
              description: 'Filter by domain'
            },
            time_range: {
              type: 'string',
              enum: ['7d', '30d', '90d', 'all'],
              default: '30d'
            },
            metrics: {
              type: 'array',
              items: { type: 'string' },
              default: ['success_rate', 'token_efficiency']
            }
          }
        }
      }
    ]
  };
});

// Tool implementations
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'retrieve_prompts':
        return await handleRetrievePrompts(args as unknown as RetrievePromptsArgs);

      case 'record_feedback':
        return await handleRecordFeedback(args as unknown as RecordFeedbackArgs);

      case 'optimize_prompt':
        return await handleOptimizePrompt(args as unknown as OptimizePromptArgs);

      case 'suggest_improvements':
        return await handleSuggestImprovements(args as unknown as SuggestImprovementsArgs);

      case 'get_analytics':
        return await handleGetAnalytics(args as unknown as GetAnalyticsArgs);

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return {
      content: [{ type: 'text', text: JSON.stringify({ error: errorMessage }) }],
      isError: true
    };
  }
});

// Handler implementations
async function handleRetrievePrompts(args: RetrievePromptsArgs) {
  const { query, domain, top_k = 5, min_performance = 0.7 } = args;

  // Generate embedding for query
  const { embedding } = await embeddings.embedContextual(
    query,
    domain || 'general',
    'retrieval'
  );

  // Search vector DB
  const results = await vectorDb.search(embedding, {
    topK: top_k,
    minPerformance: min_performance,
    domain
  });

  // Format results
  const formattedResults = results.map(r => ({
    prompt_id: r.id,
    prompt_text: r.payload.prompt_text as string,
    similarity_score: r.score,
    metrics: r.payload.metrics as PromptMetrics,
    domain: r.payload.domain as string
  }));

  return {
    content: [{
      type: 'text',
      text: JSON.stringify({ results: formattedResults }, null, 2)
    }]
  };
}

async function handleRecordFeedback(args: RecordFeedbackArgs) {
  const { prompt_id, prompt_text, domain = 'general', outcome, user_feedback } = args;

  let id = prompt_id;
  let metrics: PromptMetrics;

  if (prompt_id === 'new') {
    if (!prompt_text) {
      throw new Error('prompt_text is required when prompt_id is "new"');
    }

    // Create new prompt
    id = uuidv4();

    // Generate embedding
    const { embedding } = await embeddings.embedContextual(prompt_text, domain, 'storage');

    // Initial metrics
    metrics = {
      success_rate: outcome.success ? 1.0 : 0.0,
      avg_latency_ms: outcome.latency_ms || 0,
      token_efficiency: outcome.quality_score || 0,
      observation_count: 1,
      last_updated: new Date().toISOString()
    };

    // Store in vector DB
    await vectorDb.upsert(id, embedding, {
      prompt_text,
      contextualized_text: `Domain: ${domain}\n\n${prompt_text}`,
      domain,
      task_type: 'general',
      metrics,
      created_at: new Date().toISOString(),
      tags: []
    });
  } else {
    // Update existing prompt
    const existing = await vectorDb.get(prompt_id);
    if (!existing) {
      throw new Error(`Prompt not found: ${prompt_id}`);
    }

    const oldMetrics = existing.metrics;

    // Exponential moving average update
    metrics = {
      success_rate: EMA_ALPHA * (outcome.success ? 1 : 0) + (1 - EMA_ALPHA) * oldMetrics.success_rate,
      avg_latency_ms: EMA_ALPHA * (outcome.latency_ms || oldMetrics.avg_latency_ms) + (1 - EMA_ALPHA) * oldMetrics.avg_latency_ms,
      token_efficiency: EMA_ALPHA * (outcome.quality_score || oldMetrics.token_efficiency) + (1 - EMA_ALPHA) * oldMetrics.token_efficiency,
      observation_count: oldMetrics.observation_count + 1,
      last_updated: new Date().toISOString()
    };

    // Update metrics
    await vectorDb.updateMetrics(prompt_id, metrics);
  }

  return {
    content: [{
      type: 'text',
      text: JSON.stringify({
        status: 'recorded',
        prompt_id: id,
        updated_metrics: metrics
      }, null, 2)
    }]
  };
}

async function handleOptimizePrompt(args: OptimizePromptArgs) {
  const { prompt, domain = 'general', max_iterations = 5, target_score = 0.9 } = args;

  // Get similar high-performing prompts for RAG
  const { embedding } = await embeddings.embedContextual(prompt, domain, 'optimization');
  const similarResults = await vectorDb.search(embedding, {
    topK: 5,
    minPerformance: 0.7,
    domain
  });

  // Convert to PromptRecord format
  const similarPrompts: PromptRecord[] = similarResults.map(r => ({
    id: r.id,
    prompt_text: r.payload.prompt_text as string,
    contextualized_text: r.payload.contextualized_text as string || '',
    domain: r.payload.domain as string || 'general',
    task_type: r.payload.task_type as string || 'general',
    metrics: r.payload.metrics as PromptMetrics,
    created_at: r.payload.created_at as string || '',
    tags: r.payload.tags as string[] || []
  }));

  // Run optimization
  const result = await optimizer.optimize(prompt, similarPrompts, domain);

  return {
    content: [{
      type: 'text',
      text: JSON.stringify(result, null, 2)
    }]
  };
}

async function handleSuggestImprovements(args: SuggestImprovementsArgs) {
  const { prompt, current_performance, improvement_focus = 'all' } = args;

  // Get pattern-based suggestions
  const suggestions = optimizer.getSuggestions(prompt);

  // Get similar high-performing prompts for context
  const { embedding } = await embeddings.embed(prompt);
  const similarResults = await vectorDb.search(embedding, {
    topK: 5,
    minPerformance: 0.8
  });

  const avgPerformance = similarResults.length > 0
    ? similarResults.reduce((sum, r) => sum + (r.payload.metrics as PromptMetrics).success_rate, 0) / similarResults.length
    : 0;

  return {
    content: [{
      type: 'text',
      text: JSON.stringify({
        suggestions: suggestions.map(s => ({
          type: s.type,
          description: s.description,
          example: s.example,
          expected_improvement: s.expectedImprovement
        })),
        based_on: {
          similar_prompts_analyzed: similarResults.length,
          avg_performance_of_similar: avgPerformance
        }
      }, null, 2)
    }]
  };
}

async function handleGetAnalytics(args: GetAnalyticsArgs) {
  const { domain, time_range = '30d' } = args;

  // Get stats from vector DB
  const stats = await vectorDb.getStats();

  // Scroll through prompts for detailed analytics
  const { records } = await vectorDb.scroll(100);

  // Filter by domain if specified
  const filtered = domain
    ? records.filter(r => r.domain === domain)
    : records;

  // Calculate aggregates
  const avgSuccessRate = filtered.length > 0
    ? filtered.reduce((sum, r) => sum + r.metrics.success_rate, 0) / filtered.length
    : 0;

  // Group by domain
  const byDomain: Record<string, { count: number; avg_success: number }> = {};
  for (const record of filtered) {
    const d = record.domain || 'unknown';
    if (!byDomain[d]) {
      byDomain[d] = { count: 0, avg_success: 0 };
    }
    byDomain[d].count++;
    byDomain[d].avg_success += record.metrics.success_rate;
  }
  for (const d of Object.keys(byDomain)) {
    byDomain[d].avg_success /= byDomain[d].count;
  }

  return {
    content: [{
      type: 'text',
      text: JSON.stringify({
        summary: {
          total_prompts: stats.total,
          avg_success_rate: avgSuccessRate,
          improvement_trend: 0 // Would need time-series data
        },
        by_domain: byDomain,
        top_patterns: [] // Would need pattern extraction
      }, null, 2)
    }]
  };
}

// Start server
async function main() {
  try {
    // Initialize vector DB
    await vectorDb.initialize();
    console.error('[prompt-learning] Vector DB initialized');

    // Start MCP server
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error('[prompt-learning] MCP server started');
  } catch (error) {
    console.error('[prompt-learning] Failed to start:', error);
    process.exit(1);
  }
}

main();
