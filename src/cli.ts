#!/usr/bin/env node
/**
 * CLI for Prompt Learning
 *
 * Provides direct command-line access to prompt optimization.
 * Use this for:
 * - One-off prompt optimization
 * - Integration with shell scripts
 * - Testing the optimization pipeline
 */

import OpenAI from 'openai';
import { createInterface } from 'readline';
import { readFileSync, existsSync } from 'fs';
import { QdrantClient } from '@qdrant/js-client-rest';
import { PromptOptimizer } from './optimizer.js';
import { EmbeddingService } from './embeddings.js';
import { VectorDB } from './vectordb.js';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!OPENAI_API_KEY) {
  console.error('Error: OPENAI_API_KEY environment variable is required');
  process.exit(1);
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const optimizer = new PromptOptimizer(openai);

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'optimize':
      await handleOptimize(args.slice(1));
      break;

    case 'score':
      await handleScore(args.slice(1));
      break;

    case 'suggest':
      await handleSuggest(args.slice(1));
      break;

    case 'interactive':
    case '-i':
      await handleInteractive();
      break;

    case 'record':
    case 'record-feedback':
      await handleRecordFeedback(args.slice(1));
      break;

    case 'hook':
      await handleHook();
      break;

    case 'rewrite-hook':
      await handleRewriteHook();
      break;

    case 'help':
    case '--help':
    case '-h':
      printHelp();
      break;

    default:
      // If no command but has args, treat as prompt to optimize
      if (args.length > 0) {
        await handleOptimize(args);
      } else {
        printHelp();
      }
  }
}

async function handleOptimize(args: string[]) {
  let prompt: string;
  let domain = 'general';
  let maxIterations = 5;
  let verbose = false;

  // Parse arguments
  const positional: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--domain' || args[i] === '-d') {
      domain = args[++i];
    } else if (args[i] === '--iterations' || args[i] === '-n') {
      maxIterations = parseInt(args[++i], 10);
    } else if (args[i] === '--verbose' || args[i] === '-v') {
      verbose = true;
    } else if (args[i] === '--stdin') {
      // Read from stdin
      prompt = await readStdin();
    } else {
      positional.push(args[i]);
    }
  }

  // Get prompt from positional args or stdin
  if (!prompt!) {
    if (positional.length > 0) {
      prompt = positional.join(' ');
    } else {
      prompt = await readStdin();
    }
  }

  if (!prompt || prompt.trim().length === 0) {
    console.error('Error: No prompt provided');
    process.exit(1);
  }

  console.error(`Optimizing prompt (domain: ${domain}, max iterations: ${maxIterations})...`);
  console.error('');

  const result = await optimizer.optimize(prompt, [], domain);

  if (verbose) {
    console.error('Improvements made:');
    for (const improvement of result.improvements_made) {
      console.error(`  • ${improvement}`);
    }
    console.error('');
    console.error(`Iterations: ${result.iterations}`);
    console.error(`Estimated improvement: ${(result.estimated_improvement * 100).toFixed(1)}%`);
    console.error('');
    console.error('--- Optimized Prompt ---');
  }

  // Output only the optimized prompt to stdout (for piping)
  console.log(result.optimized_prompt);
}

async function handleScore(args: string[]) {
  let prompt: string;
  let domain = 'general';

  const positional: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--domain' || args[i] === '-d') {
      domain = args[++i];
    } else {
      positional.push(args[i]);
    }
  }

  prompt = positional.length > 0 ? positional.join(' ') : await readStdin();

  if (!prompt || prompt.trim().length === 0) {
    console.error('Error: No prompt provided');
    process.exit(1);
  }

  console.error(`Scoring prompt (domain: ${domain})...`);

  const score = await optimizer.scorePrompt(prompt, domain);
  const evaluation = optimizer.getLastEvaluation();

  console.log(JSON.stringify({
    score: score,
    percentage: `${(score * 100).toFixed(1)}%`,
    breakdown: evaluation?.scores,
    reasoning: evaluation?.reasoning
  }, null, 2));
}

async function handleSuggest(args: string[]) {
  let prompt: string;

  const positional: string[] = [];
  for (let i = 0; i < args.length; i++) {
    positional.push(args[i]);
  }

  prompt = positional.length > 0 ? positional.join(' ') : await readStdin();

  if (!prompt || prompt.trim().length === 0) {
    console.error('Error: No prompt provided');
    process.exit(1);
  }

  const suggestions = optimizer.getSuggestions(prompt);

  console.log(JSON.stringify({
    suggestions: suggestions,
    total_expected_improvement: `${(suggestions.reduce((sum, s) => sum + s.expectedImprovement, 0) * 100).toFixed(1)}%`
  }, null, 2));
}

/**
 * Record feedback for a prompt (used by hooks and direct CLI)
 */
async function handleRecordFeedback(args: string[]) {
  const QDRANT_URL = process.env.VECTOR_DB_URL || 'http://localhost:6333';

  let promptText = '';
  let domain = 'general';
  let success = true;
  let qualityScore = 0.8;
  let promptId = 'new';

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--prompt' || args[i] === '-p') {
      promptText = args[++i];
    } else if (args[i] === '--domain' || args[i] === '-d') {
      domain = args[++i];
    } else if (args[i] === '--success') {
      success = args[++i] !== 'false';
    } else if (args[i] === '--quality' || args[i] === '-q') {
      qualityScore = parseFloat(args[++i]);
    } else if (args[i] === '--id') {
      promptId = args[++i];
    } else if (args[i] === '--stdin') {
      promptText = await readStdin();
    }
  }

  if (!promptText && promptId === 'new') {
    console.error('Error: --prompt or --stdin required for new prompts');
    process.exit(1);
  }

  // Initialize services
  const vectorDb = new VectorDB(QDRANT_URL);
  const embeddings = new EmbeddingService(openai, null);

  try {
    await vectorDb.initialize();

    if (promptId === 'new') {
      // Generate embedding and store
      const { embedding } = await embeddings.embedContextual(promptText, domain, 'storage');
      const id = crypto.randomUUID();

      await vectorDb.upsert(id, embedding, {
        prompt_text: promptText,
        contextualized_text: `Domain: ${domain}\n\n${promptText}`,
        domain,
        task_type: 'general',
        metrics: {
          success_rate: success ? 1.0 : 0.0,
          avg_latency_ms: 0,
          token_efficiency: qualityScore,
          observation_count: 1,
          last_updated: new Date().toISOString()
        },
        created_at: new Date().toISOString(),
        tags: []
      });

      console.log(JSON.stringify({ status: 'recorded', prompt_id: id, success, quality_score: qualityScore }));
    } else {
      // Update existing
      const existing = await vectorDb.get(promptId);
      if (!existing) {
        console.error(`Error: Prompt not found: ${promptId}`);
        process.exit(1);
      }

      const oldMetrics = existing.metrics;
      const EMA_ALPHA = 0.3;

      const newMetrics = {
        success_rate: EMA_ALPHA * (success ? 1 : 0) + (1 - EMA_ALPHA) * oldMetrics.success_rate,
        avg_latency_ms: oldMetrics.avg_latency_ms,
        token_efficiency: EMA_ALPHA * qualityScore + (1 - EMA_ALPHA) * oldMetrics.token_efficiency,
        observation_count: oldMetrics.observation_count + 1,
        last_updated: new Date().toISOString()
      };

      await vectorDb.updateMetrics(promptId, newMetrics);
      console.log(JSON.stringify({ status: 'updated', prompt_id: promptId, metrics: newMetrics }));
    }
  } catch (error) {
    console.error('Error recording feedback:', error);
    process.exit(1);
  }
}

/**
 * Hook handler - reads Claude Code hook input from stdin and records feedback
 */
async function handleHook() {
  const QDRANT_URL = process.env.VECTOR_DB_URL || 'http://localhost:6333';

  // Read hook input from stdin
  const hookInput = await readStdin();
  if (!hookInput) {
    // No input means not called from a hook
    console.error('Error: hook command expects JSON input from stdin');
    process.exit(1);
  }

  let hookData: any;
  try {
    hookData = JSON.parse(hookInput);
  } catch {
    console.error('Error: Invalid JSON input');
    process.exit(1);
  }

  const { hook_event_name, transcript_path } = hookData;

  // Only process Stop events (end of conversation turn)
  if (hook_event_name !== 'Stop') {
    // Output empty response for non-Stop events
    console.log(JSON.stringify({}));
    process.exit(0);
  }

  // Read and parse transcript
  if (!transcript_path || !existsSync(transcript_path)) {
    console.log(JSON.stringify({}));
    process.exit(0);
  }

  try {
    const transcript = readFileSync(transcript_path, 'utf-8');
    const lines = transcript.trim().split('\n').filter(Boolean);

    // Find the last user message and assistant response
    let lastUserPrompt = '';
    let lastAssistantResponse = '';
    let toolsUsed: string[] = [];

    for (const line of lines) {
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'human' || entry.role === 'user') {
          lastUserPrompt = entry.content || entry.message || '';
        } else if (entry.type === 'assistant' || entry.role === 'assistant') {
          lastAssistantResponse = entry.content || entry.message || '';
        } else if (entry.type === 'tool_use' || entry.tool_name) {
          toolsUsed.push(entry.tool_name || entry.name || 'unknown');
        }
      } catch {
        // Skip malformed lines
      }
    }

    // Skip if no meaningful prompt
    if (!lastUserPrompt || lastUserPrompt.length < 10) {
      console.log(JSON.stringify({}));
      process.exit(0);
    }

    // Skip simple queries that don't benefit from learning
    const skipPatterns = [
      /^(hi|hello|hey|thanks|ok|yes|no)$/i,
      /^(what is|who is|when was)/i,  // Simple questions
      /^\/\w+/,  // Slash commands
    ];

    if (skipPatterns.some(p => p.test(lastUserPrompt.trim()))) {
      console.log(JSON.stringify({}));
      process.exit(0);
    }

    // Infer domain from content
    let domain = 'general';
    if (toolsUsed.includes('Bash') || toolsUsed.includes('Edit') || lastUserPrompt.toLowerCase().includes('code')) {
      domain = 'code';
    } else if (lastUserPrompt.toLowerCase().includes('review') || lastUserPrompt.toLowerCase().includes('pr')) {
      domain = 'code_review';
    } else if (lastUserPrompt.toLowerCase().includes('test')) {
      domain = 'testing';
    }

    // Estimate success (heuristic: response length, tool usage indicates engagement)
    const success = lastAssistantResponse.length > 100 || toolsUsed.length > 0;
    const qualityScore = Math.min(0.95, 0.6 + (toolsUsed.length * 0.1) + (lastAssistantResponse.length > 500 ? 0.1 : 0));

    // Initialize and record
    const vectorDb = new VectorDB(QDRANT_URL);
    const embeddings = new EmbeddingService(openai, null);

    await vectorDb.initialize();
    const { embedding } = await embeddings.embedContextual(lastUserPrompt, domain, 'storage');
    const id = crypto.randomUUID();

    await vectorDb.upsert(id, embedding, {
      prompt_text: lastUserPrompt,
      contextualized_text: `Domain: ${domain}\nTools used: ${toolsUsed.join(', ')}\n\n${lastUserPrompt}`,
      domain,
      task_type: toolsUsed.length > 0 ? 'task' : 'query',
      metrics: {
        success_rate: success ? 1.0 : 0.5,
        avg_latency_ms: 0,
        token_efficiency: qualityScore,
        observation_count: 1,
        last_updated: new Date().toISOString()
      },
      created_at: new Date().toISOString(),
      tags: toolsUsed
    });

    // Output for hook (don't block)
    console.log(JSON.stringify({
      recorded: true,
      prompt_id: id,
      domain,
      inferred_success: success
    }));

  } catch (error) {
    // Don't fail the hook on errors - just log and continue
    console.error('[prompt-learning hook]', error);
    console.log(JSON.stringify({}));
  }
}

/**
 * Rewrite hook handler - intercepts user prompts and rewrites them
 * Used with UserPromptSubmit hook event
 */
async function handleRewriteHook() {
  const QDRANT_URL = process.env.VECTOR_DB_URL || 'http://localhost:6333';

  // Read hook input from stdin
  const hookInput = await readStdin();
  if (!hookInput) {
    console.log(JSON.stringify({}));
    process.exit(0);
  }

  let hookData: any;
  try {
    hookData = JSON.parse(hookInput);
  } catch {
    console.log(JSON.stringify({}));
    process.exit(0);
  }

  const { hook_event_name, prompt } = hookData;

  // Only process UserPromptSubmit events
  if (hook_event_name !== 'UserPromptSubmit' || !prompt) {
    console.log(JSON.stringify({}));
    process.exit(0);
  }

  const userPrompt = typeof prompt === 'string' ? prompt : prompt.content || '';

  // Skip short prompts, commands, and simple messages
  if (userPrompt.length < 20) {
    console.log(JSON.stringify({}));
    process.exit(0);
  }

  const skipPatterns = [
    /^(hi|hello|hey|thanks|ok|yes|no|y|n)$/i,
    /^\/\w+/,  // Slash commands
    /^(what|who|when|where|why|how) (is|are|was|were|do|does|did)/i,  // Simple questions
    /^\d+$/,  // Just numbers
    /^(continue|go ahead|proceed|do it|yes please)$/i,
  ];

  if (skipPatterns.some(p => p.test(userPrompt.trim()))) {
    console.log(JSON.stringify({}));
    process.exit(0);
  }

  try {
    // Initialize services
    const vectorDb = new VectorDB(QDRANT_URL);
    const embeddings = new EmbeddingService(openai, null);

    await vectorDb.initialize();

    // Infer domain
    let domain = 'general';
    const lowerPrompt = userPrompt.toLowerCase();
    if (lowerPrompt.includes('code') || lowerPrompt.includes('function') || lowerPrompt.includes('bug') || lowerPrompt.includes('error')) {
      domain = 'code';
    } else if (lowerPrompt.includes('review') || lowerPrompt.includes('pr')) {
      domain = 'code_review';
    } else if (lowerPrompt.includes('test')) {
      domain = 'testing';
    }

    // Get similar high-performing prompts
    const { embedding } = await embeddings.embedContextual(userPrompt, domain, 'retrieval');
    const similarPrompts = await vectorDb.search(embedding, {
      topK: 3,
      minPerformance: 0.7,
      domain
    });

    // If we have good examples, use them to enhance
    let enhancedPrompt = userPrompt;
    let wasRewritten = false;
    let improvementPct = 0;

    if (similarPrompts.length > 0) {
      // Use optimizer with similar prompts as context
      const similarRecords = similarPrompts.map(r => ({
        id: r.id,
        prompt_text: r.payload.prompt_text as string,
        contextualized_text: r.payload.contextualized_text as string || '',
        domain: r.payload.domain as string || 'general',
        task_type: r.payload.task_type as string || 'general',
        metrics: r.payload.metrics as any,
        created_at: r.payload.created_at as string || '',
        tags: r.payload.tags as string[] || []
      }));

      const result = await optimizer.optimize(userPrompt, similarRecords, domain);

      // Only use rewritten prompt if there's meaningful improvement
      if (result.estimated_improvement > 0.1 && result.optimized_prompt !== userPrompt) {
        enhancedPrompt = result.optimized_prompt;
        wasRewritten = true;
        improvementPct = Math.round(result.estimated_improvement * 100);

        // Log what we did (to stderr so it doesn't affect JSON output)
        console.error(`[prompt-learning] Rewrote prompt (${improvementPct}% improvement)`);
        console.error(`[prompt-learning] Original: ${userPrompt.substring(0, 50)}...`);
        console.error(`[prompt-learning] Rewritten: ${enhancedPrompt.substring(0, 50)}...`);
      }
    } else {
      // No similar prompts - use pattern-based quick improvements
      const suggestions = optimizer.getSuggestions(userPrompt);

      if (suggestions.length > 0 && suggestions[0].expectedImprovement > 0.1) {
        // Apply the top suggestion pattern
        const result = await optimizer.optimize(userPrompt, [], domain);

        if (result.estimated_improvement > 0.15 && result.optimized_prompt !== userPrompt) {
          enhancedPrompt = result.optimized_prompt;
          wasRewritten = true;
          improvementPct = Math.round(result.estimated_improvement * 100);

          console.error(`[prompt-learning] Applied patterns (${improvementPct}% improvement)`);
        }
      }
    }

    if (wasRewritten) {
      // Inject visible transformation notice into the prompt itself
      const visiblePrompt = `---
🧠 PROMPT LEARNING: Transformed your input
📝 Original: "${userPrompt}"
✨ Enhanced: "${enhancedPrompt}"
📊 Improvement: ${improvementPct}%
---

${enhancedPrompt}`;

      // Return the rewritten prompt with visible header
      console.log(JSON.stringify({
        hookSpecificOutput: {
          hookEventName: 'UserPromptSubmit',
          modifiedPrompt: visiblePrompt,
          originalPrompt: userPrompt,
          reason: `Optimized based on ${similarPrompts.length} similar high-performing prompts`
        }
      }));
    } else {
      // No changes needed
      console.log(JSON.stringify({}));
    }

  } catch (error) {
    // Don't block on errors
    console.error('[prompt-learning rewrite]', error);
    console.log(JSON.stringify({}));
  }
}

async function handleInteractive() {
  const rl = createInterface({
    input: process.stdin,
    output: process.stderr  // Use stderr for prompts so stdout can be piped
  });

  console.error('🧠 Prompt Learning - Interactive Mode');
  console.error('Commands: optimize, score, suggest, quit');
  console.error('');

  const askQuestion = (query: string): Promise<string> => {
    return new Promise(resolve => rl.question(query, resolve));
  };

  while (true) {
    const command = await askQuestion('> ');

    if (command === 'quit' || command === 'exit' || command === 'q') {
      console.error('Goodbye!');
      rl.close();
      break;
    }

    if (command === 'optimize' || command === 'o') {
      const prompt = await askQuestion('Prompt to optimize: ');
      const domain = await askQuestion('Domain (press enter for "general"): ') || 'general';

      console.error('\nOptimizing...\n');
      const result = await optimizer.optimize(prompt, [], domain);

      console.error('Improvements:');
      for (const improvement of result.improvements_made) {
        console.error(`  • ${improvement}`);
      }
      console.error('');
      console.error('Optimized prompt:');
      console.log(result.optimized_prompt);
      console.error('');
    }

    if (command === 'score' || command === 's') {
      const prompt = await askQuestion('Prompt to score: ');
      const domain = await askQuestion('Domain (press enter for "general"): ') || 'general';

      console.error('\nScoring...\n');
      const score = await optimizer.scorePrompt(prompt, domain);
      const evaluation = optimizer.getLastEvaluation();

      console.error(`Score: ${(score * 100).toFixed(1)}%`);
      if (evaluation?.scores) {
        console.error('Breakdown:');
        for (const [criterion, value] of Object.entries(evaluation.scores)) {
          if (criterion !== 'reasoning' && typeof value === 'number') {
            console.error(`  ${criterion}: ${value}/10`);
          }
        }
      }
      if (evaluation?.reasoning) {
        console.error(`\nAnalysis: ${evaluation.reasoning}`);
      }
      console.error('');
    }

    if (command === 'suggest') {
      const prompt = await askQuestion('Prompt to analyze: ');
      const suggestions = optimizer.getSuggestions(prompt);

      console.error('\nSuggestions:');
      for (const s of suggestions) {
        console.error(`  • ${s.type}: ${s.description} (+${(s.expectedImprovement * 100).toFixed(0)}%)`);
      }
      console.error('');
    }

    if (command === 'help' || command === '?') {
      console.error('Commands:');
      console.error('  optimize (o) - Optimize a prompt');
      console.error('  score (s)    - Score a prompt');
      console.error('  suggest      - Get quick suggestions');
      console.error('  quit (q)     - Exit');
      console.error('');
    }
  }
}

function printHelp() {
  console.log(`
Prompt Learning CLI

Usage:
  prompt-learn optimize <prompt>    Optimize a prompt
  prompt-learn score <prompt>       Score a prompt (0-1)
  prompt-learn suggest <prompt>     Get quick improvement suggestions
  prompt-learn interactive          Interactive mode
  prompt-learn record-feedback      Record prompt outcome (for learning)
  prompt-learn hook                 Claude Code hook handler (reads stdin)

Options:
  -d, --domain <domain>    Domain context (default: general)
  -n, --iterations <n>     Max optimization iterations (default: 5)
  -v, --verbose            Show detailed output
  --stdin                  Read prompt from stdin

Record Feedback Options:
  -p, --prompt <text>      The prompt text
  -d, --domain <domain>    Domain (code, code_review, testing, general)
  --success true|false     Whether the prompt succeeded
  -q, --quality <0-1>      Quality score (0.0-1.0)
  --id <prompt_id>         Update existing prompt by ID

Examples:
  prompt-learn optimize "Summarize the document"
  prompt-learn optimize -d code_review "Review this PR"
  echo "My prompt" | prompt-learn optimize --stdin
  prompt-learn score "Write a haiku about coding"
  prompt-learn record -p "Fix the bug" --success true -q 0.9
  prompt-learn -i

Hook Mode (for Claude Code integration):
  Configure in ~/.claude/settings.json to auto-record prompts
  `);
}

async function readStdin(): Promise<string> {
  return new Promise((resolve) => {
    let data = '';

    if (process.stdin.isTTY) {
      resolve('');
      return;
    }

    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => {
      resolve(data.trim());
    });

    // Timeout for stdin read
    setTimeout(() => {
      if (data === '') {
        resolve('');
      }
    }, 100);
  });
}

main().catch((error) => {
  console.error('Error:', error.message);
  process.exit(1);
});
