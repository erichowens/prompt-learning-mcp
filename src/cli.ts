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
import { PromptOptimizer } from './optimizer.js';

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

Options:
  -d, --domain <domain>    Domain context (default: general)
  -n, --iterations <n>     Max optimization iterations (default: 5)
  -v, --verbose            Show detailed output
  --stdin                  Read prompt from stdin

Examples:
  prompt-learn optimize "Summarize the document"
  prompt-learn optimize -d code_review "Review this PR"
  echo "My prompt" | prompt-learn optimize --stdin
  prompt-learn score "Write a haiku about coding"
  prompt-learn -i
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
