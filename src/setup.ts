#!/usr/bin/env tsx
/**
 * Setup script for Prompt Learning MCP Server
 *
 * This script:
 * 1. Checks prerequisites (Docker, Node, API keys)
 * 2. Starts Qdrant and Redis containers
 * 3. Initializes the vector collection
 * 4. Configures Claude Code
 */

import { execSync, spawn } from 'child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import { QdrantClient } from '@qdrant/js-client-rest';
import { Redis } from 'ioredis';

const QDRANT_PORT = 6333;
const REDIS_PORT = 6379;
const COLLECTION_NAME = 'prompt_embeddings';
const EMBEDDING_DIM = 1536;

interface SetupOptions {
  skipDocker?: boolean;
  skipClaude?: boolean;
  verbose?: boolean;
}

async function main() {
  const args = process.argv.slice(2);
  const options: SetupOptions = {
    skipDocker: args.includes('--skip-docker'),
    skipClaude: args.includes('--skip-claude'),
    verbose: args.includes('--verbose') || args.includes('-v')
  };

  console.log('🚀 Prompt Learning MCP Server Setup\n');

  // Step 1: Check prerequisites
  console.log('📋 Checking prerequisites...');
  await checkPrerequisites(options);

  // Step 2: Start Docker containers
  if (!options.skipDocker) {
    console.log('\n🐳 Starting Docker containers...');
    await startDockerContainers(options);
  }

  // Step 3: Initialize vector database
  console.log('\n📦 Initializing vector database...');
  await initializeVectorDB(options);

  // Step 4: Test Redis connection
  console.log('\n🔴 Testing Redis connection...');
  await testRedis(options);

  // Step 5: Configure Claude Code
  if (!options.skipClaude) {
    console.log('\n⚙️  Configuring Claude Code...');
    await configureClaudeCode(options);
  }

  console.log('\n✅ Setup complete!\n');
  console.log('Next steps:');
  console.log('  1. Restart Claude Code to load the new MCP server');
  console.log('  2. Test with: "optimize this prompt: summarize the document"');
  console.log('  3. Check logs at: ~/.claude/logs/mcp.log\n');
}

async function checkPrerequisites(options: SetupOptions) {
  const checks = [
    { name: 'Node.js 18+', cmd: 'node --version', pattern: /v(\d+)\./, minVersion: 18 },
    { name: 'Docker', cmd: 'docker --version', pattern: /Docker version/ },
    { name: 'OPENAI_API_KEY', env: 'OPENAI_API_KEY' }
  ];

  for (const check of checks) {
    if (check.cmd) {
      try {
        const output = execSync(check.cmd, { encoding: 'utf-8' });
        if (check.pattern && check.minVersion) {
          const match = output.match(check.pattern);
          if (match && parseInt(match[1]) >= check.minVersion) {
            console.log(`  ✓ ${check.name}: ${output.trim()}`);
          } else {
            throw new Error(`Version too low: ${output.trim()}`);
          }
        } else if (check.pattern && check.pattern.test(output)) {
          console.log(`  ✓ ${check.name}: Found`);
        }
      } catch (e) {
        console.error(`  ✗ ${check.name}: Not found or version too low`);
        process.exit(1);
      }
    } else if (check.env) {
      if (process.env[check.env]) {
        console.log(`  ✓ ${check.name}: Set`);
      } else {
        console.error(`  ✗ ${check.name}: Not set`);
        console.error(`     Please set: export ${check.env}=your-key`);
        process.exit(1);
      }
    }
  }
}

async function startDockerContainers(options: SetupOptions) {
  // Check if containers already running
  const runningContainers = execSync('docker ps --format "{{.Names}}"', { encoding: 'utf-8' });

  // Start Qdrant
  if (!runningContainers.includes('prompt-learning-qdrant')) {
    console.log('  Starting Qdrant...');
    try {
      // Remove existing stopped container if present
      execSync('docker rm -f prompt-learning-qdrant 2>/dev/null || true', { encoding: 'utf-8' });

      execSync(`docker run -d \
        --name prompt-learning-qdrant \
        -p ${QDRANT_PORT}:6333 \
        -v prompt_learning_qdrant:/qdrant/storage \
        qdrant/qdrant`, { encoding: 'utf-8' });
      console.log('  ✓ Qdrant started on port', QDRANT_PORT);
    } catch (e) {
      console.error('  ✗ Failed to start Qdrant:', e);
      process.exit(1);
    }
  } else {
    console.log('  ✓ Qdrant already running');
  }

  // Start Redis
  if (!runningContainers.includes('prompt-learning-redis')) {
    console.log('  Starting Redis...');
    try {
      execSync('docker rm -f prompt-learning-redis 2>/dev/null || true', { encoding: 'utf-8' });

      execSync(`docker run -d \
        --name prompt-learning-redis \
        -p ${REDIS_PORT}:6379 \
        -v prompt_learning_redis:/data \
        redis:alpine redis-server --appendonly yes`, { encoding: 'utf-8' });
      console.log('  ✓ Redis started on port', REDIS_PORT);
    } catch (e) {
      console.error('  ✗ Failed to start Redis:', e);
      process.exit(1);
    }
  } else {
    console.log('  ✓ Redis already running');
  }

  // Wait for services to be ready
  console.log('  Waiting for services to be ready...');
  await sleep(3000);
}

async function initializeVectorDB(options: SetupOptions) {
  const client = new QdrantClient({ url: `http://localhost:${QDRANT_PORT}` });

  try {
    // Check connection by getting collections (simpler health check)
    const collections = await client.getCollections();
    console.log('  ✓ Connected to Qdrant');

    // Check if collection exists
    const exists = collections.collections.some(c => c.name === COLLECTION_NAME);

    if (exists) {
      const info = await client.getCollection(COLLECTION_NAME);
      console.log(`  ✓ Collection exists (${info.points_count || 0} prompts stored)`);
      return;
    }

    // Create collection
    console.log('  Creating collection...');
    await client.createCollection(COLLECTION_NAME, {
      vectors: {
        size: EMBEDDING_DIM,
        distance: 'Cosine'
      }
    });

    // Create indexes using the proper API
    console.log('  Creating indexes...');
    await client.createPayloadIndex(COLLECTION_NAME, {
      field_name: 'metrics.success_rate',
      field_schema: 'float'
    } as any);

    await client.createPayloadIndex(COLLECTION_NAME, {
      field_name: 'domain',
      field_schema: 'keyword'
    } as any);

    await client.createPayloadIndex(COLLECTION_NAME, {
      field_name: 'created_at',
      field_schema: 'datetime'
    } as any);

    console.log('  ✓ Collection and indexes created');
  } catch (e) {
    console.error('  ✗ Failed to initialize Qdrant:', e);
    process.exit(1);
  }
}

async function testRedis(options: SetupOptions) {
  const redis = new Redis({
    port: REDIS_PORT,
    host: 'localhost',
    maxRetriesPerRequest: 3
  });

  try {
    const pong = await redis.ping();
    if (pong === 'PONG') {
      console.log('  ✓ Redis connection successful');
    }
  } catch (e) {
    console.error('  ✗ Failed to connect to Redis:', e);
    console.error('    Will continue - caching will be disabled');
  } finally {
    redis.disconnect();
  }
}

async function configureClaudeCode(options: SetupOptions) {
  const claudeConfigPath = join(homedir(), '.claude.json');
  const serverPath = join(process.cwd(), 'dist', 'index.js');

  // Read existing config or create new
  let config: any = {};
  if (existsSync(claudeConfigPath)) {
    try {
      config = JSON.parse(readFileSync(claudeConfigPath, 'utf-8'));
    } catch (e) {
      console.log('  Creating new Claude config...');
    }
  }

  // Ensure mcpServers object exists
  if (!config.mcpServers) {
    config.mcpServers = {};
  }

  // Add our server
  config.mcpServers['prompt-learning'] = {
    command: 'node',
    args: [serverPath],
    env: {
      VECTOR_DB_URL: `http://localhost:${QDRANT_PORT}`,
      REDIS_URL: `redis://localhost:${REDIS_PORT}`,
      OPENAI_API_KEY: process.env.OPENAI_API_KEY
    }
  };

  // Write config
  writeFileSync(claudeConfigPath, JSON.stringify(config, null, 2));
  console.log('  ✓ Claude Code configured');
  console.log(`    Config: ${claudeConfigPath}`);
  console.log(`    Server: ${serverPath}`);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Run
main().catch(e => {
  console.error('Setup failed:', e);
  process.exit(1);
});
