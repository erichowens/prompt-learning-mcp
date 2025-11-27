/**
 * Embedding generation and caching
 */

import OpenAI from 'openai';
import { createHash } from 'crypto';
import type { Redis } from 'ioredis';
import type { EmbeddingResult } from './types.js';

const EMBEDDING_MODEL = 'text-embedding-3-small';
const EMBEDDING_DIM = 1536;
const CACHE_TTL_HOURS = 24;

export { EMBEDDING_DIM };

export class EmbeddingService {
  private openai: OpenAI;
  private redis: Redis | null;
  private cache: Map<string, number[]> = new Map();

  constructor(openai: OpenAI, redis: Redis | null = null) {
    this.openai = openai;
    this.redis = redis;
  }

  /**
   * Generate embedding with optional caching
   */
  async embed(text: string, useCache: boolean = true): Promise<EmbeddingResult> {
    const cacheKey = this.getCacheKey(text);

    // Check in-memory cache first
    if (useCache && this.cache.has(cacheKey)) {
      return {
        embedding: this.cache.get(cacheKey)!,
        tokens_used: 0
      };
    }

    // Check Redis cache
    if (useCache && this.redis) {
      try {
        const cached = await this.redis.get(`emb:${cacheKey}`);
        if (cached) {
          const embedding = JSON.parse(cached);
          this.cache.set(cacheKey, embedding);
          return { embedding, tokens_used: 0 };
        }
      } catch (e) {
        // Cache miss or error, continue to generate
      }
    }

    // Generate new embedding
    const response = await this.openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: text,
    });

    const embedding = response.data[0].embedding;
    const tokens_used = response.usage?.total_tokens || 0;

    // Cache the result
    this.cache.set(cacheKey, embedding);
    if (this.redis) {
      try {
        await this.redis.setex(
          `emb:${cacheKey}`,
          CACHE_TTL_HOURS * 3600,
          JSON.stringify(embedding)
        );
      } catch (e) {
        // Cache write failed, continue anyway
      }
    }

    return { embedding, tokens_used };
  }

  /**
   * Generate contextual embedding with domain metadata
   */
  async embedContextual(
    text: string,
    domain: string,
    taskType: string
  ): Promise<EmbeddingResult> {
    // Add context to improve retrieval (following Anthropic's contextual retrieval pattern)
    const contextualizedText = `Domain: ${domain}\nTask: ${taskType}\n\n${text}`;
    return this.embed(contextualizedText);
  }

  /**
   * Batch embed multiple texts
   */
  async embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
    const results: EmbeddingResult[] = [];
    const uncached: { index: number; text: string }[] = [];

    // Check cache first
    for (let i = 0; i < texts.length; i++) {
      const cacheKey = this.getCacheKey(texts[i]);
      if (this.cache.has(cacheKey)) {
        results[i] = { embedding: this.cache.get(cacheKey)!, tokens_used: 0 };
      } else {
        uncached.push({ index: i, text: texts[i] });
      }
    }

    // Batch generate uncached embeddings
    if (uncached.length > 0) {
      const response = await this.openai.embeddings.create({
        model: EMBEDDING_MODEL,
        input: uncached.map(u => u.text),
      });

      for (let i = 0; i < uncached.length; i++) {
        const embedding = response.data[i].embedding;
        const cacheKey = this.getCacheKey(uncached[i].text);
        this.cache.set(cacheKey, embedding);
        results[uncached[i].index] = {
          embedding,
          tokens_used: Math.floor((response.usage?.total_tokens || 0) / uncached.length)
        };
      }
    }

    return results;
  }

  /**
   * Compute cosine similarity between two embeddings
   */
  cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) throw new Error('Embedding dimensions must match');

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private getCacheKey(text: string): string {
    return createHash('sha256').update(text).digest('hex').slice(0, 16);
  }
}
