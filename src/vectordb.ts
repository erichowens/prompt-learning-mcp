/**
 * Vector database client for Qdrant
 */

import { QdrantClient } from '@qdrant/js-client-rest';
import type { PromptRecord, PromptMetrics, QdrantSearchResult } from './types.js';
import { EMBEDDING_DIM } from './embeddings.js';

const COLLECTION_NAME = 'prompt_embeddings';

export class VectorDB {
  private client: QdrantClient;
  private initialized: boolean = false;

  constructor(url: string = 'http://localhost:6333') {
    this.client = new QdrantClient({ url });
  }

  /**
   * Initialize the collection if it doesn't exist
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Check if collection exists
      const collections = await this.client.getCollections();
      const exists = collections.collections.some(c => c.name === COLLECTION_NAME);

      if (!exists) {
        // Create collection
        await this.client.createCollection(COLLECTION_NAME, {
          vectors: {
            size: EMBEDDING_DIM,
            distance: 'Cosine'
          }
        });

        // Create indexes for filtering
        await this.client.createPayloadIndex(COLLECTION_NAME, {
          field_name: 'metrics.success_rate',
          field_schema: 'float'
        });

        await this.client.createPayloadIndex(COLLECTION_NAME, {
          field_name: 'domain',
          field_schema: 'keyword'
        });

        await this.client.createPayloadIndex(COLLECTION_NAME, {
          field_name: 'created_at',
          field_schema: 'datetime'
        });

        console.error('[VectorDB] Created collection and indexes');
      }

      this.initialized = true;
    } catch (error) {
      console.error('[VectorDB] Initialization error:', error);
      throw error;
    }
  }

  /**
   * Search for similar prompts
   */
  async search(
    embedding: number[],
    options: {
      topK?: number;
      minPerformance?: number;
      domain?: string;
    } = {}
  ): Promise<QdrantSearchResult[]> {
    await this.initialize();

    const { topK = 10, minPerformance = 0, domain } = options;

    // Build filter
    const must: any[] = [];

    if (minPerformance > 0) {
      must.push({
        key: 'metrics.success_rate',
        range: { gte: minPerformance }
      });
    }

    if (domain) {
      must.push({
        key: 'domain',
        match: { value: domain }
      });
    }

    const filter = must.length > 0 ? { must } : undefined;

    try {
      const results = await this.client.search(COLLECTION_NAME, {
        vector: embedding,
        limit: topK,
        filter,
        with_payload: true
      });

      return results.map(r => ({
        id: String(r.id),
        score: r.score,
        payload: r.payload as Record<string, unknown>
      }));
    } catch (error) {
      console.error('[VectorDB] Search error:', error);
      return [];
    }
  }

  /**
   * Upsert a prompt record
   */
  async upsert(
    id: string,
    embedding: number[],
    record: Omit<PromptRecord, 'id'>
  ): Promise<void> {
    await this.initialize();

    try {
      await this.client.upsert(COLLECTION_NAME, {
        wait: true,
        points: [{
          id,
          vector: embedding,
          payload: record as Record<string, unknown>
        }]
      });
    } catch (error) {
      console.error('[VectorDB] Upsert error:', error);
      throw error;
    }
  }

  /**
   * Get a prompt by ID
   */
  async get(id: string): Promise<PromptRecord | null> {
    await this.initialize();

    try {
      const results = await this.client.retrieve(COLLECTION_NAME, {
        ids: [id],
        with_payload: true,
        with_vector: false
      });

      if (results.length === 0) return null;

      const point = results[0];
      return {
        id: String(point.id),
        ...point.payload as Omit<PromptRecord, 'id'>
      };
    } catch (error) {
      console.error('[VectorDB] Get error:', error);
      return null;
    }
  }

  /**
   * Update metrics for a prompt
   */
  async updateMetrics(id: string, metrics: PromptMetrics): Promise<void> {
    await this.initialize();

    try {
      await this.client.setPayload(COLLECTION_NAME, {
        wait: true,
        points: [id],
        payload: { metrics }
      });
    } catch (error) {
      console.error('[VectorDB] Update metrics error:', error);
      throw error;
    }
  }

  /**
   * Get collection statistics
   */
  async getStats(): Promise<{ total: number; byDomain: Record<string, number> }> {
    await this.initialize();

    try {
      const info = await this.client.getCollection(COLLECTION_NAME);
      const total = info.points_count || 0;

      // Get domain breakdown (simplified - in production would use aggregation)
      return {
        total,
        byDomain: {} // Would require scrolling all points
      };
    } catch (error) {
      console.error('[VectorDB] Stats error:', error);
      return { total: 0, byDomain: {} };
    }
  }

  /**
   * Delete a prompt
   */
  async delete(id: string): Promise<void> {
    await this.initialize();

    try {
      await this.client.delete(COLLECTION_NAME, {
        wait: true,
        points: [id]
      });
    } catch (error) {
      console.error('[VectorDB] Delete error:', error);
      throw error;
    }
  }

  /**
   * Scroll through all prompts (for analytics)
   */
  async scroll(
    limit: number = 100,
    offset?: string
  ): Promise<{ records: PromptRecord[]; nextOffset?: string }> {
    await this.initialize();

    try {
      const result = await this.client.scroll(COLLECTION_NAME, {
        limit,
        offset: offset ? { id: offset } : undefined,
        with_payload: true,
        with_vector: false
      });

      const records = result.points.map(p => ({
        id: String(p.id),
        ...p.payload as Omit<PromptRecord, 'id'>
      }));

      const nextOffset = result.next_page_offset
        ? String(result.points[result.points.length - 1]?.id)
        : undefined;

      return { records, nextOffset };
    } catch (error) {
      console.error('[VectorDB] Scroll error:', error);
      return { records: [] };
    }
  }
}
