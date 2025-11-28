import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/integration/**/*.test.ts'],
    testTimeout: 60000,
    // Integration tests run sequentially
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: true
      }
    }
  }
});
