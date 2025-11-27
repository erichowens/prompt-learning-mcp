# Prompt Learning MCP Server

**Stateful prompt optimization that learns over time.**

An MCP (Model Context Protocol) server that optimizes your prompts using research-backed techniques (APE, OPRO, DSPy patterns) and learns from performance history via embedding-based retrieval.

## Features

- **🧠 Smart Optimization**: Uses actual LLM-based evaluation, not heuristics
- **📚 Learns Over Time**: Stores prompt performance in vector database
- **🔍 RAG-Powered**: Retrieves similar high-performing prompts
- **⚡ Pattern-Based Quick Wins**: Instant improvements without API calls
- **📊 Analytics**: Track what's working across domains

## Quick Install

```bash
curl -fsSL https://someclaudeskills.com/install/prompt-learning.sh | bash
```

Or manually:

```bash
cd ~/mcp-servers/prompt-learning
npm install
npm run build
npm run setup
```

## Requirements

- Node.js 18+
- Docker (for Qdrant and Redis)
- OpenAI API key (for embeddings)

## Usage

Once installed, use these tools in Claude Code:

### `optimize_prompt`

Optimize a prompt using pattern-based and RAG-based techniques:

```
"optimize this prompt: summarize the document"
```

Returns the optimized prompt with improvement details.

### `retrieve_prompts`

Find similar high-performing prompts:

```
"find similar prompts for: code review feedback"
```

### `record_feedback`

Record how a prompt performed (enables learning):

```
"record that my last prompt succeeded with quality score 0.9"
```

### `suggest_improvements`

Get quick suggestions without full optimization:

```
"suggest improvements for this prompt: [your prompt]"
```

### `get_analytics`

View performance trends:

```
"show prompt analytics for the last 30 days"
```

## How It Works

### Cold Start (No History)

1. **Pattern-based improvements**: Adds structure, chain-of-thought, constraints
2. **OPRO-style iteration**: LLM generates candidates, evaluates, selects best
3. **APE-style generation**: Creates multiple instruction variants

### Warm Start (With History)

1. **Embed the prompt**: Creates vector representation
2. **Retrieve similar**: Finds high-performing prompts from database
3. **Learn from winners**: Synthesizes improvements from what worked
4. **Iterate with feedback**: Uses evaluation to guide optimization

### Evaluation

All prompts are scored by an LLM evaluator on:
- **Clarity** (25%): How unambiguous
- **Specificity** (25%): Appropriate guidance level
- **Completeness** (20%): Covers all requirements
- **Structure** (15%): Well-organized
- **Effectiveness** (15%): Likely to produce desired output

## Architecture

```
Claude Code
     │
     │ MCP Protocol
     ▼
┌─────────────────────────────┐
│  prompt-learning MCP Server │
│                             │
│  Tools:                     │
│  • optimize_prompt          │
│  • retrieve_prompts         │
│  • record_feedback          │
│  • suggest_improvements     │
│  • get_analytics            │
└─────────┬───────────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐   ┌───────┐
│Qdrant │   │ Redis │
│(Vector│   │(Cache)│
│  DB)  │   │       │
└───────┘   └───────┘
```

## Configuration

### Claude Code Config (~/.claude.json)

```json
{
  "mcpServers": {
    "prompt-learning": {
      "command": "node",
      "args": ["~/mcp-servers/prompt-learning/dist/index.js"],
      "env": {
        "VECTOR_DB_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_DB_URL` | `http://localhost:6333` | Qdrant server URL |
| `REDIS_URL` | `redis://localhost:6379` | Redis server URL |
| `OPENAI_API_KEY` | (required) | For embeddings |

## Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Build for production
npm run build

# Run setup (starts Docker, initializes DB)
npm run setup

# Run tests
npm test
```

## Troubleshooting

### MCP Server Not Starting

Check Docker containers are running:
```bash
docker ps | grep prompt-learning
```

### Vector DB Connection Failed

```bash
# Check Qdrant health
curl http://localhost:6333/health

# Restart Qdrant
docker restart prompt-learning-qdrant
```

### No Improvements Seen

- Ensure OPENAI_API_KEY is set correctly
- Check Claude Code logs: `~/.claude/logs/mcp.log`
- Try with a simple prompt first

## Research Foundation

This server implements techniques from:

- **APE** (Zhou et al., 2022): Automatic Prompt Engineer
- **OPRO** (Yang et al., 2023): Optimization by Prompting
- **DSPy** (Khattab et al., 2023): Programmatic prompt optimization
- **Contextual Retrieval** (Anthropic, 2024): Enhanced embedding retrieval

## License

MIT

## Links

- **Documentation**: https://www.someclaudeskills.com/skills/automatic-stateful-prompt-improver
- **Skill Definition**: Part of the [Some Claude Skills](https://www.someclaudeskills.com) collection
- **GitHub**: https://github.com/erichowens/prompt-learning-mcp
- **Issues**: https://github.com/erichowens/prompt-learning-mcp/issues
