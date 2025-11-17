# Multi-Model Routing Implementation

## Overview

Extended the `GeminiClient` with intelligent multi-model routing to optimize cost, latency, and quality across different task types in the Self-Learning Web Crawler Agent.

## Architecture

### Model Selection Strategy

```
Task Type → Mode → Complexity → Rate Limits → Model Selection
```

#### Available Models

| Model | Cost (Input/Output per 1M tokens) | Latency | Use Cases |
|-------|----------------------------------|---------|-----------|
| `gemini-2.0-flash-exp` | $0.10 / $0.40 | ~500ms | Fast crawling, simple extraction |
| `gemini-2.0-flash-lite` | $0.05 / $0.20 | ~300ms | Rate limit fallback, cached patterns |
| `gemini-2.5-flash-exp` | $0.15 / $0.60 | ~800ms | Feedback, prompts, RL decisions, thinking |
| `gemini-2.5-pro-exp` | $1.25 / $5.00 | ~2000ms | Deep analysis, clustering (training only) |
| `learnlm-2.0-flash-experimental` | $0.10 / $0.40 | ~600ms | Pedagogical clarifications |

### Routing Logic

#### By Task Type

- **`crawl`**: Uses `2.0-flash` for speed and volume
- **`feedback`**: Uses `2.5-flash` for reasoning and interpretation
- **`clarification`**: Uses `learnlm` for pedagogical Q&A
- **`prompt_gen`**: Uses `2.5-flash` for creative/coding tasks
- **`analysis`**: Uses `2.5-flash` (prod) or `2.5-pro` (training)
- **`rl_decide`**: Uses `2.5-flash` for agentic planning

#### By Mode

- **Production**: Prioritize speed and cost (`2.0-flash` default)
- **Training**: Prioritize quality and learning (`2.5-flash` default)

#### By Complexity

- **Simple tasks** (<5,000 tokens): Standard model for task type
- **Complex tasks** (>5,000 tokens): Escalate to `2.5-pro` in training mode

#### By Rate Limits

- **TPM/RPM >80%**: Automatic fallback to `2.0-flash-lite`
- **429 errors**: Retry with cheaper model + exponential backoff

## API Reference

### `route_model(task_type, input_data, current_metrics)`

Main routing method that selects and executes the optimal model.

**Parameters:**
- `task_type` (str): One of `'crawl'`, `'feedback'`, `'clarification'`, `'prompt_gen'`, `'analysis'`, `'rl_decide'`
- `input_data` (dict): 
  - `prompt` (str): The prompt text
  - `json_mode` (bool, optional): Request JSON response
- `current_metrics` (dict):
  - `tokens_used` (int): Running total
  - `cost_usd` (float): Running cost
  - `mode` (str): `'production'` or `'training'`

**Returns:**
- Tuple of `(model_id, response_text, updated_metrics)`

**Example:**
```python
metrics = {"tokens_used": 0, "cost_usd": 0.0, "mode": "production"}
model_id, response, metrics = await client.route_model(
    task_type="feedback",
    input_data={
        "prompt": "Interpret: 'Missing prices'",
        "json_mode": True
    },
    current_metrics=metrics
)
print(f"Used {model_id}, Cost: ${metrics['last_request_cost']:.6f}")
```

### Helper Methods

#### `_estimate_tokens(data)`
Estimates token count from strings, dicts, or lists (1 token ≈ 4 chars).

#### `_check_rate_limits(estimated_tokens)`
Returns `(is_safe, warning_msg)` based on sliding window tracking.

#### `_select_model_for_task(task_type, estimated_tokens, mode, current_cost)`
Applies routing logic and returns selected model ID.

## Configuration

### In `config.py`

```python
@dataclass
class GeminiConfig:
    ROUTING_ENABLED: bool = True
    MAX_RPM: int = 15  # Free tier limit
    MAX_TPM: int = 250000
    TPM_WARNING_THRESHOLD: float = 0.8  # Warn at 80%
    COMPLEX_TASK_TOKEN_THRESHOLD: int = 5000
    MODELS: dict = None  # Auto-initialized with model definitions
```

### Model Definitions

Models are auto-initialized in `__post_init__()` with:
- Cost per 1M input/output tokens
- Average latency
- Rate limits (RPM/TPM)

## Statistics Tracking

Enhanced `get_stats()` includes:

```python
{
    "model_usage": {
        "gemini-2.5-flash-exp": 5,
        "gemini-2.0-flash-exp": 12,
        ...
    },
    "rate_limit_warnings": 2,
    "model_fallbacks": 1,
    "routing_enabled": True,
    ...existing stats...
}
```

## Error Handling

### Rate Limit (429) Errors
1. Detect `429` or `quota` in error message
2. Fallback to `gemini-2.0-flash-lite`
3. Exponential backoff: 2^retry_count seconds
4. Max 2 retries

### Other Errors
1. Retry once with 1-second delay
2. After max retries, raise exception
3. Log all errors with model context

## Integration with Existing Features

### Caching
- Works seamlessly with existing `TTLCache`
- Cache key includes model selection parameters
- Routing skipped if cached response found

### Batching
- Routing disabled for batched requests
- Models with `json_mode` skip batching (line 192 check)

### Local LLM Fallback
- Local LLM check happens before routing
- Can still use routing for Gemini fallback

## Usage Examples

See `example_routing_usage.py` for:
1. Basic task routing across all types
2. Production vs training mode differences
3. Rate limit handling simulation
4. Cost tracking and statistics

## Performance Considerations

### Latency
- **Production crawls**: 500ms avg (2.0-flash)
- **Training analysis**: 2000ms max (2.5-pro)
- **Feedback loops**: 800ms avg (2.5-flash)

### Cost Optimization
- **10,000 crawls/day**: ~$50/month (2.0-flash at 500 tokens avg)
- **100 analysis tasks/day**: ~$6/month (2.5-pro at 5000 tokens avg)
- **Rate limit fallback**: ~50% cost reduction (lite vs flash)

### Rate Limits (Free Tier)
- **Max sustainable load**: ~900 requests/hour (15 RPM)
- **Burst handling**: Automatic lite fallback prevents 429s
- **Token budget**: 250K TPM = ~1M chars/min

## Monitoring

Track these metrics in production:

```python
stats = client.get_stats()
assert stats['rate_limit_warnings'] < 10, "Too many warnings"
assert stats['model_fallbacks'] < 5, "Too many fallbacks"
assert stats['cache_hit_rate'] > 0.3, "Low cache efficiency"
```

## Future Enhancements

1. **Dynamic rate limit adjustment**: Learn optimal TPM/RPM from API responses
2. **Cost-based routing**: Switch to cheaper models when budget exceeded
3. **A/B testing**: Compare quality across models for same task
4. **Fine-tuned routing**: Train a small classifier for model selection
5. **Gemini 2.5 Flash Live**: Stream responses for real-time UI updates

## Migration Guide

### From Old Code
```python
# Before
response = await gemini_client.generate(prompt)

# After (with routing)
model_id, response, metrics = await gemini_client.route_model(
    task_type="crawl",
    input_data={"prompt": prompt},
    current_metrics=metrics
)
```

### Backward Compatibility
- `generate()` method unchanged
- Routing optional via `ROUTING_ENABLED = False`
- Falls back to default model if routing unavailable

## Testing

Run the example:
```bash
export GEMINI_API_KEY="your-key"
cd /root/projects/crawldata/MCP-Servers/self-learning-agent
python example_routing_usage.py
```

Expected output:
- 6 different task examples
- Model selection logs (`🎯 Routing...`)
- Cost tracking per request
- Final statistics summary

## Troubleshooting

### "Model X not available"
- Check `config.MODELS` contains the model ID
- Verify model exists in Gemini API (some are experimental)
- Falls back to `config.MODEL` automatically

### High rate limit warnings
- Increase `MAX_RPM`/`MAX_TPM` for paid tier
- Reduce `TPM_WARNING_THRESHOLD` to trigger earlier
- Add delays between requests

### Unexpected model selection
- Check task type spelling (must be exact)
- Verify mode is `'production'` or `'training'`
- Review complexity threshold settings

## License

Same as parent project (Self-Learning Agent System)
