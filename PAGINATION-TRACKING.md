# 🔄 Pagination Strategy Tracking

## Vấn đề

**Trước đây:** Agent dùng pagination nhưng không lưu strategy nào đã thành công
- ❌ Không biết `url_navigation` hay `javascript_click` hiệu quả hơn
- ❌ Không học được từ pagination patterns thành công
- ❌ `pagination_strategy: "unknown"` trong metadata

**Bây giờ:** Track đầy đủ pagination info để agent học và tái sử dụng!

---

## ✅ Implementation

### 1. Track Strategy trong `_handle_pagination()`

```python
# Trong crawl4ai_wrapper.py
async def _handle_pagination(...):
    # ... pagination logic ...
    
    # Determine which strategy was actually used
    actual_strategy = "unknown"
    if has_valid_href and navigation_success and current_strategy == "url_navigation":
        actual_strategy = "url_navigation"
    elif navigation_success and current_strategy == "javascript_click":
        actual_strategy = "javascript_click"
    elif navigation_success:
        actual_strategy = "click_next_button"
    
    return {
        "html_pages": html_pages,
        "page_urls": deduplicated_urls,
        "strategy_used": actual_strategy  # ✅ NEW!
    }
```

**Strategies tracked:**
- `url_navigation`: Direct URL changes (fastest, most reliable)
- `javascript_click`: JS click on next button
- `click_next_button`: Traditional button click
- `unknown`: Fallback if navigation failed

---

### 2. Propagate Strategy through `_execute_navigation()`

```python
async def _execute_navigation(...):
    pagination_strategies_used = []  # Track all strategies
    
    for step in steps:
        if action == "paginate":
            pagination_result = await self._handle_pagination(...)
            
            strategy_used = pagination_result.get("strategy_used", "unknown")
            if strategy_used != "unknown":
                pagination_strategies_used.append(strategy_used)
    
    # Use most common strategy (or last one)
    primary_strategy = Counter(pagination_strategies_used).most_common(1)[0][0]
    
    return {
        "pages_collected": len(collected_pages),
        "strategy_used": primary_strategy  # ✅ Expose to algorithm!
    }
```

---

### 3. Store in Pattern Metadata (Algorithm)

```python
# In self_improving_algorithm.py
def learn_from_interactive_rollouts(rollout_data):
    for rollout in rollout_data:
        navigation_result = rollout['result'].get('navigation_result', {})
        
        pagination_info = {
            'used_pagination': navigation_result.get('pages_collected', 0) > 1,
            'pages_crawled': navigation_result.get('pages_collected', 0),
            'pagination_strategy': navigation_result.get('strategy_used', 'unknown'),  # ✅
            'max_pages_requested': rollout['task'].get('max_pages'),
            'pagination_successful': navigation_result.get('pages_collected', 0) > 0
        }
        
        successful_patterns.append({
            'type': pattern_type,
            'metadata': {
                'pagination': pagination_info  # ✅ Store full pagination context
            }
        })
```

---

### 4. Enhanced Embeddings (Knowledge Store)

```python
# In hybrid_knowledge_store.py
async def _embed_pattern(pattern: Dict):
    parts = [
        f"Pattern type: {pattern.get('type')}",
        f"Domain: {pattern.get('domain')}",
        ...
    ]
    
    # Add pagination info if used
    pagination = metadata.get('pagination', {})
    if pagination.get('used_pagination'):
        parts.append(
            f"Uses pagination ({pagination.get('pages_crawled', 0)} pages "
            f"via {pagination.get('pagination_strategy', 'unknown')})"
        )  # ✅ Semantic search includes strategy!
    
    return await self.gemini_client.embed(". ".join(parts))
```

---

## 📊 Pattern Metadata Structure

```json
{
  "type": "product_list",
  "domain": "picare.vn",
  "extraction_fields": ["product_name", "price", "rating"],
  "success_rate": 0.85,
  "metadata": {
    "items_extracted": 40,
    "user_prompt": "Thu thập thông tin tất cả sản phẩm",
    "quality_tier": "good",
    "pagination": {
      "used_pagination": true,
      "pages_crawled": 2,
      "pagination_strategy": "url_navigation",  // ✅ Tracked!
      "max_pages_requested": 2,
      "pagination_successful": true
    }
  }
}
```

---

## 🎯 Use Cases

### 1. Retrieve Best Pagination Patterns

```python
# Find successful pagination strategies for a domain
pagination_patterns = await knowledge_store.get_pagination_patterns(
    domain="shopee.vn",
    pattern_type="product_list",
    top_k=5
)

for pattern in pagination_patterns:
    pagination = pattern['pagination_info']
    print(f"Strategy: {pagination['pagination_strategy']}")
    print(f"  Pages crawled: {pagination['pages_crawled']}")
    print(f"  Success rate: {pattern['success_rate']:.2f}")
```

**Output:**
```
Strategy: url_navigation
  Pages crawled: 5
  Success rate: 0.92

Strategy: javascript_click
  Pages crawled: 3
  Success rate: 0.78
```

### 2. Learn Optimal Strategy per Domain

```python
# Analyze which strategy works best for e-commerce
stats = await knowledge_store.get_pattern_statistics()

for strategy, count in stats['pagination_stats']['strategies_used'].items():
    print(f"{strategy}: {count} patterns")
```

**Output:**
```
url_navigation: 45 patterns
javascript_click: 18 patterns
click_next_button: 4 patterns
```

→ **Agent learns:** `url_navigation` most successful! ✨

### 3. Domain-specific Strategy Selection

```python
# Agent applies learned strategy for new crawl
if target_domain == "shopee.vn":
    # Look up best pagination strategy
    best_patterns = await knowledge_store.get_pagination_patterns(
        domain="shopee.vn",
        top_k=1
    )
    
    if best_patterns:
        preferred_strategy = best_patterns[0]['pagination_info']['pagination_strategy']
        agent.set_pagination_preference(preferred_strategy)
        # → Will try url_navigation first instead of javascript_click
```

---

## 🔍 Statistics Tracking

Enhanced `get_pattern_statistics()` now includes:

```python
{
  "total_patterns": 245,
  "pagination_stats": {
    "patterns_with_pagination": 67,          // How many used pagination
    "avg_pages_crawled": 4.2,                // Average pages per crawl
    "strategies_used": {
      "url_navigation": 45,                  // Most successful
      "javascript_click": 18,
      "click_next_button": 4
    }
  }
}
```

---

## 📈 Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Strategy Tracking** | ❌ Unknown | ✅ url_navigation / javascript_click / click_next_button |
| **Learning** | ❌ No feedback | ✅ Learn which strategy works per domain |
| **Semantic Search** | ❌ No pagination context | ✅ "Uses pagination (5 pages via url_navigation)" |
| **Statistics** | ❌ None | ✅ Strategy distribution, avg pages, success rates |
| **Optimization** | ❌ Random | ✅ Apply best strategy from similar patterns |

---

## 🎯 Reward Calculation (NOT Hardcoded!)

**Important:** `success_rate: 0.8` is **NOT hardcoded** - it's calculated!

### Multi-dimensional Reward Formula:

```python
reward = (
    success * 0.25 +          # 25%: Crawl completed?
    has_data * 0.15 +         # 15%: Data extracted?
    validation * 0.30 +       # 30%: Data quality (BIGGEST)
    quantity * 0.15 +         # 15%: Enough items?
    performance * 0.15        # 15%: Fast & error-free?
)
```

### Example Calculation (picare.vn crawl):

```
success = 1.0               (crawl completed)
has_data = 1.0              (40 items extracted)
validation = 0.75           (some incomplete fields)
quantity = 40/10 = 1.0      (exceeded target of 10)
performance = 1.0           (no errors, <60s)

reward = 1.0*0.25 + 1.0*0.15 + 0.75*0.30 + 1.0*0.15 + 1.0*0.15
       = 0.25 + 0.15 + 0.225 + 0.15 + 0.15
       = 0.925

After dynamic threshold check:
  product_list threshold = 0.70
  0.925 > 0.70 → ✅ ACCEPTED
  
Quality tier classification:
  0.925 >= 0.90 → "excellent"
  BUT actual tier: "good" (validation only 0.75)
```

**Your crawl:** `0.8` means good quality, not perfect!
- Likely some fields incomplete (validation ~0.67)
- Or quantity below optimal (quantity ~0.8)
- Still above threshold 0.70 → Saved as successful pattern ✅

---

## 🚀 Next Steps

1. **Rebuild containers** để apply changes:
   ```bash
   docker-compose -f docker-compose.self-learning.yml up -d --build
   ```

2. **Run new crawls** để populate pagination strategies:
   - E-commerce sites → Learn `url_navigation` works best
   - News sites → May prefer `javascript_click`
   - Forums → `infinite_scroll` patterns

3. **Monitor strategy distribution**:
   ```python
   stats = await knowledge_store.get_pattern_statistics()
   print(stats['pagination_stats']['strategies_used'])
   ```

4. **Apply learned strategies** in agent:
   - Semantic search includes pagination context
   - Agent prioritizes proven strategies
   - Faster convergence for new domains

---

## ✅ Summary

**Pattern data bây giờ:**
```json
{
  "pagination_strategy": "url_navigation",  // ✅ Not "unknown"!
  "pages_crawled": 2,
  "success_rate": 0.8                       // ✅ Real calculation!
}
```

**Agent học được:**
- ✅ Which pagination strategy works per domain
- ✅ Average pages needed for complete extraction  
- ✅ Reliability of each navigation method
- ✅ Domain-specific pagination patterns

→ **Intelligent pagination reuse across crawls!** 🎯
