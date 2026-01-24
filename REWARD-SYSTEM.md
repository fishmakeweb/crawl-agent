# 🎯 Reward System cho Intelligent Pattern Learning

## Vấn đề với Reward đơn giản

**Trước đây:**
```python
# Chỉ có binary reward
if result.get("success"):
    reward = 1.0
else:
    reward = 0.0
```

**Vấn đề:**
- ❌ Không phân biệt quality (1 item vs 100 items đều = 1.0)
- ❌ Không phân biệt pattern types (article vs product cần quality khác nhau)
- ❌ Không reward incremental improvements
- ❌ Threshold cứng nhắc (0.7 cho tất cả patterns)

---

## ✅ Giải pháp: Multi-dimensional Reward Function

### 1. Reward Components (5 dimensions)

```python
reward = (
    success * 0.25 +          # 25%: Crawl completed?
    has_data * 0.15 +         # 15%: Data extracted?
    validation * 0.30 +       # 30%: Data quality (BIGGEST)
    quantity * 0.15 +         # 15%: Enough items?
    performance * 0.15        # 15%: Fast & error-free?
)
```

### Breakdown:

#### Component 1: Success (25%)
```python
success = 1.0 if result.get("success") else 0.0
```
- Crawl hoàn thành không crash
- Cơ bản nhất

#### Component 2: Data Presence (15%)
```python
has_data = 1.0 if extracted_data else 0.0
```
- Có extract được data không?

#### Component 3: Validation (30%) - **QUAN TRỌNG NHẤT**

```python
def _validate_extraction(extracted, schema):
    for item in extracted:
        # 40%: Field presence
        fields_present = sum(1 for field in required if field in item)
        field_score = fields_present / len(required)
        
        # 30%: Completeness (non-null values)
        fields_complete = sum(1 for field in required 
                             if field in item and item[field] not in [None, "", []])
        completeness_score = fields_complete / len(required)
        
        # 30%: Data quality
        quality_score = assess_data_quality(item)
        
        item_score = field_score*0.4 + completeness_score*0.3 + quality_score*0.3
    
    return avg(item_scores)
```

**Data Quality Checks:**
- ✅ Price fields: numeric, positive
- ✅ Rating fields: in range [0-5]
- ✅ String fields: not generic ("N/A", "null", "unknown")
- ✅ Dates: parseable
- ✅ No empty values

#### Component 4: Quantity (15%)

```python
# Different targets for different pattern types
if pattern_type == "product_list":
    target = 10  # E-commerce has many items
elif pattern_type == "article_extraction":
    target = 5   # Articles fewer per page
else:
    target = 5

quantity_score = min(1.0, len(extracted) / target)
```

#### Component 5: Efficiency (15%) - **PAGINATION-AWARE!**

```python
no_errors = 1.0 if not errors else 0.0

# NEW: Calculate time per page (fair for pagination)
pages_collected = max(1, navigation_result.get('pages_collected', 1))
time_per_page = execution_time_ms / pages_collected

# Dynamic penalty based on crawl type
if pages_collected == 1:
    # Single page: allow up to 30s
    if time_per_page > 30000:
        time_penalty = min(0.3, (time_per_page - 30000) / 60000)
    else:
        time_penalty = 0.0
else:
    # Multi-page: expect efficiency (< 15s per page)
    if time_per_page > 15000:
        time_penalty = min(0.4, (time_per_page - 15000) / 45000)
    elif time_per_page < 10000:
        time_penalty = -0.1  # 10% BONUS for fast pagination!
    else:
        time_penalty = 0.0

efficiency = no_errors * (1.0 - time_penalty)
```

**Benefits:**
- ✅ Fair comparison: 1-page vs 10-page crawls
- ✅ Rewards fast pagination strategies
- ✅ Penalizes inefficient multi-page crawls
- ✅ Bonus for crawls < 10s/page

---

## 2. Dynamic Thresholds per Pattern Type

**Vấn đề:** Không phải tất cả patterns đều cần quality giống nhau!

### Threshold Table

| Pattern Type | Min Items | Success Threshold | Excellent Threshold |
|--------------|-----------|-------------------|---------------------|
| `article_extraction` | 1 | **0.80** (cao) | 0.95 |
| `contact_info` | 1 | **0.75** (cao) | 0.95 |
| `price_extraction` | 3 | **0.75** (cao) | 0.90 |
| `product_list` | 5 | **0.70** (trung bình) | 0.90 |
| `review_extraction` | 5 | **0.65** (thấp) | 0.85 |
| `product_with_reviews` | 3 | **0.65** (thấp) | 0.85 |
| `generic_extraction` | 1 | **0.65** (thấp) | 0.85 |

**Tại sao khác nhau?**

- **Article Extraction** (0.80): 
  - Ít items (1-5 articles)
  - Cần complete & accurate
  - Content chất lượng cao
  
- **Product List** (0.70):
  - Nhiều items (10-50 products)
  - Chấp nhận một vài incomplete
  - Quantity > Quality

- **Review Extraction** (0.65):
  - Rất nhiều items (50-100 reviews)
  - Bulk data
  - Ok nếu một vài reviews thiếu fields

---

## 3. Quality Tiers

Mỗi pattern được classify vào tier:

```python
def _classify_quality_tier(reward, threshold):
    if reward >= threshold["excellent"]:      # ≥ 0.90
        return "excellent"
    elif reward >= threshold["success"]:      # ≥ 0.70
        return "good"
    elif reward >= threshold["failure"]:      # ≥ 0.40
        return "acceptable"
    else:                                     # < 0.40
        return "poor"
```

**Lưu trong metadata:**
```json
{
  "type": "product_list",
  "success_rate": 0.85,
  "metadata": {
    "quality_tier": "good",
    "items_extracted": 45
  }
}
```

---

## 4. Ví dụ Thực tế

### Example 1: Product List - Excellent Quality (Fast Pagination)

**Input:**
```json
{
  "url": "https://shopee.vn/products",
  "extraction_schema": {
    "required": ["product_name", "price", "rating"]
  },
  "max_pages": 2
}
```

**Output:**
```json
{
  "success": true,
  "data": [
    {"product_name": "iPhone 15", "price": 999.99, "rating": 4.5},
    {"product_name": "MacBook Pro", "price": 2499.99, "rating": 4.8},
    // ... 38 more items (total 40 from 2 pages)
  ],
  "navigation_result": {
    "pages_collected": 2,
    "strategy_used": "url_navigation"
  },
  "execution_time_ms": 18000  // 18 seconds total
}
```

**Reward Calculation:**
```
success = 1.0           (crawl completed)
has_data = 1.0          (40 items extracted)
validation:
  - field_presence: 1.0  (all items have all 3 fields)
  - completeness: 1.0    (no null/empty)
  - quality: 0.95        (prices valid, ratings in [0-5])
  → validation = 0.98

quantity = 40/10 = 1.0  (target: 10 for product_list)

efficiency:
  - no_errors = 1.0
  - pages_collected = 2
  - time_per_page = 18000/2 = 9000ms (9s per page)
  - time_per_page < 10000 → time_penalty = -0.1 (BONUS!)
  - efficiency = 1.0 * (1.0 - (-0.1)) = 1.1 → clamped to 1.0
  → efficiency = 1.0

REWARD = 1.0*0.25 + 1.0*0.15 + 0.98*0.30 + 1.0*0.15 + 1.0*0.15
       = 0.25 + 0.15 + 0.294 + 0.15 + 0.15
       = 0.994

Tier: "excellent" (≥ 0.90)
```

**Why excellent?** Fast pagination (9s/page) + high validation!

### Example 2: Product List - Good Quality (Slower Pagination)

**Output:**
```json
{
  "success": true,
  "data": [
    {"product_name": "iPhone 15", "price": 999.99, "rating": null},
    {"product_name": "MacBook Pro", "price": null, "rating": 4.8},
    {"product_name": null, "price": 1299.99, "rating": 4.2},
    // ... 17 more items (total 20 from 2 pages)
  ],
  "navigation_result": {
    "pages_collected": 2,
    "strategy_used": "javascript_click"
  },
  "execution_time_ms": 40000  // 40 seconds total
}
```

**Reward Calculation:**
```
success = 1.0
has_data = 1.0
validation:
  - field_presence: 0.67 (some fields missing)
  - completeness: 0.53   (many null values)
  - quality: 0.60        (some prices/ratings missing)
  → validation = 0.60

quantity = 20/10 = 1.0

efficiency:
  - no_errors = 1.0
  - pages_collected = 2
  - time_per_page = 40000/2 = 20000ms (20s per page)
  - time_per_page > 15000 → time_penalty = (20000-15000)/45000 = 0.11
  - efficiency = 1.0 * (1.0 - 0.11) = 0.89
  → efficiency = 0.89

REWARD = 1.0*0.25 + 1.0*0.15 + 0.60*0.30 + 1.0*0.15 + 0.89*0.15
       = 0.25 + 0.15 + 0.18 + 0.15 + 0.134
       = 0.864

Tier: "good" (≥ 0.70 but < 0.90)
```

**Pattern sẽ được lưu:** ✅ (reward 0.864 > threshold 0.70)  
**Note:** Slower than Example 1 (20s/page vs 9s/page) → Lower efficiency score

### Example 3: Single Page - Poor Quality (Very Slow)

**Output:**
```json
{
  "success": true,
  "data": [
    {"headline": "N/A", "author": null, "content": ""},
    {"headline": "Some title", "author": "Unknown", "content": "Short"}
  ],
  "navigation_result": {
    "pages_collected": 1,
    "strategy_used": "unknown"
  },
  "execution_time_ms": 45000  // 45 seconds for 1 page!
}
```

**Reward Calculation:**
```
success = 1.0
has_data = 1.0
validation:
  - field_presence: 1.0
  - completeness: 0.33  (many null/empty)
  - quality: 0.25       ("N/A", "Unknown" are generic)
  → validation = 0.53

quantity = 2/5 = 0.4

efficiency:
  - no_errors = 1.0
  - pages_collected = 1 (single page)
  - time_per_page = 45000ms (45s - very slow!)
  - time_per_page > 30000 → time_penalty = (45000-30000)/60000 = 0.25
  - efficiency = 1.0 * (1.0 - 0.25) = 0.75
  → efficiency = 0.75

REWARD = 1.0*0.25 + 1.0*0.15 + 0.53*0.30 + 0.4*0.15 + 0.75*0.15
       = 0.25 + 0.15 + 0.159 + 0.06 + 0.113
       = 0.732
```

**Nhưng:** article_extraction có threshold = 0.80  
→ `0.732 < 0.80` → **Không được lưu vào successful_patterns**  
→ Được lưu vào `acceptable patterns` (giữa 0.40-0.80)

**Why failed?** Poor validation + very slow (45s for 1 page)

---

### 📊 Pagination Efficiency Comparison

| Crawl | Pages | Total Time | Time/Page | Efficiency Score | Final Reward |
|-------|-------|-----------|-----------|-----------------|--------------|
| **Fast Multi-page** | 2 | 18s | 9s | 1.0 (bonus!) | 0.994 ✨ |
| **Slow Multi-page** | 2 | 40s | 20s | 0.89 (penalty) | 0.864 |
| **Fast Single** | 1 | 12s | 12s | 1.0 | 0.920 |
| **Slow Single** | 1 | 45s | 45s | 0.75 (penalty) | 0.732 |

**Real Example - picare.vn:**
```json
{
  "domain": "picare.vn",
  "execution_time_ms": 33847,  // 33.8s total
  "pages_collected": 2,
  "items_extracted": 40,
  "pagination_strategy": "url_navigation"
}
```

**Calculation:**
```
time_per_page = 33847 / 2 = 16923ms (16.9s per page)
→ time_per_page > 15000 → penalty = (16923-15000)/45000 = 0.043
→ efficiency = 1.0 * (1.0 - 0.043) = 0.957

If validation = 0.75:
reward = 0.25 + 0.15 + 0.75*0.30 + 1.0*0.15 + 0.957*0.15
       = 0.25 + 0.15 + 0.225 + 0.15 + 0.144
       = 0.919 → "excellent" tier! ✨
```

**Key Insights:**
- ✅ Multi-page crawls judged by efficiency (time/page)
- ✅ Fast pagination (< 10s/page) gets bonus!
- ✅ Slow pagination (> 15s/page) gets penalty
- ✅ Single pages have more lenient threshold (30s)
- ⚠️ Very slow single pages still penalized

---

## 5. Learning Flow với Dynamic Thresholds

```python
# Training cycle
for rollout in rollout_data:
    # 1. Classify pattern type
    pattern_type = classify_pattern_type(extracted_data)
    # → "product_list"
    
    # 2. Get thresholds for this type
    threshold = get_success_threshold_for_type(pattern_type)
    # → {"success": 0.70, "failure": 0.40, "excellent": 0.90}
    
    # 3. Calculate reward
    reward = calculate_reward(result)
    # → 0.85
    
    # 4. Classify quality tier
    tier = classify_quality_tier(reward, threshold)
    # → "good" (0.70 ≤ 0.85 < 0.90)
    
    # 5. Decide fate
    if reward > threshold["success"]:
        successful_patterns.append({
            "type": pattern_type,
            "success_rate": reward,
            "metadata": {"quality_tier": tier}
        })
        # → Lưu vào knowledge store
    elif reward < threshold["failure"]:
        failure_patterns.append({...})
        # → Học từ failures
```

---

## 6. Benefits

### Before (binary reward):
```
Amazon crawl: reward = 1.0 (success)
  → Lưu vào successful_patterns
  
eBay crawl: reward = 1.0 (success)  
  → Lưu vào successful_patterns

❌ Không biết cái nào quality tốt hơn!
```

### After (multi-dimensional):
```
Amazon crawl:
  - Pattern: product_with_reviews
  - Reward: 0.92 (validation: 0.95, quantity: 0.8)
  - Tier: excellent
  → Lưu vào best_practices ✨
  
eBay crawl:
  - Pattern: product_list  
  - Reward: 0.73 (validation: 0.65, quantity: 0.7)
  - Tier: good
  → Lưu vào successful_patterns ✓

✅ Biết rõ Amazon pattern tốt hơn eBay!
✅ Agent ưu tiên học từ Amazon!
```

---

## 7. Impact on Knowledge Store

```python
# Retrieve best practices
best = await knowledge_store.get_best_practices(
    pattern_type="product_list"
)

# Trả về patterns sorted by:
# score = success_rate * log(frequency + 1)

# → Amazon pattern (0.92 * log(10)) = 2.12
# → eBay pattern (0.73 * log(5)) = 1.17

# Agent sẽ học từ Amazon pattern trước!
```

---

## 8. Adaptive Thresholds (Future)

Thresholds có thể adapt theo thời gian:

```python
# Sau 100 crawls của product_list
pattern_stats = get_pattern_statistics("product_list")
# → avg_reward: 0.82, std: 0.15

# Raise threshold nếu agent đã giỏi
if avg_reward > 0.80:
    new_threshold = avg_reward - 0.1  # 0.72
    # Maintain high standards
```

---

## ✅ Summary

| Aspect | Old System | New System |
|--------|-----------|------------|
| **Dimensions** | 1 (binary) | 5 (multi-dimensional) |
| **Threshold** | Fixed (0.7) | Dynamic per type |
| **Quality Tiers** | None | 4 tiers (excellent/good/acceptable/poor) |
| **Validation** | Simple | Field presence + completeness + quality |
| **Pattern Differentiation** | No | Yes, by type |
| **Learning Priority** | Random | Best practices first |
| **Pagination Awareness** | ❌ Total time only | ✅ **Time per page** |
| **Efficiency Bonus** | ❌ No | ✅ **10% bonus for fast pagination** |

**New Features:**
- ✅ **Pagination-aware scoring**: Time per page instead of total time
- ✅ **Dynamic penalties**: Different thresholds for single vs multi-page
- ✅ **Efficiency bonus**: Reward fast pagination (< 10s/page)
- ✅ **Fair comparison**: 1-page crawl vs 10-page crawl judged fairly

**Result:**  
Agent learns from high-quality **AND** efficient patterns → Faster improvement → Better crawls!
