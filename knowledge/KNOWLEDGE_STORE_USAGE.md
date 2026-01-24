# Knowledge Store Usage Guide

## 🎯 Tại sao cần Intelligent Pattern Classification?

Trước đây, tất cả patterns đều được lưu với type `"successful_crawl"` → không phân biệt được:
- Product extraction
- Article parsing  
- Review scraping
- Contact info extraction

**Bây giờ**: Patterns được classify thông minh thành các loại cụ thể!

---

## 📊 Pattern Types được hỗ trợ

| Type | Khi nào được classify | Ví dụ fields |
|------|----------------------|--------------|
| `product_list` | Có price + product name | `["product_name", "price"]` |
| `product_with_reviews` | Có product + price + rating | `["title", "price", "rating"]` |
| `product_catalog` | Có product name, không có price | `["product_name", "brand"]` |
| `price_extraction` | Chỉ focus vào giá | `["price", "discount", "currency"]` |
| `review_extraction` | Chỉ focus vào reviews | `["rating", "comment", "author"]` |
| `article_extraction` | Có headline/content + author/date | `["headline", "author", "published_date"]` |
| `content_extraction` | Có title/body | `["title", "content"]` |
| `contact_info` | Có email/phone/address | `["email", "phone", "address"]` |
| `navigation_pattern` | Có pagination/next_page | `["next_page", "load_more"]` |
| `tabular_data` | Nhiều fields (>5), numeric data | `["col1", "col2", ..., "col10"]` |
| `media_extraction` | Có image/photo URLs | `["image_url", "thumbnail"]` |
| `multi_field_extraction` | Nhiều fields nhưng không rõ type | 5+ fields |
| `generic_extraction` | Fallback | Bất kỳ |

---

## 🚀 Cách sử dụng

### 1. Retrieve patterns theo type

```python
# Tìm tất cả product extraction patterns đã học
product_patterns = await knowledge_store.get_patterns_by_type(
    pattern_type="product_list",
    top_k=10
)

for pattern in product_patterns:
    print(f"Domain: {pattern['domain']}")
    print(f"Success rate: {pattern['success_rate']}")
    print(f"Fields: {pattern['extraction_fields']}")
```

### 2. Lấy best practices

```python
# Best practices cho amazon.com
best = await knowledge_store.get_best_practices(
    domain="amazon.com",
    pattern_type="product_with_reviews"
)

# Best practices cho tất cả article extraction
article_best = await knowledge_store.get_best_practices(
    pattern_type="article_extraction"
)
```

### 3. Semantic search với enriched query

```python
# Query với context đầy đủ
patterns = await knowledge_store.retrieve_patterns(
    query={
        "domain": "shopee.vn",
        "intent": "extract product information",
        "description": "Get product name, price, and ratings from listing page",
        "extraction_fields": ["product_name", "price", "rating"],
        "user_description": "Crawl product prices with reviews",
        "include_related": True  # Graph enrichment
    },
    top_k=5
)

# Kết quả:
# - Top matches từ Qdrant (semantic similarity)
# - Related patterns từ Neo4j (cùng domain, similar patterns)
```

### 4. Pattern statistics

```python
stats = await knowledge_store.get_pattern_statistics()

print(f"Total patterns: {stats['total_patterns']}")
print(f"\nBy type:")
for pattern_type, count in stats['by_type'].items():
    print(f"  {pattern_type}: {count}")

print(f"\nBy domain:")
for domain, count in stats['by_domain'].items():
    print(f"  {domain}: {count}")

print(f"\nHigh success patterns (>80%): {stats['high_success_patterns']}")
print(f"Frequently used (>5 times): {stats['frequently_used']}")

# Pagination statistics
pagination_stats = stats['pagination_stats']
print(f"\n🔄 Pagination:")
print(f"  Patterns with pagination: {pagination_stats['patterns_with_pagination']}")
print(f"  Average pages crawled: {pagination_stats['avg_pages_crawled']:.1f}")
print(f"  Strategies used:")
for strategy, count in pagination_stats['strategies_used'].items():
    print(f"    {strategy}: {count}")
```

### 5. Retrieve pagination patterns

```python
# Get successful pagination patterns for a domain
pagination_patterns = await knowledge_store.get_pagination_patterns(
    domain="shopee.vn",
    pattern_type="product_list",
    top_k=5
)

for pattern in pagination_patterns:
    pagination_info = pattern['pagination_info']
    print(f"Domain: {pattern['domain']}")
    print(f"  Strategy: {pagination_info['pagination_strategy']}")
    print(f"  Pages crawled: {pagination_info['pages_crawled']}")
    print(f"  Max requested: {pagination_info['max_pages_requested']}")
    print(f"  Success rate: {pattern['success_rate']:.2f}")
```

---

## 🔍 Flow hoàn chỉnh

### Training Phase:

```python
# 1. User crawl amazon.com for products
task = {
    "url": "https://amazon.com/products",
    "user_description": "Extract product names and prices",
    "extraction_schema": {"required": ["product_name", "price"]}
}

# 2. Agent executes → result có data
result = {
    "success": True,
    "data": [
        {"product_name": "iPhone 15", "price": "$999"},
        {"product_name": "MacBook Pro", "price": "$2499"}
    ]
}

# 3. Algorithm learns → classify thông minh
rollout_data = [{
    "reward": 0.95,
    "task": task,
    "result": result
}]

learned = await algorithm.learn_from_interactive_rollouts(rollout_data)

# Pattern được lưu với:
# - type: "product_list" (intelligent!)
# - extraction_fields: ["product_name", "price"]
# - domain: "amazon.com"
# - success_rate: 0.95
```

### Retrieval Phase:

```python
# User mới muốn crawl ebay.com (domain khác nhưng intent giống)
new_task = {
    "url": "https://ebay.com/items",
    "user_description": "Get product titles and prices"
}

# Semantic search tìm patterns tương tự
similar_patterns = await knowledge_store.retrieve_patterns({
    "domain": "ebay.com",
    "intent": "product extraction",
    "description": new_task["user_description"],
    "extraction_fields": ["product_title", "price"]  # Infer from description
})

# Kết quả:
# 1. Pattern từ amazon.com (type: product_list, score: 0.92)
#    → Vì semantic giống: "product + price"
# 2. Pattern từ shopee.vn (type: product_with_reviews, score: 0.85)
#    → Graph enrichment: cùng category
```

---

## 💡 Lợi ích

### Before (hardcoded `"successful_crawl"`):
```json
{
  "type": "successful_crawl",
  "domain": "amazon.com",
  "description": "Successful crawl for https://amazon.com"
}
```
❌ Không biết pattern này extract cái gì  
❌ Semantic search không hiệu quả  
❌ Không group được theo loại  

### After (intelligent classification):
```json
{
  "type": "product_with_reviews",
  "domain": "amazon.com",
  "extraction_fields": ["product_name", "price", "rating", "review_count"],
  "description": "product_with_reviews pattern for amazon.com",
  "metadata": {
    "user_prompt": "Get products with ratings",
    "items_extracted": 50,
    "pagination": {
      "used_pagination": true,
      "pages_crawled": 5,
      "pagination_strategy": "click_next_button",
      "max_pages_requested": 10,
      "pagination_successful": true
    }
  }
}
```
✅ Biết rõ pattern này làm gì  
✅ Semantic search chính xác  
✅ Group được theo type  
✅ Best practices per type  
✅ **Biết pattern này dùng pagination như thế nào**  

---

## 🎯 Use cases thực tế

### 1. Type-specific strategy selection

```python
# Agent quyết định strategy dựa trên pattern type
user_wants = "extract product prices"

# Tìm best product_list patterns
strategies = await knowledge_store.get_best_practices(
    pattern_type="product_list"
)

# Apply strategy từ pattern có success_rate cao nhất
best_strategy = strategies[0]
agent.apply_strategy(best_strategy)
```

### 2. Domain adaptation

```python
# User chưa từng crawl target.com nhưng đã có walmart.com
target_patterns = await knowledge_store.retrieve_patterns({
    "domain": "target.com",
    "intent": "product extraction",
    "extraction_fields": ["name", "price"]
})

# Graph enrichment sẽ trả về:
# - Walmart patterns (cùng industry: retail)
# - Amazon patterns (cùng type: product_list)
# → Agent có baseline strategy ngay lập tức!
```

### 3. Failure analysis

```python
# Tìm xem pattern type nào hay fail
stats = await knowledge_store.get_pattern_statistics()

failure_patterns = await knowledge_store.get_patterns_by_type(
    pattern_type="failure_pattern",
    top_k=20
)

# Phân tích: "article_extraction hay fail vì selector thay đổi"
# → Cải thiện strategy cho type đó
```

### 4. Learn pagination strategies

```python
# User muốn crawl e-commerce site với nhiều pages
# Tìm patterns đã thành công với pagination

pagination_patterns = await knowledge_store.get_pagination_patterns(
    pattern_type="product_list",
    top_k=5
)

# Analyze successful strategies
for pattern in pagination_patterns:
    pagination = pattern['pagination_info']
    print(f"Domain: {pattern['domain']}")
    print(f"  Strategy: {pagination['pagination_strategy']}")
    print(f"  Success: {pattern['success_rate']:.2f}")
    print(f"  Pages: {pagination['pages_crawled']}")

# Apply best strategy
best_pagination = pagination_patterns[0]['pagination_info']
agent.set_pagination_strategy(best_pagination['pagination_strategy'])
```

---

## 🔧 Tích hợp vào Agent

```python
class SharedCrawlerAgent(_BaseAgent):
    def __init__(self, gemini_client, mode: str = "production", 
                 knowledge_store: Optional[HybridKnowledgeStore] = None):
        self.knowledge_store = knowledge_store
        # ...
    
    async def _training_rollout(self, task, resources, rollout):
        # Bước 1: Lấy base resources (versioned)
        learned_patterns = resources.get("domain_patterns", {})
        
        # Bước 2: Enrich với real-time semantic search
        if self.knowledge_store:
            # Infer extraction fields from task
            extraction_fields = []
            if task.get("extraction_schema"):
                extraction_fields = task["extraction_schema"].get("required", [])
            
            query = {
                "domain": self._extract_domain(task["url"]),
                "intent": task.get("user_description", ""),
                "description": task.get("user_description", ""),
                "extraction_fields": extraction_fields,
                "include_related": True
            }
            
            # Semantic search + graph enrichment
            similar_patterns = await self.knowledge_store.retrieve_patterns(
                query, top_k=3
            )
            
            # Apply best matching pattern
            if similar_patterns and similar_patterns[0].get("score", 0) > 0.85:
                best_pattern = similar_patterns[0]
                logger.info(f"🎯 Using learned pattern: {best_pattern['type']} "
                          f"(score: {best_pattern['score']:.2f})")
                # Merge strategy...
```

---

## 📈 Monitoring

```python
# Xem knowledge store đang học gì
async def monitor_knowledge():
    stats = await knowledge_store.get_pattern_statistics()
    
    print(f"📊 KNOWLEDGE STORE STATUS")
    print(f"=" * 60)
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"High-success patterns: {stats['high_success_patterns']}")
    print(f"Frequently used: {stats['frequently_used']}")
    print(f"\nPattern types:")
    for ptype, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {ptype:30s} {count:3d}")
    print(f"\nTop domains:")
    for domain, count in sorted(stats['by_domain'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {domain:30s} {count:3d}")
```

Output:
```
📊 KNOWLEDGE STORE STATUS
============================================================
Total patterns: 245
High-success patterns: 178
Frequently used: 45

Pattern types:
  product_list                   89
  article_extraction             45
  product_with_reviews           32
  review_extraction              28
  price_extraction               18
  contact_info                   12
  ...

Top domains:
  amazon.com                     45
  shopee.vn                      38
  ebay.com                       27
  ...

🔄 Pagination:
  Patterns with pagination: 67
  Average pages crawled: 4.2
  Strategies used:
    click_next_button: 45
    url_navigation: 18
    infinite_scroll: 4
```

---

## ✅ Checklist Implementation

- [x] Intelligent pattern classification trong `learn_from_interactive_rollouts()`
- [x] Lưu `extraction_fields` vào pattern
- [x] Enriched embedding (type + domain + fields + user_prompt)
- [x] `get_patterns_by_type()` - filter theo type
- [x] `get_best_practices()` - high success + high frequency
- [x] `get_pattern_statistics()` - overview
- [ ] Tích hợp vào `SharedCrawlerAgent` (TODO)
- [ ] Real-time semantic search trong agent runtime (TODO)

---

## 🎓 Kết luận

**Trước**: Knowledge store = "black box" chứa "successful_crawl"  
**Sau**: Knowledge store = "smart library" với classified patterns, semantic search, và best practices

Bây giờ agent có thể:
1. Học từ patterns cụ thể (product, article, review, etc.)
2. Tìm kiếm semantic chính xác
3. Áp dụng best practices theo type
4. Adapt nhanh cho domains mới
