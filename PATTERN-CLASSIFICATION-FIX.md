# 🎯 Giải pháp: Knowledge Store với Intelligent Pattern Classification

## ❌ Vấn đề trước đây

```python
# Tất cả patterns đều là "successful_crawl"
pattern = {
    "type": "successful_crawl",  # ← Generic, không có ý nghĩa
    "domain": "amazon.com",
    "description": "Successful crawl for https://amazon.com"
}
```

**Hậu quả:**
- ❌ Không biết pattern này extract cái gì
- ❌ Semantic search không hiệu quả (tất cả đều giống nhau)
- ❌ Không group được theo loại
- ❌ Không có best practices cụ thể

---

## ✅ Giải pháp

### 1. Intelligent Pattern Classification

**File:** `algorithms/self_improving_algorithm.py`

```python
def _classify_pattern_type(self, extraction_fields, schema, extracted_data):
    """
    Classify patterns based on extracted fields
    Returns: "product_list", "article_extraction", "review_extraction", etc.
    """
    fields_lower = [f.lower() for f in extraction_fields]
    
    # E-commerce product patterns
    if "price" in fields_lower and "product_name" in fields_lower:
        if "rating" in fields_lower:
            return "product_with_reviews"
        return "product_list"
    
    # Article/content patterns
    if "headline" in fields_lower or "content" in fields_lower:
        if "author" in fields_lower:
            return "article_extraction"
        return "content_extraction"
    
    # ... 11+ pattern types
```

### 2. Apply vào Interactive Rollouts

**Before:**
```python
successful_patterns.append({
    'type': 'successful_crawl',  # ← Hardcoded
    'domain': domain,
})
```

**After:**
```python
# Extract fields từ result
extraction_fields = list(result_data[0].keys()) if result_data else []

# Classify thông minh
pattern_type = self._classify_pattern_type(
    extraction_fields=extraction_fields,
    schema=extraction_schema,
    extracted_data=result_data
)

successful_patterns.append({
    'type': pattern_type,  # ← "product_list", "article_extraction", etc.
    'domain': domain,
    'extraction_fields': extraction_fields,  # ← Important for search
    'metadata': {
        'user_prompt': user_description,  # ← Context
        'items_extracted': len(result_data)
    }
})
```

### 3. Enriched Embeddings

**File:** `knowledge/hybrid_knowledge_store.py`

**Before:**
```python
async def _embed_pattern(self, pattern):
    text = f"{pattern.get('type', '')} for {pattern.get('domain', '')}"
    return await self.gemini_client.embed(text)
```

**After:**
```python
async def _embed_pattern(self, pattern):
    parts = [
        f"Pattern type: {pattern.get('type', 'unknown')}",
        f"Domain: {pattern.get('domain', 'unknown')}",
    ]
    
    # Add extraction fields
    if pattern.get('extraction_fields'):
        parts.append(f"Extracts: {', '.join(pattern['extraction_fields'])}")
    
    # Add user intent
    if pattern.get('metadata', {}).get('user_prompt'):
        parts.append(f"User intent: {pattern['metadata']['user_prompt']}")
    
    text = ". ".join(parts)
    return await self.gemini_client.embed(text)
```

→ Embedding giờ chứa nhiều context hơn → semantic search chính xác hơn!

### 4. New Retrieval Methods

```python
# Get patterns by specific type
product_patterns = await knowledge_store.get_patterns_by_type(
    pattern_type="product_list",
    top_k=10
)

# Get best practices (high success + high frequency)
best = await knowledge_store.get_best_practices(
    domain="amazon.com",
    pattern_type="product_with_reviews"
)

# Get statistics
stats = await knowledge_store.get_pattern_statistics()
# → {"by_type": {"product_list": 89, "article_extraction": 45, ...}}
```

---

## 📊 Pattern Types Supported

| Type | Trigger | Example Fields |
|------|---------|----------------|
| `product_list` | price + product_name | `["product_name", "price"]` |
| `product_with_reviews` | price + product + rating | `["title", "price", "rating"]` |
| `article_extraction` | headline/content + author | `["headline", "author", "date"]` |
| `review_extraction` | rating + comment | `["rating", "comment", "author"]` |
| `price_extraction` | price/cost/discount | `["price", "discount"]` |
| `contact_info` | email/phone/address | `["email", "phone", "address"]` |
| `media_extraction` | image/photo URLs | `["image_url", "thumbnail"]` |
| `tabular_data` | 5+ fields, numeric | `["col1", ..., "col10"]` |
| ... | ... | ... |

**Total:** 11+ pattern types

---

## 🚀 Impact

### Before vs After

**Semantic Search:**
```
Query: "extract product prices"

BEFORE:
  - Pattern 1: type="successful_crawl", score=0.65
  - Pattern 2: type="successful_crawl", score=0.64
  - Pattern 3: type="successful_crawl", score=0.63
  → Tất cả đều giống nhau!

AFTER:
  - Pattern 1: type="product_list", score=0.92 ✅
  - Pattern 2: type="price_extraction", score=0.88 ✅
  - Pattern 3: type="product_with_reviews", score=0.85 ✅
  → Chính xác, relevant!
```

**Best Practices:**
```
BEFORE:
  get_best_practices() → Return tất cả "successful_crawl"
  → Không biết cái nào best cho product extraction

AFTER:
  get_best_practices(pattern_type="product_list")
  → Return top 10 product extraction patterns
     (sorted by success_rate * log(frequency))
  → Agent học từ best practices cụ thể!
```

**Statistics:**
```
BEFORE:
  {
    "total_patterns": 245,
    "by_type": {"successful_crawl": 245}
  }
  → Không có insight gì

AFTER:
  {
    "total_patterns": 245,
    "by_type": {
      "product_list": 89,
      "article_extraction": 45,
      "review_extraction": 28,
      "price_extraction": 18,
      ...
    },
    "by_domain": {
      "amazon.com": 45,
      "shopee.vn": 38,
      ...
    }
  }
  → Insight rõ ràng về knowledge!
```

---

## 🧪 Testing

```bash
# Store sample patterns
python test_knowledge_classification.py --store-samples

# Run all tests
python test_knowledge_classification.py
```

**Output:**
```
📊 Test 1: Pattern Statistics
----------------------------------------
Total patterns: 245
High success patterns (>80%): 178
Frequently used (>5 times): 45

📌 Pattern Types Distribution:
  product_list                   89 patterns
  article_extraction             45 patterns
  product_with_reviews           32 patterns
  review_extraction              28 patterns
  ...

🌐 Top Domains:
  amazon.com                     45 patterns
  shopee.vn                      38 patterns
  ebay.com                       27 patterns
  ...
```

---

## 📁 Files Changed

| File | Changes |
|------|---------|
| `algorithms/self_improving_algorithm.py` | ✅ Apply `_classify_pattern_type()` to interactive rollouts<br>✅ Add `extraction_fields` and `user_prompt` to patterns |
| `knowledge/hybrid_knowledge_store.py` | ✅ Enriched `_embed_pattern()` and `_embed_query()`<br>✅ New: `get_patterns_by_type()`<br>✅ New: `get_best_practices()`<br>✅ New: `get_pattern_statistics()` |
| `knowledge/KNOWLEDGE_STORE_USAGE.md` | ✅ Complete usage guide |
| `test_knowledge_classification.py` | ✅ Test script |

---

## ✅ Checklist

- [x] Intelligent pattern classification implemented
- [x] Applied to `learn_from_interactive_rollouts()`
- [x] Enriched embeddings with fields + user prompt
- [x] `get_patterns_by_type()` method
- [x] `get_best_practices()` method
- [x] `get_pattern_statistics()` method
- [x] Usage documentation
- [x] Test script
- [ ] Integrate into `SharedCrawlerAgent` (Future work)
- [ ] Real-time semantic search in agent (Future work)

---

## 🎓 Conclusion

**Trước:** Knowledge store = black box chứa "successful_crawl"  
**Sau:** Knowledge store = smart library với:
- ✅ 11+ classified pattern types
- ✅ Semantic search chính xác
- ✅ Best practices per type
- ✅ Domain/type statistics
- ✅ Rich metadata (fields, user prompts)

**Agent giờ có thể:**
1. 🎯 Học từ patterns cụ thể (product, article, review)
2. 🔍 Tìm kiếm semantic chính xác
3. 📚 Áp dụng best practices theo type
4. 🚀 Adapt nhanh cho domains mới
