"""
External Prompt Configuration
==============================

All LLM prompts externalized from code for easy editing.
This replaces ~500 LOC of embedded prompts in the old modules.

Prompts are stored as Python dict for:
- Version control (git trackable)
- Easy editing without code changes
- No external dependencies (YAML/JSON parsers)
"""

PROMPTS = {
    # =============================================================================
    # UNIFIED ANALYSIS + CODE GENERATION PROMPT
    # =============================================================================
    
    "unified_analysis_and_code": """You are a smart data analysis assistant that generates Python code to answer user questions about product data.

**User Query:**
{query}

**Available Data:**
- Total products: {total_products}
- Data schema: {schema}
- Sample products:
{sample_products}

⚠️ **CRITICAL - Multiple Schemas**:
Different data sources may have DIFFERENT field structures:
- **Crawl products** (from websites): May have `price`, `product_name`, `brand`
- **CSV products** (from uploaded files): May have `current_price_vnd`, `original_price_vnd`, `product_name`

When filtering by source (_source == 'csv_file'), inspect sample CSV products to detect their schema.
When working with all products, create flexible helpers that handle BOTH schemas.

**⚠️ SOURCE METADATA (for filtering by file/URL):**
Each product has metadata fields for source tracking:
- `_source`: "crawl_job" or "csv_file" (data source type)
- `_source_url`: URL for crawl jobs, filename for CSV files (e.g., "haircare_crawl_non_picare_fixed.csv")
- `_crawl_job_id`: Unique identifier for the data source
- `_job_prompt`: Original user prompt or "Uploaded file: X.csv"

**Available Data Sources:**
{sources_info}

**CRITICAL FILTERING RULES (Use Normalized/Fuzzy Matching!):**
⚠️ Filenames may have underscores/hyphens removed by user. ALWAYS use normalized partial matching!

1. "sản phẩm trong file X.csv" → Filter with NORMALIZED matching:
   ```python
   # Normalize both query and _source_url (remove _, -, spaces, lowercase)
   query_norm = 'X'.replace('_', '').replace('-', '').replace(' ', '').lower()
   csv_products = [p for p in products 
                   if query_norm in p.get('_source_url', '').replace('_', '').replace('-', '').replace(' ', '').lower()]
   ```

2. "sản phẩm từ link/URL Y" → Filter by: `if 'Y' in p.get('_source_url', '')` 

3. "sản phẩm từ crawl" → Filter by: `if p.get('_source') == 'crawl_job'`

4. "sản phẩm từ file/CSV" → Filter by: `if p.get('_source') == 'csv_file'`

5. No mention of source → Use ALL products (no filter)

⚠️ IMPORTANT: User may type "haircarefile.csv" but actual filename is "hair_care_file.csv" - ALWAYS normalize!

**Your Task:**
1. Analyze the query to understand the intent
2. Generate Python code to compute the answer

**Response Format (MUST follow exactly):**

First, provide analysis in JSON:
```json
{{
  "intent_type": "computational|listing|comparison|visualization|descriptive",
  "operation": "sum|count|average|min|max|filter|sort|group|find",
  "output_format": "number|list|chart|table|text",
  "target_fields": ["field1", "field2"],
  "filters": {{}},
  "reasoning": "Brief explanation of your understanding",
  "confidence": 0.95,
  "requires_visualization": false,
  "chart_type": null
}}
```

Then, provide executable Python code:
```python
# STEP 1: Analyze the schema to detect available fields
# Schema shows: {schema}
# Look for price fields, name fields, etc. in the schema above

# STEP 2: Create helper functions based on detected fields
# Example: If schema has 'current_price_vnd' and 'original_price_vnd':
def get_current_price(p):
    # Check which fields exist in schema and use them
    price_val = p.get('current_price_vnd') or p.get('price') or 0
    return float(str(price_val).replace('₫', '').replace(',', '')) if price_val else 0

def get_name(p):
    # Check which name fields exist in schema
    return p.get('product_name') or p.get('name') or p.get('title') or 'Unknown'

# STEP 3: Write code to answer the query
result = ...
```

**Guidelines:**
1. **Intent Types:**
   - `computational`: Requires calculation (sum, count, average, min, max)
   - `listing`: List/enumerate items (top 10, all brands, etc.)
   - `comparison`: Compare items (cheaper than X, better than Y)
   - `visualization`: Needs chart/graph - Keywords: "vẽ biểu đồ", "tạo biểu đồ", "thống kê", "phân bổ", "distribution", "chart", "graph", "visualization", "theo nhóm", "theo loại", "theo thương hiệu"
   - `descriptive`: General information

2. **Operations:**
   - `sum`: Add numbers (total price, total quantity)
   - `count`: Count items (how many, number of)
   - `average`: Calculate mean (average price, mean rating)
   - `min`: Find minimum (cheapest, lowest)
   - `max`: Find maximum (most expensive, highest)
   - `filter`: Select subset (products with X, items where Y)
   - `sort`: Order items (sort by price, rank by rating)
   - `group`: Group by field (group by brand, category)
   - `find`: Search/lookup (find product X, get details of Y)

3. **Output Formats:**
   - `number`: Single numerical value (1234, 56.78)
   - `list`: Array of items (list of names, list of prices)
   - `chart`: Data for visualization (bar chart, pie chart)
   - `table`: Structured tabular data
   - `text`: Free-form text response

4. **Code Requirements:**
   - Must be executable Python (no pseudo-code)
   - **NEVER use import statements - Counter and defaultdict are already available globally**
   - DO NOT write `from collections import Counter` - just use `Counter()` directly
   - Use only allowed built-in functions and Counter/defaultdict
   - **CRITICAL: Auto-detect field names from the provided schema**
     * Look at the "Data schema" section to see which fields exist
     * Price fields might be: `price`, `current_price_vnd`, `original_price_vnd`, `salePrice`, etc.
     * Name fields might be: `name`, `product_name`, `productName`, `title`, etc.
     * **Create helper function that checks schema and tries all variations**
     * Example pattern:
       ```python
       def get_price(p):
           # Check schema to see which price field exists
           return (p.get('current_price_vnd') or p.get('price') or 
                   p.get('original_price_vnd') or p.get('salePrice') or 0)
       ```
   - **CRITICAL: Always convert prices to float before math operations**
     * Prices may be strings like "117,000₫" or numbers like 579500
     * Always use: `float(str(value).replace('₫', '').replace(',', ''))`
   - **CRITICAL: Handle None values before calling string methods**
     * When extracting string fields (brand, name, etc.), always check for None first
     * WRONG: `p.get('brand', '').lower()` ❌ (fails if brand is None, not missing)
     * RIGHT: `(p.get('brand') or '').lower()` ✅ (converts None to empty string first)
     * Example pattern:
       ```python
       def get_brand(p):
           brand = p.get('brand')
           return brand.lower() if brand else ''
       ```
   - **For charts** (when `requires_visualization: true`):
     * Code MUST return: `result = {{"labels": [...], "values": [...]}}`
     * Do NOT return Counter objects or text summaries directly
   - Handle missing/null values gracefully
   - Assign final result to variable `result`
   - **ABSOLUTELY NO IMPORTS** - Counter and defaultdict are already globally available
   - Just use `Counter(...)` directly without any import statement

5. **Field Names:**
   - Auto-detect from schema
   - Use .get() for safe access: `p.get('price', 0)`
   - Handle different naming conventions: price/Price/PRICE

**CRITICAL - Visualization vs Listing:**
- If query has "thống kê", "phân bổ", "distribution", "theo nhóm", "theo loại" → use `visualization` (chart)
- If query just asks for "danh sách", "list", "show me", "get all" → use `listing` (text list)
- Statistics/grouping queries ALWAYS need charts, not lists!

**Examples (showing patterns, NOT hardcoded solutions):**

Query: "Có bao nhiêu sản phẩm?"
```json
{{"intent_type": "computational", "operation": "count", "output_format": "number", "target_fields": [], "reasoning": "Simple count query", "confidence": 0.99}}
```
```python
result = len(products)
```

Query: "Tổng giá trị tất cả sản phẩm?" (Schema-aware price detection)
```json
{{"intent_type": "computational", "operation": "sum", "output_format": "number", "target_fields": ["price"], "reasoning": "Sum all prices - detect price field from schema", "confidence": 0.95}}
```
```python
# Look at schema to determine which price field to use
# If schema has 'current_price_vnd', use that; if 'price', use that; etc.
def get_price(p):
    # Adapt based on actual schema
    price_val = p.get('current_price_vnd') or p.get('price') or p.get('salePrice') or 0
    return float(str(price_val).replace('₫', '').replace(',', '')) if price_val else 0

result = sum(get_price(p) for p in products if get_price(p) > 0)
```

Query: "Sản phẩm đắt nhất?" (Flexible field detection)
```json
{{"intent_type": "listing", "operation": "max", "output_format": "text", "target_fields": ["price", "name"], "reasoning": "Find product with highest price", "confidence": 0.90}}
```
```python
# Auto-detect price and name fields from schema
def get_price(p):
    price_val = p.get('current_price_vnd') or p.get('price') or p.get('original_price_vnd') or 0
    return float(str(price_val).replace('₫', '').replace(',', '')) if price_val else 0

def get_name(p):
    return p.get('product_name') or p.get('name') or p.get('title') or 'Unknown'

max_product = max([p for p in products if get_price(p) > 0], key=get_price)
result = {{
    "name": get_name(max_product),
    "price": get_price(max_product),
    "type": "max_product"
}}
```

Query: "Thống kê phân bổ theo thương hiệu" (Chart - group by brand)
```json
{{"intent_type": "visualization", "operation": "group", "output_format": "chart", "target_fields": ["brand"], "reasoning": "Group by brand for pie chart", "confidence": 0.95, "requires_visualization": true, "chart_type": "pie"}}
```
```python
# NOTE: Counter is already available - DO NOT import it!
# CRITICAL: Handle None values properly
def get_brand(p):
    brand = p.get('brand')
    return brand if brand else 'Unknown'

brands = [get_brand(p) for p in products]
brand_counts = Counter(brands)  # Use Counter directly without import
result = {{
    "labels": list(brand_counts.keys()),
    "values": list(brand_counts.values())
}}
```

Query: "Giá trung bình sản phẩm thương hiệu X" (Brand filtering with None handling)
```json
{{"intent_type": "computational", "operation": "average", "output_format": "number", "target_fields": ["price", "brand"], "reasoning": "Average price for specific brand", "confidence": 0.92}}
```
```python
# CRITICAL: Handle None values before calling .lower()
def get_brand(p):
    brand = p.get('brand')
    return brand.lower() if brand else ''

def get_price(p):
    price_val = p.get('current_price_vnd') or p.get('price') or 0
    return float(str(price_val).replace('₫', '').replace(',', '')) if price_val else 0

# Filter by brand (normalize for case-insensitive match)
target_brand = 'easydew'  # Extract from query
brand_products = [p for p in products if get_brand(p) == target_brand and get_price(p) > 0]

if brand_products:
    total_price = sum(get_price(p) for p in brand_products)
    result = total_price / len(brand_products)
else:
    result = 0
```

Query: "Sản phẩm từ file X.csv" (Source filtering - extract filename from query)
```json
{{"intent_type": "computational", "operation": "count", "output_format": "number", "target_fields": [], "reasoning": "Count products from CSV file", "confidence": 0.95}}
```
```python
# Extract the filename mentioned in the query (user said "file X.csv")
# In this case, the query contains the filename we need to match
# Normalize for flexible matching (remove underscores, hyphens, spaces, lowercase)
# The actual filename will be in _source_url field

# Strategy: Extract key words from query, normalize, and match
query_lower = '{query}'.lower()
# Remove common words, extract the filename part
query_normalized = query_lower.replace('_', '').replace('-', '').replace(' ', '')

# Filter products where normalized _source_url contains normalized query
csv_products = [
    p for p in products 
    if p.get('_source') == 'csv_file' and 
    any(word in p.get('_source_url', '').replace('_', '').replace('-', '').replace(' ', '').lower() 
        for word in ['haircare', 'file', 'csv'])  # Extract keywords from query dynamically
]
result = len(csv_products)
```

Query: "Tỉ lệ giảm giá trung bình" (Discount calculation - schema-aware)
```json
{{"intent_type": "computational", "operation": "average", "output_format": "number", "target_fields": ["price", "original_price"], "reasoning": "Calculate average discount rate", "confidence": 0.90}}
```
```python
# CRITICAL: First check which products HAVE discount information
# CSV products might have: original_price_vnd + current_price_vnd
# Crawl products might only have: price (no discount info)

# Helper to convert ANY price to float
def to_float_price(val):
    if not val:
        return 0
    return float(str(val).replace('₫', '').replace(',', ''))

# Calculate discount rate - check ALL possible field combinations
def get_discount_rate(p):
    # Try multiple field combinations for original vs current price
    original = to_float_price(
        p.get('original_price_vnd') or 
        p.get('originalPrice') or 
        p.get('price_before_discount')
    )
    current = to_float_price(
        p.get('current_price_vnd') or 
        p.get('price') or 
        p.get('salePrice')
    )
    
    # Only calculate if we have BOTH prices and original > current
    if original > 0 and current > 0 and original > current:
        return ((original - current) / original) * 100
    return 0

# Filter to only products that HAVE discount information
products_with_discount = [p for p in products if get_discount_rate(p) > 0]

if products_with_discount:
    result = sum(get_discount_rate(p) for p in products_with_discount) / len(products_with_discount)
else:
    result = 0  # No products have discount information
```

Query: "Tỉ lệ giảm giá từ file X.csv" (Discount + Source Filtering)
```json
{{"intent_type": "computational", "operation": "average", "output_format": "number", "target_fields": ["price"], "reasoning": "Average discount from CSV file only", "confidence": 0.92}}
```
```python
# Step 1: Filter by source first
query_norm = 'X.csv'.replace('_', '').replace('-', '').replace(' ', '').lower()
csv_products = [
    p for p in products 
    if p.get('_source') == 'csv_file' and
    query_norm in p.get('_source_url', '').replace('_', '').replace('-', '').replace(' ', '').lower()
]

# Step 2: CSV products likely have different schema than crawl products
# Check for original_price_vnd and current_price_vnd (common in CSV)
def to_float_price(val):
    return float(str(val).replace('₫', '').replace(',', '')) if val else 0

def get_discount_rate(p):
    # CSV schema: original_price_vnd, current_price_vnd
    original = to_float_price(p.get('original_price_vnd'))
    current = to_float_price(p.get('current_price_vnd'))
    
    if original > 0 and current > 0 and original > current:
        return ((original - current) / original) * 100
    return 0

# Step 3: Calculate average discount for CSV products
csv_with_discount = [p for p in csv_products if get_discount_rate(p) > 0]
if csv_with_discount:
    result = sum(get_discount_rate(p) for p in csv_with_discount) / len(csv_with_discount)
else:
    result = 0
```

Query: "So sánh giá từ file vs từ crawl" (Multi-source comparison)
```json
{{"intent_type": "comparison", "operation": "average", "output_format": "text", "target_fields": ["price"], "reasoning": "Compare prices between sources", "confidence": 0.92}}
```
```python
# Detect price field from schema
def get_price(p):
    price_val = p.get('current_price_vnd') or p.get('price') or p.get('salePrice') or 0
    return float(str(price_val).replace('₫', '').replace(',', '')) if price_val else 0

# Filter by source type
csv_products = [p for p in products if p.get('_source') == 'csv_file' and get_price(p) > 0]
crawl_products = [p for p in products if p.get('_source') == 'crawl_job' and get_price(p) > 0]

avg_csv = sum(get_price(p) for p in csv_products) / len(csv_products) if csv_products else 0
avg_crawl = sum(get_price(p) for p in crawl_products) / len(crawl_products) if crawl_products else 0

result = {{
    "csv_avg": avg_csv,
    "crawl_avg": avg_crawl,
    "csv_count": len(csv_products),
    "crawl_count": len(crawl_products)
}}
```

Now analyze and generate code for the user's query above.
""",

    # =============================================================================
    # FALLBACK PROMPTS (if needed for specific cases)
    # =============================================================================
    
    "simple_analysis": """Analyze this query briefly:

Query: {query}

Respond with JSON only:
{{
  "intent_type": "computational|listing|comparison|visualization|descriptive",
  "operation": "sum|count|average|filter|sort|group|find",
  "confidence": 0.0-1.0
}}
""",

    "error_recovery": """The previous code failed with an error. Analyze the error and generate a corrected version.

Original Query: {query}

Failed Code:
```python
{code}
```

Error Message:
{error}

**Common Error Patterns & Fixes:**

1. **Price Range Error** - `could not convert string to float: '349000 ~ 698000'`
   - Problem: Prices like "349000 ~ 698000" or "100,000₫ - 200,000₫"
   - Fix: Use robust price parsing helper:
   ```python
   def parse_price(val):
       if not val:
           return 0
       val_str = str(val).replace('₫', '').replace(',', '').strip()
       # Handle ranges
       if '~' in val_str:
           parts = val_str.split('~')
       elif ' - ' in val_str:
           parts = val_str.split(' - ')
       else:
           parts = [val_str]
       # Parse and average
       prices = []
       for part in parts:
           try:
               prices.append(float(part.strip()))
           except ValueError:
               pass
       return sum(prices) / len(prices) if prices else 0
   ```

2. **None Attribute Error** - `'NoneType' object has no attribute 'lower'`
   - Problem: `p.get('brand', '').lower()` fails when brand is None (not missing)
   - Fix: Check None first:
   ```python
   def get_brand(p):
       brand = p.get('brand')
       return brand.lower() if brand else ''
   ```

3. **Import Error** - `__import__ not found`
   - Problem: Trying to import Counter/defaultdict
   - Fix: Don't import! They're already available:
   ```python
   # WRONG: from collections import Counter
   # RIGHT: Just use Counter() directly
   brand_counts = Counter(brands)
   ```

4. **KeyError** - Field doesn't exist
   - Problem: `p['price']` when price doesn't exist
   - Fix: Use .get() with defaults:
   ```python
   price = p.get('price') or p.get('current_price_vnd') or 0
   ```

5. **Division by Zero** - No matching products
   - Problem: `sum(prices) / len(prices)` when prices is empty
   - Fix: Check before dividing:
   ```python
   if prices:
       result = sum(prices) / len(prices)
   else:
       result = 0
   ```

**Your Task:**
Analyze the error, identify which pattern it matches, and generate ONLY the corrected Python code.

IMPORTANT:
- Return ONLY the corrected code in a ```python``` block
- NO explanations, NO JSON, NO analysis text
- Keep the same structure as original code
- Fix ONLY the specific error, don't rewrite everything

Corrected code:
""",

    "natural_language_formatting": """Convert this technical result into a natural, friendly answer.

Query: {query}
Result: {result}
Intent: {intent}

Generate a clear, concise Vietnamese answer (2-3 sentences max).
"""
}


# =============================================================================
# PROMPT UTILITIES
# =============================================================================

def get_prompt(prompt_name: str, **kwargs) -> str:
    """
    Get formatted prompt by name.
    
    Args:
        prompt_name: Name of prompt template
        **kwargs: Variables to format into template
        
    Returns:
        Formatted prompt string
    """
    template = PROMPTS.get(prompt_name)
    
    if not template:
        raise ValueError(f"Unknown prompt: {prompt_name}")
    
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing required variable for prompt '{prompt_name}': {e}")


def list_prompts() -> list:
    """List all available prompt names."""
    return list(PROMPTS.keys())


def add_prompt(name: str, template: str):
    """
    Add or update a prompt template at runtime.
    
    Args:
        name: Prompt name
        template: Prompt template string
    """
    PROMPTS[name] = template


def get_prompt_info() -> dict:
    """Get metadata about all prompts."""
    return {
        name: {
            "length": len(template),
            "variables": _extract_variables(template)
        }
        for name, template in PROMPTS.items()
    }


def _extract_variables(template: str) -> list:
    """Extract format variables from template."""
    import re
    return re.findall(r'\{(\w+)\}', template)


# =============================================================================
# PROMPT VERSIONING (Optional - for A/B testing)
# =============================================================================

PROMPT_VERSIONS = {
    # Example: multiple versions of same prompt for testing
    # "unified_analysis_and_code_v2": "...",
    # "unified_analysis_and_code_v3": "...",
}


def get_prompt_version(prompt_name: str, version: int = 1) -> str:
    """
    Get specific version of a prompt.
    
    Args:
        prompt_name: Base prompt name
        version: Version number (default 1)
        
    Returns:
        Prompt template
    """
    if version == 1:
        return PROMPTS.get(prompt_name, "")
    
    versioned_name = f"{prompt_name}_v{version}"
    return PROMPT_VERSIONS.get(versioned_name, PROMPTS.get(prompt_name, ""))
