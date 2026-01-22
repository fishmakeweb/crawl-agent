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
# Generate code to answer the query
# Available variables: products (list of dicts), Counter (for counting), defaultdict
# Available functions: len, sum, min, max, sorted, set, list, dict, str, int, float, round, abs, enumerate, zip, range, any, all
# DO NOT use import statements - Counter and other utilities are already available
# MUST assign final answer to variable: result

# Example for "How many products?"
# result = len(products)

# Example for "Total price of all products?"
# result = sum(float(p.get('price', 0)) for p in products)

# Example for "List all brands?"
# result = list(set(p.get('brand', '') for p in products if p.get('brand')))

# Your code here:
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
   - Use only allowed built-in functions and Counter/defaultdict
   - **For charts** (when `requires_visualization: true`):
     * Code MUST return: `result = {{"labels": [...], "values": [...]}}`
     * Do NOT return Counter objects or text summaries directly
     * Labels must be list of strings, values must be list of numbers
     * Example: `result = {{"labels": ["A", "B"], "values": [10, 20]}}`
   - Handle missing/null values gracefully
   - Assign final result to variable `result`
   - Keep code simple and efficient
   - NO imports - Counter, defaultdict are already provided

5. **Field Names:**
   - Auto-detect from schema
   - Use .get() for safe access: `p.get('price', 0)`
   - Handle different naming conventions: price/Price/PRICE

**CRITICAL - Visualization vs Listing:**
- If query has "thống kê", "phân bổ", "distribution", "theo nhóm", "theo loại" → use `visualization` (chart)
- If query just asks for "danh sách", "list", "show me", "get all" → use `listing` (text list)
- Statistics/grouping queries ALWAYS need charts, not lists!

**Examples:**

Query: "Có bao nhiêu sản phẩm?"
```json
{{"intent_type": "computational", "operation": "count", "output_format": "number", "target_fields": [], "reasoning": "Simple count query", "confidence": 0.99}}
```
```python
result = len(products)
```

Query: "Tổng giá trị tất cả sản phẩm?"
```json
{{"intent_type": "computational", "operation": "sum", "output_format": "number", "target_fields": ["price"], "reasoning": "Sum all prices", "confidence": 0.95}}
```
```python
result = sum(float(p.get('price', 0)) for p in products if p.get('price'))
```

Query: "10 sản phẩm đắt nhất?"
```json
{{"intent_type": "listing", "operation": "sort", "output_format": "list", "target_fields": ["price", "name"], "reasoning": "Sort by price descending, take top 10", "confidence": 0.90}}
```
```python
sorted_products = sorted(
    [p for p in products if p.get('price')],
    key=lambda x: float(x.get('price', 0)),
    reverse=True
)
result = [p.get('name', 'Unknown') for p in sorted_products[:10]]
```

Query: "Sản phẩm nào có giá cao nhất?"
```json
{{"intent_type": "listing", "operation": "max", "output_format": "text", "target_fields": ["price", "name"], "reasoning": "Find product with highest price", "confidence": 0.90}}
```
```python
max_product = max([p for p in products if p.get('price')], key=lambda x: float(x.get('price', 0)))
result = {{
    "name": max_product.get('name', 'Unknown'),
    "price": float(max_product.get('price', 0)),
    "type": "max_product"
}}
```

Query: "Thống kê số lượng theo thương hiệu?" or "Vẽ biểu đồ tròn theo brand" or "Thống kê phân bổ sản phẩm theo thương hiệu" (visualization)
```json
{{"intent_type": "visualization", "operation": "group", "output_format": "chart", "target_fields": ["brand"], "reasoning": "Group by brand and count for pie chart - keywords: thống kê/phân bổ/theo", "confidence": 0.95, "requires_visualization": true, "chart_type": "pie"}}
```
```python
# Count products by brand using Counter (already available - no import needed)
brands = [p.get('brand', 'Unknown') for p in products if p.get('brand')]
brand_counts = Counter(brands)
# IMPORTANT: For charts, always return labels and values as lists
result = {{
    "labels": list(brand_counts.keys()),
    "values": list(brand_counts.values())
}}
```

Query: "Vẽ biểu đồ cột theo khoảng giá" (bar chart by price ranges)
```json
{{"intent_type": "visualization", "operation": "group", "output_format": "chart", "target_fields": ["price"], "reasoning": "Group products by price ranges for bar chart", "confidence": 0.88, "requires_visualization": true, "chart_type": "bar"}}
```
```python
# Group products by price ranges
prices = [float(p.get('price', 0)) for p in products if p.get('price')]
ranges = {{"<100k": 0, "100-300k": 0, "300-500k": 0, ">500k": 0}}
for price in prices:
    if price < 100000:
        ranges["<100k"] += 1
    elif price < 300000:
        ranges["100-300k"] += 1
    elif price < 500000:
        ranges["300-500k"] += 1
    else:
        ranges[">500k"] += 1
result = {{
    "labels": list(ranges.keys()),
    "values": list(ranges.values())
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

    "error_recovery": """The previous code generated an error. Please fix it.

Original Query: {query}
Original Code:
{code}

Error:
{error}

Generate corrected Python code (same format as before).
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
