"""
Production Agent Entry Point
Serves frozen resources with optimized performance
"""
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import os

from config import Config
from gemini_client import GeminiClient
from agents.shared_crawler_agent import SharedCrawlerAgent

# Initialize
app = FastAPI(title="Production Crawler Agent", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
config = Config()
config.MODE = "production"
config.validate()

# Initialize Gemini client
gemini_client = GeminiClient(config.gemini)

# Load frozen resources
frozen_resources = {}
if os.path.exists(config.frozen_resources_path):
    with open(config.frozen_resources_path, 'r') as f:
        frozen_resources = json.load(f)
    print(f"✅ Loaded frozen resources version {frozen_resources.get('version', 'unknown')}")
else:
    print(f"⚠️  No frozen resources found at {config.frozen_resources_path}")
    frozen_resources = {
        "version": 0,
        "extraction_prompt": "",
        "crawl_config": {},
        "domain_patterns": {}
    }

# Initialize agent in production mode
agent = SharedCrawlerAgent(gemini_client, mode="production")

print(f"🚀 Production Agent started on port 8000")


# Request models
class CrawlRequest(BaseModel):
    url: str
    user_description: Optional[str] = None
    prompt: Optional[str] = None  # Added for compatibility with .NET client
    extraction_schema: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    user_id: Optional[str] = None


class CrawlResponse(BaseModel):
    success: bool
    data: list
    metadata: Dict[str, Any]
    error: Optional[str] = None




class QueryRequest(BaseModel):
    context: str
    query: str


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "production",
        "resources_version": frozen_resources.get("version", 0),
        "gemini_stats": gemini_client.get_stats()
    }


@app.post("/query")
async def query_data(request: QueryRequest):
    """RAG Endpoint: Answer questions based on provided context"""
    try:
        if not request.context:
            return {"answer": "No context provided to answer the question."}
            
        # Direct RAG implementation using gemini_client for better control
        prompt = f"""
You are a helpful data assistant. Answer the user's question based ONLY on the provided context data.

CONTEXT DATA (JSON/Text):
{request.context[:100000]}

USER QUESTION: "{request.query}"

INSTRUCTIONS:
1. Analyze the context data to find the answer.
2. If the answer is found, provide a clear, concise summary.
3. If the answer is NOT in the context, say "I cannot find that information in the crawled data."
4. Do not hallucinate information not present in the context.

Answer:
"""
        # Use gemini_client directly
        answer = await gemini_client.generate(prompt)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))







class SummaryRequest(BaseModel):
    job_id: str
    data: list
    source: Optional[str] = "manual"

@app.post("/summary")
async def generate_summary(request: SummaryRequest):
    """Generate summary and charts for crawled data"""
    try:
        if not request.data:
            return {"summary": "No data available to summarize.", "charts": []}

        # 1. Generate Text Summary
        data_sample = request.data[:50] # Limit sample size
        data_json = json.dumps(data_sample, indent=2)
        
        prompt = f"""
You are a Data Analyst. Summarize the following dataset.

DATA SAMPLE ({len(request.data)} total records):
{data_json}

TASK:
1. Provide a concise summary of what this data represents (e.g., "List of 32 products from Nintendo Wii U category").
2. Highlight 3 key insights (e.g., "Price range is $10-$50", "Most common brand is Nintendo").
3. Mention data quality (e.g., "All records have prices", "Some descriptions are missing").

OUTPUT FORMAT (JSON):
{{
    "summaryText": "Concise summary here...",
    "insightHighlights": ["Insight 1", "Insight 2", "Insight 3"],
    "fieldCoverage": [
        {{"fieldName": "price", "coveragePercent": 100}},
        {{"fieldName": "brand", "coveragePercent": 80}}
    ]
}}
"""
        response_text = await gemini_client.generate(prompt)
        
        # Parse JSON response
        try:
            # Extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
                
            summary_data = json.loads(response_text)
        except:
            # Fallback if JSON parsing fails
            summary_data = {
                "summaryText": response_text,
                "insightHighlights": [],
                "fieldCoverage": []
            }

        # 2. Generate Charts (Python Logic)
        charts = []
        if request.data:
            keys = request.data[0].keys()
            
            # Helper to clean currency
            def clean_price(val):
                if isinstance(val, (int, float)): return val
                if not isinstance(val, str): return 0
                clean = val.replace('$', '').replace('€', '').replace(',', '').strip()
                try:
                    return float(clean)
                except:
                    return 0

            # A. Price Histogram
            price_keys = [k for k in keys if 'price' in k.lower() or 'cost' in k.lower()]
            if price_keys:
                pk = price_keys[0]
                prices = [clean_price(d.get(pk)) for d in request.data if d.get(pk)]
                if prices:
                    # Simple binning
                    min_p, max_p = min(prices), max(prices)
                    if min_p != max_p:
                        charts.append({
                            "title": f"Price Distribution ({pk})",
                            "chartType": "bar",
                            "chartData": {
                                "labels": ["Low", "Medium", "High"], # Simplified for now
                                "datasets": [{
                                    "label": "Count",
                                    "data": [
                                        len([p for p in prices if p < min_p + (max_p-min_p)*0.33]),
                                        len([p for p in prices if p >= min_p + (max_p-min_p)*0.33 and p < min_p + (max_p-min_p)*0.66]),
                                        len([p for p in prices if p >= min_p + (max_p-min_p)*0.66])
                                    ]
                                }]
                            }
                        })

            # B. Categorical Top 10
            cat_keys = [k for k in keys if k not in price_keys and 'url' not in k.lower() and 'image' not in k.lower() and 'desc' not in k.lower()]
            for ck in cat_keys:
                values = [str(d.get(ck)) for d in request.data if d.get(ck)]
                counts = {}
                for v in values:
                    counts[v] = counts.get(v, 0) + 1
                
                # If cardinality is good (2-20)
                if 1 < len(counts) < 20:
                    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    charts.append({
                        "title": f"Top {ck}",
                        "chartType": "doughnut" if len(counts) < 5 else "bar",
                        "chartData": {
                            "labels": [x[0] for x in sorted_counts],
                            "datasets": [{
                                "label": "Count",
                                "data": [x[1] for x in sorted_counts]
                            }]
                        }
                    })
                    if len(charts) >= 3: break # Limit to 3 charts

        return {
            "summaryText": summary_data.get("summaryText", ""),
            "insightHighlights": summary_data.get("insightHighlights", []),
            "fieldCoverage": summary_data.get("fieldCoverage", []),
            "chartPreviews": charts
        }

    except Exception as e:
        print(f"❌ Summary generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest):
    """Execute production crawl with frozen resources"""
    try:
        # Handle prompt/user_description compatibility
        description = request.user_description or request.prompt
        if not description:
             # Fallback if both are missing, though usually prompt is sent
             description = "Extract main content"
             print("⚠️ Warning: No prompt or user_description provided. Using default.")

        task = {
            "url": request.url,
            "user_description": description,
            "extraction_schema": request.extraction_schema or {}
        }

        # Execute with frozen resources
        result = await agent.execute_crawl(task, resources=frozen_resources)

        return CrawlResponse(
            success=result["success"],
            data=result.get("data", []),
            metadata=result.get("metadata", {}),
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get production agent statistics"""
    return {
        "mode": "production",
        "resources_version": frozen_resources.get("version", 0),
        "frozen_at": frozen_resources.get("frozen_at"),
        "gemini_stats": gemini_client.get_stats(),
        "domain_patterns_count": len(frozen_resources.get("domain_patterns", {}))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
