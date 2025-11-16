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
    user_description: str
    extraction_schema: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None
    user_id: Optional[str] = None


class CrawlResponse(BaseModel):
    success: bool
    data: list
    metadata: Dict[str, Any]
    error: Optional[str] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": "production",
        "resources_version": frozen_resources.get("version", 0),
        "gemini_stats": gemini_client.get_stats()
    }


@app.post("/crawl", response_model=CrawlResponse)
async def crawl(request: CrawlRequest):
    """Execute production crawl with frozen resources"""
    try:
        task = {
            "url": request.url,
            "user_description": request.user_description,
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
