"""
crawl4ai Agent FastAPI Server
Provides intelligent web crawling with Gemini LLM integration
"""
import os
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from crawl4ai_wrapper import Crawl4AIWrapper
from kafka_publisher import Crawl4AIKafkaPublisher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="crawl4ai Agent",
    description="Intelligent web crawling with Gemini LLM",
    version="1.0.0"
)

# CORS middleware - Configure allowed origins from environment
# For production, set ALLOWED_ORIGINS="http://localhost:3000,http://localhost:5006"
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = allowed_origins_env.split(",") if allowed_origins_env != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS configured with origins: {allowed_origins}")

# Initialize Kafka publisher for real-time progress events
kafka_publisher = Crawl4AIKafkaPublisher()
logger.info(f"Kafka publisher initialized (enabled={kafka_publisher.enabled})")

# Initialize crawler with Kafka publisher
crawler_wrapper = Crawl4AIWrapper(kafka_publisher=kafka_publisher)


class NavigationStepRequest(BaseModel):
    """Single navigation step"""
    action: str  # "click", "select", "input", "scroll", "paginate", "extract"
    selector: Optional[str] = None
    value: Optional[str] = None
    description: Optional[str] = None


class CrawlRequest(BaseModel):
    """Request for intelligent crawl"""
    url: str
    prompt: str
    job_id: Optional[str] = None  # For Kafka progress tracking
    user_id: Optional[str] = None  # For Kafka progress tracking
    navigation_steps: Optional[List[Dict[str, Any]]] = None
    extract_schema: Optional[Dict[str, Any]] = None
    max_pages: int = 50


class NavigationResult(BaseModel):
    """Result of navigation execution"""
    final_url: str
    executed_steps: List[Dict[str, Any]]
    pages_collected: int


class CrawlResponse(BaseModel):
    """Response from crawl operation"""
    success: bool
    data: List[Dict[str, Any]]
    navigation_result: NavigationResult
    execution_time_ms: float
    error: Optional[str] = None


class CrawlJobAcceptedResponse(BaseModel):
    """Response when job is accepted for background processing"""
    success: bool
    job_id: str
    status: str  # "accepted"
    message: str


class QueryRequest(BaseModel):
    """Request for RAG query"""
    context: str
    query: str


class SummaryRequest(BaseModel):
    job_id: str
    data: Any
    source: str = "manual"
    prompt: Optional[str] = None


@app.post("/query")
async def answer_query(request: QueryRequest):
    """Answer a question based on provided context"""
    try:
        answer = await crawler_wrapper.answer_query(request.context, request.query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "crawl4ai-agent",
        "version": "1.0.0"
    }


async def process_crawl_job(
    request: CrawlRequest,
    crawler: Crawl4AIWrapper,
    publisher: Crawl4AIKafkaPublisher
):
    """
    Background task to process crawl job asynchronously

    Args:
        request: CrawlRequest with job details
        crawler: Crawl4AIWrapper instance
        publisher: Kafka publisher for events
    """
    try:
        logger.info(f"[Background] Starting crawl job {request.job_id}")

        result = await crawler.intelligent_crawl(
            url=request.url,
            prompt=request.prompt,
            job_id=request.job_id,
            user_id=request.user_id,
            navigation_steps=request.navigation_steps,
            extract_schema=request.extract_schema,
            max_pages=request.max_pages
        )

        logger.info(f"[Background] Job {request.job_id} completed. Extracted {len(result['data'])} items")

        # Publish final completion event with results to Kafka
        if publisher and request.job_id:
            publisher.publish_progress(
                "CrawlJobCompleted",
                request.job_id,
                request.user_id or "unknown",
                {
                    "success": True,
                    "items_count": len(result['data']),
                    "extracted_data": result['data'],
                    "final_url": result['navigation_result']['final_url'],
                    "execution_time_ms": result['execution_time_ms'],
                    "pages_collected": result['navigation_result']['pages_collected'],
                    "conversation_name": result.get('conversation_name', 'Data Collection')
                }
            )
            publisher.flush(timeout=5.0)

    except Exception as e:
        logger.error(f"[Background] Job {request.job_id} failed: {str(e)}", exc_info=True)

        # Publish error event to Kafka
        if publisher and request.job_id:
            publisher.publish_error(
                request.job_id,
                request.user_id or "unknown",
                str(e),
                {"url": request.url, "prompt": request.prompt}
            )
            publisher.flush(timeout=5.0)


@app.post("/crawl", response_model=CrawlJobAcceptedResponse, status_code=202)
async def intelligent_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """
    Accept intelligent crawl job for background processing (Fire-and-Forget)

    This endpoint returns immediately after accepting the job.
    Progress and results are published to Kafka topics:
    - Topic: crawler.job.progress
    - Events: NavigationPlanningStarted, NavigationStepCompleted, DataExtractionCompleted, CrawlJobCompleted

    Args:
        request: CrawlRequest with URL, prompt, and optional navigation steps
        background_tasks: FastAPI background task manager

    Returns:
        CrawlJobAcceptedResponse with job_id and accepted status (HTTP 202)
    """
    try:
        # Ensure we have a job_id (generate one if caller omitted it)
        if request.job_id:
            job_request = request
            job_id = request.job_id
        else:
            job_id = str(uuid.uuid4())
            job_request = request.copy(update={"job_id": job_id})
            logger.info(f"No job_id provided. Generated {job_id} for request")

        logger.info(f"Accepting crawl job: {job_id}")
        logger.info(f"URL: {job_request.url}, User: {job_request.user_id}")

        # Add job to background processing queue
        background_tasks.add_task(
            process_crawl_job,
            job_request,
            crawler_wrapper,
            kafka_publisher
        )

        # Publish job accepted event
        if kafka_publisher and job_id:
            kafka_publisher.publish_progress(
                "CrawlJobAccepted",
                job_id,
                job_request.user_id or "unknown",
                {
                    "url": job_request.url,
                    "prompt": job_request.prompt,
                    "max_pages": job_request.max_pages
                }
            )

        # Return immediately with 202 Accepted
        return CrawlJobAcceptedResponse(
            success=True,
            job_id=job_id,
            status="accepted",
            message="Crawl job accepted for background processing. Monitor Kafka topic 'crawler.job.progress' for updates."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error accepting crawl job: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to accept crawl job: {str(e)}"
        )


@app.post("/analyze-page")
async def analyze_page_structure(url: str, prompt: str):
    """
    Analyze page structure and suggest navigation strategy

    Args:
        url: Target URL
        prompt: User's intent

    Returns:
        Suggested navigation steps
    """
    try:
        logger.info(f"Analyzing page structure for: {url}")

        steps = await crawler_wrapper.analyze_and_plan(url, prompt)

        return {
            "success": True,
            "suggested_steps": steps,
            "url": url
        }

    except Exception as e:
        logger.error(f"Error analyzing page: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.post("/summary")
async def generate_summary_endpoint(request: SummaryRequest):
    """
    Generate summary and chart recommendations based on data and optional prompt.
    """
    try:
        result = await crawler_wrapper.generate_summary(request.data, request.prompt)
        return result
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "crawl4ai Agent",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "crawl": "/crawl (POST)",
            "analyze": "/analyze-page (POST)",
            "query": "/query (POST)",
            "summary": "/summary (POST)"
        }
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully close Kafka publisher on shutdown"""
    logger.info("Shutting down crawl4ai agent...")
    if kafka_publisher:
        kafka_publisher.close()
        logger.info("Kafka publisher closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
