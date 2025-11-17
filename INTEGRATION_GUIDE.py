"""
Integration Guide: Using Multi-Model Routing in Training Agent

This file shows how to integrate the new route_model() method into
the existing training_agent.py endpoints.
"""

# Example 1: Integrate routing into /train endpoint
# ================================================

# BEFORE (in training_agent.py around line 200):
"""
@app.post("/train")
async def train_crawl(job: CrawlJob):
    result = await shared_agent.execute_crawl(
        url=job.url,
        task_description=job.task_description,
    )
    return result
"""

# AFTER (with routing):
"""
@app.post("/train")
async def train_crawl(job: CrawlJob):
    # Initialize or get session metrics
    session_id = job.session_id or str(uuid.uuid4())
    if session_id not in active_sessions:
        active_sessions[session_id] = {
            "tokens_used": 0,
            "cost_usd": 0.0,
            "mode": "training",
            "requests": 0
        }
    
    metrics = active_sessions[session_id]
    
    # Execute crawl (internal uses gemini_client which now has routing)
    result = await shared_agent.execute_crawl(
        url=job.url,
        task_description=job.task_description,
        metrics=metrics  # Pass metrics if you modify execute_crawl
    )
    
    # Update session metrics
    metrics["requests"] += 1
    
    # Broadcast stats via Socket.IO
    await sio.emit('training_stats', {
        "session_id": session_id,
        "metrics": metrics,
        "model_usage": gemini_client.get_stats()["model_usage"]
    })
    
    return result
"""


# Example 2: Use routing explicitly for feedback interpretation
# ============================================================

# ADD to training_agent.py:
"""
@app.post("/feedback-with-routing")
async def submit_feedback_routed(feedback: FeedbackSubmission):
    try:
        # Get or create session metrics
        session_id = feedback.session_id or "default"
        metrics = active_sessions.get(session_id, {
            "tokens_used": 0,
            "cost_usd": 0.0,
            "mode": "training"
        })
        
        # Use routing for feedback interpretation
        model_id, interpretation_json, updated_metrics = await gemini_client.route_model(
            task_type="feedback",
            input_data={
                "prompt": f'''
User provided this feedback on a web crawl:
"{feedback.feedback}"

Crawl context:
- URL: {feedback.url}
- Job ID: {feedback.job_id}

Interpret this feedback as structured learning signals:
{{
    "confidence": 0.0-1.0,
    "quality_rating": 1-5,
    "specific_issues": ["issue1", "issue2"],
    "desired_improvements": ["improvement1", "improvement2"],
    "clarification_needed": true/false,
    "clarification_question": "..."
}}

Return as JSON only.
''',
                "json_mode": True
            },
            current_metrics=metrics
        )
        
        # Parse interpretation
        interpretation = json.loads(interpretation_json)
        
        # Store in session
        active_sessions[session_id] = updated_metrics
        
        # Check if clarification needed
        if interpretation.get("clarification_needed"):
            # Use LearnLM for pedagogical clarification
            clarification_model, clarification_text, metrics = await gemini_client.route_model(
                task_type="clarification",
                input_data={
                    "prompt": f"Rephrase this question pedagogically: {interpretation['clarification_question']}"
                },
                current_metrics=updated_metrics
            )
            
            return {
                "status": "clarification_needed",
                "question": clarification_text,
                "model_used": clarification_model,
                "cost": metrics["last_request_cost"]
            }
        
        # Apply learning
        await learning_algorithm.apply_feedback(
            feedback=feedback.feedback,
            interpretation=interpretation,
            crawl_result=None  # Fetch from storage
        )
        
        return {
            "status": "success",
            "interpretation": interpretation,
            "model_used": model_id,
            "cost": updated_metrics["last_request_cost"],
            "total_session_cost": updated_metrics["cost_usd"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""


# Example 3: Add cost monitoring endpoint
# =======================================

# ADD to training_agent.py:
"""
@app.get("/stats/routing")
async def get_routing_stats():
    '''Get detailed routing statistics'''
    stats = gemini_client.get_stats()
    
    return {
        "routing_enabled": stats.get("routing_enabled", False),
        "model_usage": stats.get("model_usage", {}),
        "rate_limit_warnings": stats.get("rate_limit_warnings", 0),
        "model_fallbacks": stats.get("model_fallbacks", 0),
        "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
        "total_cost_estimate": stats.get("estimated_cost_usd", 0.0),
        "savings_from_cache": stats.get("estimated_savings_usd", 0.0),
        "total_requests": stats.get("total_requests", 0),
        "sessions": {
            session_id: {
                "tokens": metrics["tokens_used"],
                "cost": metrics["cost_usd"],
                "requests": metrics.get("requests", 0)
            }
            for session_id, metrics in active_sessions.items()
        }
    }
"""


# Example 4: Add session management
# =================================

# ADD at module level in training_agent.py:
"""
# Session tracking for multi-request cost aggregation
active_sessions: Dict[str, Dict[str, Any]] = {}

@app.post("/sessions/start")
async def start_session(mode: str = "training"):
    '''Start a new session with metrics tracking'''
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "tokens_used": 0,
        "cost_usd": 0.0,
        "mode": mode,
        "requests": 0,
        "created_at": asyncio.get_event_loop().time()
    }
    return {"session_id": session_id}

@app.get("/sessions/{session_id}")
async def get_session_metrics(session_id: str):
    '''Get metrics for a specific session'''
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return active_sessions[session_id]

@app.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    '''End a session and return final metrics'''
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    final_metrics = active_sessions.pop(session_id)
    return {
        "status": "session_ended",
        "final_metrics": final_metrics
    }
"""


# Example 5: Modify CrawlJob model to include session_id
# =====================================================

# UPDATE in training_agent.py:
"""
class CrawlJob(BaseModel):
    url: str
    task_description: Optional[str] = None
    session_id: Optional[str] = None  # ADD THIS
    max_pages: int = 10
    timeout: int = 30
"""


# Example 6: Use routing for prompt generation
# ===========================================

# ADD helper function in training_agent.py:
"""
async def generate_extraction_prompt_with_routing(
    domain: str,
    examples: List[Dict[str, Any]],
    metrics: Dict[str, Any]
) -> tuple[str, Dict[str, Any]]:
    '''Generate optimized extraction prompt using 2.5-flash for creativity'''
    
    examples_text = "\\n".join([
        f"Example {i+1}: {json.dumps(ex)}"
        for i, ex in enumerate(examples[:3])
    ])
    
    model_id, prompt_text, updated_metrics = await gemini_client.route_model(
        task_type="prompt_gen",
        input_data={
            "prompt": f'''
Generate an intelligent extraction prompt for {domain}.

Examples of desired data:
{examples_text}

Return a structured extraction prompt that will work with crawl4ai.
Include CSS selectors, field names, and data transformations.
''',
            "json_mode": False
        },
        current_metrics=metrics
    )
    
    return prompt_text, updated_metrics
"""


# Example 7: Deep analysis with Pro model in training
# ==================================================

# ADD periodic analysis task:
"""
@app.post("/analyze/patterns")
async def analyze_crawl_patterns(limit: int = 100):
    '''Run deep pattern analysis using 2.5-pro (training only)'''
    
    # Fetch recent patterns
    patterns = await knowledge_store.get_recent_patterns(limit=limit)
    
    metrics = {
        "tokens_used": 0,
        "cost_usd": 0.0,
        "mode": "training"
    }
    
    # Prepare analysis prompt
    patterns_text = "\\n".join([
        f"{i+1}. Domain: {p['domain']}, Success: {p['success_rate']}, Pattern: {p['selector']}"
        for i, p in enumerate(patterns)
    ])
    
    # Use routing - will select 2.5-pro for complex analysis in training mode
    model_id, analysis_json, updated_metrics = await gemini_client.route_model(
        task_type="analysis",
        input_data={
            "prompt": f'''
Analyze these {len(patterns)} crawl patterns and identify:
1. Common successful patterns (clusters)
2. Anti-patterns (consistent failures)
3. Domain-specific optimizations
4. Recommended consolidations

Patterns:
{patterns_text}

Return as JSON with clusters, insights, and recommendations.
''',
            "json_mode": True
        },
        current_metrics=metrics
    )
    
    analysis = json.loads(analysis_json)
    
    # Apply recommendations
    if analysis.get("recommendations"):
        await learning_algorithm.apply_recommendations(analysis["recommendations"])
    
    return {
        "model_used": model_id,
        "cost": updated_metrics["last_request_cost"],
        "analysis": analysis,
        "patterns_analyzed": len(patterns)
    }
"""


# Complete Modified training_agent.py Structure
# ============================================
"""
# At the top, add session tracking
active_sessions: Dict[str, Dict[str, Any]] = {}

# Modify existing endpoints to use metrics
@app.post("/train")
async def train_crawl(job: CrawlJob):
    session_id = job.session_id or str(uuid.uuid4())
    metrics = active_sessions.get(session_id, {
        "tokens_used": 0, "cost_usd": 0.0, "mode": "training", "requests": 0
    })
    
    result = await shared_agent.execute_crawl(url=job.url, task_description=job.task_description)
    metrics["requests"] += 1
    active_sessions[session_id] = metrics
    
    return {**result, "session_id": session_id, "session_metrics": metrics}

# Add new routing-aware endpoints
@app.post("/feedback-with-routing")
async def submit_feedback_routed(feedback: FeedbackSubmission):
    # See Example 2 above
    pass

@app.get("/stats/routing")
async def get_routing_stats():
    # See Example 3 above
    pass

@app.post("/sessions/start")
async def start_session(mode: str = "training"):
    # See Example 4 above
    pass

@app.post("/analyze/patterns")
async def analyze_crawl_patterns(limit: int = 100):
    # See Example 7 above
    pass
"""

print("""
Integration Steps:
==================

1. Add session tracking at module level
2. Modify CrawlJob to include session_id
3. Update /train endpoint to track metrics
4. Add /feedback-with-routing endpoint
5. Add /stats/routing endpoint
6. Add session management endpoints
7. Add /analyze/patterns for deep analysis

Then rebuild Docker image:
cd /root/projects/crawldata/MCP-Servers
docker build -t self-learning-agent-training:latest -f self-learning-agent/Dockerfile.training .
docker stop training-server && docker rm training-server
docker run -d --name training-server --restart unless-stopped -p 8091:8091 \\
  --env-file self-learning-agent/.env \\
  -v $(pwd)/self-learning-agent/knowledge_db:/app/knowledge_db \\
  self-learning-agent-training:latest
""")
