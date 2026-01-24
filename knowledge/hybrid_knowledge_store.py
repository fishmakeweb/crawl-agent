"""
Hybrid Knowledge Store: Vector + Graph + Cache
Combines semantic search, relationship reasoning, and fast retrieval
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase
import redis
from typing import Dict, List, Any, Optional
import hashlib
import json
import asyncio
from datetime import datetime
import numpy as np
import os
import time
import logging
from functools import wraps
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_connection(max_retries=None, delay=None):
    """Retry decorator for database connections with exponential backoff"""
    if max_retries is None:
        max_retries = int(os.getenv("DB_MAX_RETRIES", "10"))
    if delay is None:
        delay = int(os.getenv("DB_RETRY_DELAY_SECONDS", "5"))

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Connection failed after {max_retries} attempts: {e}")
                        raise
            return None
        return wrapper
    return decorator


class HybridKnowledgeStore:
    """
    Three-tier knowledge architecture:
    - Vector: Semantic pattern matching (Qdrant)
    - Graph: Entity relationships and reasoning (Neo4j)
    - Cache: Hot data for fast retrieval (Redis)
    """

    def __init__(self, gemini_client, rl_controller, config):
        self.config = config
        self.gemini_client = gemini_client
        self.rl_controller = rl_controller
        self.collection_name = config.COLLECTION_NAME

        # Service availability flags
        self.vector_available = False
        self.graph_available = False
        self.cache_available = False

        # Metrics for RL controller
        self.metrics = {
            "vector_size_mb": 0.0,
            "graph_nodes": 0,
            "graph_relationships": 0,
            "cache_hit_rate": 0.0,
            "retrieval_frequency": {},
            "pattern_redundancy": 0.0,
            "total_patterns": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Initialize connections with retry logic
        self._connect_vector_store(config)
        self._connect_graph_store(config)
        self._connect_cache(config)

        # Initialize schema
        self._init_schema()

        # Log final status
        logger.info(f"Knowledge Store initialized:")
        logger.info(f"  ✓ Vector: {'Available' if self.vector_available else 'UNAVAILABLE'}")
        logger.info(f"  {'✓' if self.graph_available else '✗'} Graph: {'Available' if self.graph_available else 'UNAVAILABLE (degraded mode)'}")
        logger.info(f"  ✓ Cache: {'Available' if self.cache_available else 'UNAVAILABLE'}")

    @retry_connection()
    def _connect_vector_store(self, config):
        """Connect to Qdrant vector store with retry"""
        logger.info("Connecting to Qdrant vector store...")
        self.vector_store = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            timeout=int(os.getenv("DB_CONNECTION_TIMEOUT_SECONDS", "30"))
        )
        # Test connection
        self.vector_store.get_collections()
        self.vector_available = True
        logger.info(f"✅ Qdrant connected: {config.QDRANT_HOST}:{config.QDRANT_PORT}")

    @retry_connection()
    def _connect_graph_store(self, config):
        """Connect to Neo4j graph store with retry (optional, graceful degradation)"""
        logger.info("Connecting to Neo4j graph store...")
        self.graph_store = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            connection_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT_SECONDS", "30"))
        )
        # Test connection
        with self.graph_store.session() as session:
            session.run("RETURN 1")
        self.graph_available = True
        logger.info(f"✅ Neo4j connected: {config.NEO4J_URI}")

    @retry_connection()
    def _connect_cache(self, config):
        """Connect to Redis cache with retry"""
        logger.info("Connecting to Redis cache...")
        self.cache = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=int(os.getenv("DB_CONNECTION_TIMEOUT_SECONDS", "30"))
        )
        # Test connection
        self.cache.ping()
        self.cache_ttl = config.CACHE_TTL_SECONDS
        self.cache_available = True
        logger.info(f"✅ Redis connected: {config.REDIS_HOST}:{config.REDIS_PORT}")

    def _init_schema(self):
        """Initialize vector collection and graph schema"""
        # Vector store schema
        if self.vector_available:
            try:
                # Check if collection exists first
                collections = self.vector_store.get_collections().collections
                collection_exists = any(c.name == self.collection_name for c in collections)

                if not collection_exists:
                    # Create vector collection
                    self.vector_store.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.config.VECTOR_DIMENSION,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"✅ Created vector collection: {self.collection_name}")
                else:
                    logger.info(f"ℹ️  Vector collection already exists: {self.collection_name}")
            except Exception as e:
                logger.warning(f"⚠️  Error initializing vector collection: {e}")

        # Graph store schema (only if available)
        if self.graph_available:
            try:
                with self.graph_store.session() as session:
                    # Create indexes for performance
                    session.run("""
                        CREATE INDEX domain_idx IF NOT EXISTS
                        FOR (d:Domain) ON (d.name)
                    """)
                    session.run("""
                        CREATE INDEX pattern_idx IF NOT EXISTS
                        FOR (p:Pattern) ON (p.id)
                    """)
                    session.run("""
                        CREATE INDEX field_idx IF NOT EXISTS
                        FOR (f:Field) ON (f.name)
                    """)
                    logger.info("✅ Created graph indexes")
            except Exception as e:
                logger.warning(f"⚠️  Graph index creation error: {e}")
                self.graph_available = False
        else:
            logger.warning("⚠️  Skipping graph schema initialization (Neo4j unavailable)")

    async def store_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Store pattern in all three layers"""
        try:
            pattern_id = pattern.get("id") or self._generate_id(pattern)
            pattern["id"] = pattern_id

            # 1. Vector store: semantic embedding
            embedding = await self._embed_pattern(pattern)

            self.vector_store.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=pattern_id,
                    vector=embedding,
                    payload={
                        "domain": pattern.get("domain", "unknown"),
                        "type": pattern.get("type", "unknown"),
                        "success_rate": pattern.get("success_rate", 0.0),
                        "frequency": pattern.get("frequency", 1),
                        "metadata": pattern.get("metadata", {}),
                        "created_at": pattern.get("created_at", datetime.now().isoformat())
                    }
                )]
            )

            # 2. Graph store: relationships (only if available)
            if self.graph_available:
                await self._store_in_graph(pattern)

            # 3. Cache: if high frequency
            if pattern.get("frequency", 1) > 5:
                cache_key = f"pattern:{pattern_id}"
                self.cache.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(pattern)
                )

            # Update metrics
            self.metrics["total_patterns"] += 1

            return True

        except Exception as e:
            print(f"❌ Error storing pattern: {e}")
            return False

    async def retrieve_patterns(
        self,
        query: Dict[str, Any],
        top_k: int = None
    ) -> List[Dict]:
        """
        Multi-tier retrieval strategy:
        1. Check cache first (hot patterns)
        2. Vector search for semantic similarity
        3. Graph traversal for relationship reasoning
        """
        if top_k is None:
            top_k = self.config.TOP_K_PATTERNS

        # Tier 1: Cache lookup
        cache_key = self._generate_cache_key(query)
        cached = self.cache.get(cache_key)

        if cached:
            self.metrics["cache_hits"] += 1
            self._update_cache_hit_rate()
            return json.loads(cached)

        self.metrics["cache_misses"] += 1

        # Tier 2: Vector search (using HTTP REST API for compatibility)
        query_embedding = await self._embed_query(query)

        try:
            # Use REST API directly (compatible with Qdrant 1.7.x)
            search_url = f"http://{self.config.QDRANT_HOST}:{self.config.QDRANT_PORT}/collections/{self.collection_name}/points/search"
            
            search_body = {
                "vector": query_embedding,
                "limit": top_k,
                "score_threshold": self.config.SIMILARITY_THRESHOLD,
                "with_payload": True
            }
            
            response = requests.post(search_url, json=search_body, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"❌ Qdrant search failed: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            results_list = data.get("result", [])
            
            patterns = []
            for hit in results_list:
                payload = hit.get("payload", {})
                patterns.append({
                    **payload,
                    "id": hit.get("id"),
                    "score": hit.get("score", 0)
                })
            
            logger.info(f"🔍 Vector search found {len(patterns)} patterns (threshold={self.config.SIMILARITY_THRESHOLD})")
            
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return []

        # Tier 3: Graph enrichment (only if Neo4j available)
        if self.graph_available and query.get("include_related", True) and patterns:
            logger.info(f"🔗 Enriching with Neo4j graph relationships...")
            patterns = await self._enrich_with_graph(patterns, query)
        elif not self.graph_available:
            logger.info(f"⚠️  Skipping graph enrichment (Neo4j unavailable)")
        else:
            logger.info(f"ℹ️  Skipping graph enrichment (no patterns or disabled)")

        # Cache the result
        self.cache.setex(cache_key, self.cache_ttl, json.dumps(patterns))

        # Update metrics for RL controller
        self._update_retrieval_metrics(query, patterns)
        self._update_cache_hit_rate()

        return patterns

    async def _store_in_graph(self, pattern: Dict):
        """Store pattern and relationships in graph"""
        with self.graph_store.session() as session:
            # Create pattern node
            session.run("""
                MERGE (p:Pattern {id: $id})
                SET p.type = $type,
                    p.success_rate = $success_rate,
                    p.frequency = $frequency,
                    p.created_at = datetime($created_at),
                    p.updated_at = datetime()
            """,
            id=pattern["id"],
            type=pattern.get("type", "unknown"),
            success_rate=pattern.get("success_rate", 0.0),
            frequency=pattern.get("frequency", 1),
            created_at=pattern.get("created_at", datetime.now().isoformat()))

            # Link to domain
            if pattern.get("domain"):
                session.run("""
                    MERGE (d:Domain {name: $domain})
                    WITH d
                    MATCH (p:Pattern {id: $pattern_id})
                    MERGE (p)-[:APPLIES_TO]->(d)
                """,
                domain=pattern["domain"],
                pattern_id=pattern["id"])

            # Link to extraction fields
            for field in pattern.get("extraction_fields", []):
                session.run("""
                    MERGE (f:Field {name: $field})
                    WITH f
                    MATCH (p:Pattern {id: $pattern_id})
                    MERGE (p)-[:EXTRACTS]->(f)
                """,
                field=field,
                pattern_id=pattern["id"])

    async def _enrich_with_graph(
        self,
        patterns: List[Dict],
        query: Dict
    ) -> List[Dict]:
        """Use graph traversal to find related patterns"""
        if not patterns:
            return patterns

        pattern_ids = [p["id"] for p in patterns]
        logger.info(f"🔗 Neo4j: Searching for patterns related to {len(pattern_ids)} found patterns...")

        with self.graph_store.session() as session:
            # Find related patterns through graph traversal
            result = session.run("""
                MATCH (p:Pattern)
                WHERE p.id IN $ids

                // Find similar patterns (direct similarity)
                OPTIONAL MATCH (p)-[:SIMILAR_TO]->(similar:Pattern)
                WHERE similar.success_rate > 0.7

                // Find domain-related patterns
                OPTIONAL MATCH (p)-[:APPLIES_TO]->(d:Domain)<-[:APPLIES_TO]-(domain_related:Pattern)
                WHERE domain_related.success_rate > 0.7
                  AND NOT domain_related.id IN $ids

                RETURN DISTINCT
                    similar.id as similar_id,
                    domain_related.id as domain_id
                LIMIT 10
            """, ids=pattern_ids)

            related_ids = set()
            similar_count = 0
            domain_count = 0
            
            for record in result:
                if record["similar_id"]:
                    related_ids.add(record["similar_id"])
                    similar_count += 1
                if record["domain_id"]:
                    related_ids.add(record["domain_id"])
                    domain_count += 1
            
            logger.info(f"🔗 Neo4j: Found {similar_count} SIMILAR_TO, {domain_count} domain-related → {len(related_ids)} unique patterns")

            # Fetch related patterns from vector store
            if related_ids:
                related_points = self.vector_store.retrieve(
                    collection_name=self.collection_name,
                    ids=list(related_ids)
                )

                for point in related_points:
                    patterns.append({
                        **point.payload,
                        "id": point.id,
                        "score": 0.5,  # Lower score for related patterns
                        "relation": "graph_enriched"
                    })

        return patterns

    async def consolidate_patterns(self) -> int:
        """
        Domain-based grouping + semantic clustering + frequency merging
        Triggered by RL controller
        """
        print("🔄 Starting pattern consolidation...")

        # 1. Domain-based grouping
        domain_groups = await self._group_by_domain()

        merged_count = 0
        for domain, pattern_ids in domain_groups.items():
            if len(pattern_ids) < 2:
                continue

            print(f"  Processing domain: {domain} ({len(pattern_ids)} patterns)")

            # 2. Semantic similarity clustering
            clusters = await self._cluster_by_similarity(
                pattern_ids,
                threshold=0.85
            )

            for cluster in clusters:
                if len(cluster) < 2:
                    continue

                # 3. Frequency-based merging (high-usage → high priority)
                patterns = await self._fetch_patterns(cluster)
                merged = await self._merge_patterns(patterns)

                # Replace cluster with merged pattern
                await self._replace_cluster(cluster, merged)
                merged_count += 1

        print(f"✅ Consolidated {merged_count} pattern clusters")
        return merged_count

    async def _group_by_domain(self) -> Dict[str, List[str]]:
        """Group patterns by domain"""
        with self.graph_store.session() as session:
            result = session.run("""
                MATCH (p:Pattern)-[:APPLIES_TO]->(d:Domain)
                RETURN d.name as domain, collect(p.id) as pattern_ids
            """)

            return {
                record["domain"]: record["pattern_ids"]
                for record in result
            }

    async def _cluster_by_similarity(
        self,
        pattern_ids: List[str],
        threshold: float
    ) -> List[List[str]]:
        """Cluster patterns by semantic similarity using vector embeddings"""
        try:
            from sklearn.cluster import DBSCAN

            # Fetch vectors
            points = self.vector_store.retrieve(
                collection_name=self.collection_name,
                ids=pattern_ids
            )

            if not points or len(points) < 2:
                return []

            # Extract embeddings
            embeddings = np.array([p.vector for p in points])
            ids = [p.id for p in points]

            # DBSCAN clustering (eps = 1 - threshold for cosine similarity)
            clustering = DBSCAN(
                eps=1 - threshold,
                min_samples=2,
                metric='cosine'
            )
            labels = clustering.fit_predict(embeddings)

            # Group by cluster
            clusters = {}
            for idx, label in enumerate(labels):
                if label == -1:  # Noise
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(ids[idx])

            return list(clusters.values())

        except Exception as e:
            print(f"⚠️  Clustering error: {e}")
            return []

    async def _fetch_patterns(self, pattern_ids: List[str]) -> List[Dict]:
        """Fetch full pattern data"""
        points = self.vector_store.retrieve(
            collection_name=self.collection_name,
            ids=pattern_ids
        )

        return [
            {**p.payload, "id": p.id}
            for p in points
        ]

    async def _merge_patterns(self, patterns: List[Dict]) -> Dict:
        """
        Use Gemini to intelligently merge patterns
        Prioritize high-frequency patterns
        """

        # Sort by frequency (high-usage → high priority)
        patterns_sorted = sorted(
            patterns,
            key=lambda p: p.get("frequency", 0),
            reverse=True
        )

        merge_prompt = f"""
Merge these {len(patterns)} similar extraction patterns into one unified pattern.
Prioritize the higher-frequency patterns (they're sorted by usage).

Patterns:
{json.dumps(patterns_sorted, indent=2)}

Create a merged pattern that:
1. Captures all successful cases
2. Generalizes common elements
3. Preserves domain-specific nuances
4. Maintains high success rate

Return as JSON with structure:
{{
    "type": "...",
    "domain": "...",
    "selectors": [...],
    "extraction_logic": "...",
    "applicable_domains": [...],
    "merged_from": [pattern_ids],
    "estimated_success_rate": 0.0-1.0,
    "frequency": sum_of_frequencies
}}
"""

        merged_json = await self.gemini_client.generate(
            merge_prompt,
            response_mime_type="application/json"
        )

        merged = json.loads(merged_json)
        merged["id"] = self._generate_id(merged)
        merged["created_at"] = datetime.now().isoformat()

        return merged

    async def _replace_cluster(self, old_ids: List[str], merged: Dict):
        """Replace cluster of patterns with merged pattern"""
        # Store merged pattern
        await self.store_pattern(merged)

        # Delete old patterns
        try:
            self.vector_store.delete(
                collection_name=self.collection_name,
                points_selector=old_ids
            )

            # Delete from graph
            with self.graph_store.session() as session:
                session.run("""
                    MATCH (p:Pattern)
                    WHERE p.id IN $ids
                    DETACH DELETE p
                """, ids=old_ids)

        except Exception as e:
            print(f"⚠️  Error deleting old patterns: {e}")

    async def _embed_pattern(self, pattern: Dict) -> List[float]:
        """
        Generate embedding for pattern using Gemini
        Includes type, domain, extraction fields, and pagination info for better semantic matching
        """
        # Build rich text representation
        parts = [
            f"Pattern type: {pattern.get('type', 'unknown')}",
            f"Domain: {pattern.get('domain', 'unknown')}",
        ]
        
        # Add extraction fields if available
        fields = pattern.get('extraction_fields', [])
        if fields:
            parts.append(f"Extracts: {', '.join(fields)}")
        
        # Add description
        if pattern.get('description'):
            parts.append(pattern['description'])
        
        # Add metadata context
        metadata = pattern.get('metadata', {})
        if metadata.get('user_prompt'):
            parts.append(f"User intent: {metadata['user_prompt']}")
        
        # Add pagination info if used
        pagination = metadata.get('pagination', {})
        if pagination.get('used_pagination'):
            parts.append(f"Uses pagination ({pagination.get('pages_crawled', 0)} pages via {pagination.get('pagination_strategy', 'unknown')})")
        
        text = ". ".join(parts)
        return await self.gemini_client.embed(text)

    async def _embed_query(self, query: Dict) -> List[float]:
        """
        Generate embedding for search query
        Includes domain, intent, description, and requested fields
        """
        # Build rich query representation
        parts = []
        
        if query.get('domain'):
            parts.append(f"Domain: {query['domain']}")
        
        if query.get('intent'):
            parts.append(f"Intent: {query['intent']}")
        
        if query.get('description'):
            parts.append(query['description'])
        
        # Include requested extraction fields
        if query.get('extraction_fields'):
            fields = query['extraction_fields']
            parts.append(f"Extract: {', '.join(fields)}")
        
        # Include user prompt/description
        if query.get('user_description'):
            parts.append(f"User wants: {query['user_description']}")
        
        text = ". ".join(parts) if parts else "generic extraction"
        return await self.gemini_client.embed(text)

    def _generate_cache_key(self, query: Dict) -> str:
        """Generate cache key from query"""
        query_str = json.dumps(query, sort_keys=True)
        return f"query:{hashlib.md5(query_str.encode()).hexdigest()}"

    def _generate_id(self, pattern: Dict) -> int:
        """Generate unique ID for pattern as integer for Qdrant compatibility"""
        content = json.dumps(pattern, sort_keys=True)
        hash_hex = hashlib.sha256(content.encode()).hexdigest()[:16]
        # Convert hex to int (use first 16 hex chars for 64-bit integer)
        return int(hash_hex, 16) % (2**63 - 1)  # Keep within signed 64-bit range

    def _update_retrieval_metrics(self, query: Dict, patterns: List[Dict]):
        """Update metrics for RL controller"""
        domain = query.get("domain", "unknown")
        if domain not in self.metrics["retrieval_frequency"]:
            self.metrics["retrieval_frequency"][domain] = 0
        self.metrics["retrieval_frequency"][domain] += 1

    def _update_cache_hit_rate(self):
        """Update cache hit rate metric"""
        total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        if total > 0:
            self.metrics["cache_hit_rate"] = self.metrics["cache_hits"] / total

    def get_metrics(self) -> Dict[str, Any]:
        """Expose metrics to RL controller"""
        # Update size metrics
        try:
            collection_info = self.vector_store.get_collection(self.collection_name)
            self.metrics["total_patterns"] = collection_info.points_count
            # Estimate: 768 dims * 4 bytes per float + payload overhead
            self.metrics["vector_size_mb"] = (
                collection_info.points_count * 768 * 4 / (1024 * 1024)
            )
        except:
            pass

        try:
            with self.graph_store.session() as session:
                result = session.run("""
                    MATCH (n)
                    RETURN count(n) as node_count
                """)
                self.metrics["graph_nodes"] = result.single()["node_count"]

                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN count(r) as rel_count
                """)
                self.metrics["graph_relationships"] = result.single()["rel_count"]
        except:
            pass

        # Estimate redundancy (simplified)
        self.metrics["pattern_redundancy"] = self._estimate_redundancy()

        return self.metrics

    def _estimate_redundancy(self) -> float:
        """Estimate pattern redundancy ratio"""
        # Simplified: assume 15% redundancy on average
        # In production, this would analyze actual similarity scores
        return 0.15

    def get_domain_patterns(self) -> Dict[str, Any]:
        """Retrieve learned patterns organized by domain"""
        try:
            with self.graph_store.session() as session:
                result = session.run("""
                    MATCH (p:Pattern)-[:APPLIES_TO]->(d:Domain)
                    RETURN d.name as domain,
                           collect({
                               id: p.id,
                               type: p.type,
                               success_rate: p.success_rate,
                               frequency: p.frequency
                           }) as patterns
                """)

                domain_patterns = {}
                for record in result:
                    domain_patterns[record["domain"]] = record["patterns"]

                return domain_patterns
        except Exception as e:
            print(f"⚠️  Error fetching domain patterns: {e}")
            return {}

    async def add_patterns(self, patterns: List[Dict]):
        """Batch add patterns (async)"""
        for pattern in patterns:
            await self.store_pattern(pattern)

    async def add_failure_patterns(self, patterns: List[Dict]):
        """Store patterns from failed attempts for learning"""
        for pattern in patterns:
            pattern["type"] = "failure"
            pattern["success_rate"] = 0.0
            await self.add_patterns([pattern])

    async def add_feedback_insights(self, insights: List[Dict]):
        """Store insights from user feedback"""
        for insight in insights:
            pattern = {
                "type": "feedback_insight",
                "domain": insight.get("domain", "unknown"),
                "description": insight.get("interpreted", {}).get("specific_issues", []),
                "metadata": insight,
                "frequency": 1,
                "success_rate": 0.5  # Neutral until validated
            }
            await self.add_patterns([pattern])

    async def get_patterns_by_type(self, pattern_type: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve patterns by specific type (e.g., 'product_list', 'article_extraction')
        Useful for type-specific strategy selection
        """
        try:
            # Use Qdrant filter for efficient type-based retrieval
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            search_results = self.vector_store.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value=pattern_type)
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            patterns = [
                {**point.payload, "id": point.id}
                for point in search_results[0]  # scroll returns (points, next_offset)
            ]
            
            # Sort by success_rate and frequency
            patterns.sort(
                key=lambda p: (p.get("success_rate", 0), p.get("frequency", 0)),
                reverse=True
            )
            
            return patterns[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️  Error retrieving patterns by type '{pattern_type}': {e}")
            return []

    async def get_best_practices(self, domain: str = None, pattern_type: str = None) -> List[Dict]:
        """
        Retrieve best practice patterns (high success rate + high frequency)
        Optionally filter by domain and/or pattern type
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
            
            # Build filter conditions
            filter_conditions = []
            
            # Must have high success rate (> 0.8)
            filter_conditions.append(
                FieldCondition(
                    key="success_rate",
                    range=Range(gte=0.8)
                )
            )
            
            # Optional domain filter
            if domain:
                filter_conditions.append(
                    FieldCondition(
                        key="domain",
                        match=MatchValue(value=domain)
                    )
                )
            
            # Optional type filter
            if pattern_type:
                filter_conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=pattern_type)
                    )
                )
            
            search_results = self.vector_store.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=filter_conditions),
                limit=20,
                with_payload=True,
                with_vectors=False
            )
            
            patterns = [
                {**point.payload, "id": point.id}
                for point in search_results[0]
            ]
            
            # Sort by weighted score (success_rate * log(frequency + 1))
            import math
            for p in patterns:
                p["best_practice_score"] = (
                    p.get("success_rate", 0) * 
                    math.log(p.get("frequency", 0) + 1)
                )
            
            patterns.sort(key=lambda p: p["best_practice_score"], reverse=True)
            
            logger.info(f"📚 Found {len(patterns)} best practices (domain={domain}, type={pattern_type})")
            return patterns[:10]
            
        except Exception as e:
            logger.warning(f"⚠️  Error retrieving best practices: {e}")
            return []

    async def get_pattern_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about pattern types distribution
        Useful for understanding what the agent has learned
        """
        try:
            stats = {
                "total_patterns": 0,
                "by_type": {},
                "by_domain": {},
                "high_success_patterns": 0,
                "frequently_used": 0,
                "pagination_stats": {
                    "patterns_with_pagination": 0,
                    "avg_pages_crawled": 0.0,
                    "strategies_used": {}
                }
            }
            
            # Get all patterns
            all_patterns = self.vector_store.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Reasonable limit
                with_payload=True,
                with_vectors=False
            )[0]
            
            stats["total_patterns"] = len(all_patterns)
            
            # Track pagination stats
            total_pages = 0
            pagination_count = 0
            
            for point in all_patterns:
                payload = point.payload
                
                # Count by type
                pattern_type = payload.get("type", "unknown")
                stats["by_type"][pattern_type] = stats["by_type"].get(pattern_type, 0) + 1
                
                # Count by domain
                domain = payload.get("domain", "unknown")
                stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
                
                # High success patterns
                if payload.get("success_rate", 0) > 0.8:
                    stats["high_success_patterns"] += 1
                
                # Frequently used
                if payload.get("frequency", 0) > 5:
                    stats["frequently_used"] += 1
                
                # Pagination stats
                metadata = payload.get("metadata", {})
                pagination = metadata.get("pagination", {})
                if pagination.get("used_pagination"):
                    stats["pagination_stats"]["patterns_with_pagination"] += 1
                    pagination_count += 1
                    
                    pages = pagination.get("pages_crawled", 0)
                    total_pages += pages
                    
                    strategy = pagination.get("pagination_strategy", "unknown")
                    stats["pagination_stats"]["strategies_used"][strategy] = \
                        stats["pagination_stats"]["strategies_used"].get(strategy, 0) + 1
            
            # Calculate average pages crawled
            if pagination_count > 0:
                stats["pagination_stats"]["avg_pages_crawled"] = total_pages / pagination_count
            
            logger.info(f"📊 Pattern Statistics: {stats['total_patterns']} total, "
                       f"{len(stats['by_type'])} types, {len(stats['by_domain'])} domains, "
                       f"{stats['pagination_stats']['patterns_with_pagination']} with pagination")
            
            return stats
            
        except Exception as e:
            logger.warning(f"⚠️  Error getting pattern statistics: {e}")
            return {}
    
    async def get_pagination_patterns(self, domain: Optional[str] = None, 
                                      pattern_type: Optional[str] = None,
                                      top_k: int = 5) -> List[Dict]:
        """
        Retrieve patterns that successfully used pagination
        Helps agent learn pagination strategies from past successes
        
        Args:
            domain: Optional domain filter (e.g., "shopee.vn")
            pattern_type: Optional pattern type filter (e.g., "product_list")
            top_k: Number of results to return
            
        Returns:
            List of patterns with pagination info, sorted by success rate
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            filter_conditions = []
            
            # Must have used pagination
            # Note: Qdrant doesn't support nested field filtering directly
            # We'll filter in Python after retrieval
            
            # Optional domain filter
            if domain:
                filter_conditions.append(
                    FieldCondition(
                        key="domain",
                        match=MatchValue(value=domain)
                    )
                )
            
            # Optional type filter
            if pattern_type:
                filter_conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=pattern_type)
                    )
                )
            
            # Retrieve candidates
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            search_results = self.vector_store.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=100,  # Get more candidates for filtering
                with_payload=True,
                with_vectors=False
            )
            
            # Filter for patterns with pagination
            pagination_patterns = []
            for point in search_results[0]:
                payload = point.payload
                pagination = payload.get("metadata", {}).get("pagination", {})
                
                if pagination.get("used_pagination"):
                    pagination_patterns.append({
                        **payload,
                        "id": point.id,
                        "pagination_info": pagination
                    })
            
            # Sort by success rate
            pagination_patterns.sort(
                key=lambda p: p.get("success_rate", 0), 
                reverse=True
            )
            
            logger.info(f"🔄 Found {len(pagination_patterns)} pagination patterns "
                       f"(domain={domain}, type={pattern_type})")
            
            return pagination_patterns[:top_k]
            
        except Exception as e:
            logger.warning(f"⚠️  Error retrieving pagination patterns: {e}")
            return []

    def close(self):
        """Cleanup connections"""
        try:
            self.graph_store.close()
            self.cache.close()
        except:
            pass
