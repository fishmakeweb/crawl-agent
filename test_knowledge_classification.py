"""
Test script to verify intelligent pattern classification
Run this after training to see what patterns have been learned
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from gemini_client import GeminiClient
from knowledge.hybrid_knowledge_store import HybridKnowledgeStore
from knowledge.rl_controller import RLResourceController


async def test_pattern_classification():
    """Test pattern classification and retrieval"""
    
    print("🧪 Testing Intelligent Pattern Classification")
    print("=" * 80)
    
    # Initialize
    config = Config()
    config.validate()
    
    gemini_client = GeminiClient(config.gemini)
    rl_controller = RLResourceController(gemini_client, config.training)
    knowledge_store = HybridKnowledgeStore(gemini_client, rl_controller, config.knowledge_store)
    
    print("\n✅ Knowledge Store initialized\n")
    
    # Test 1: Get pattern statistics
    print("📊 Test 1: Pattern Statistics")
    print("-" * 80)
    stats = await knowledge_store.get_pattern_statistics()
    
    if stats:
        print(f"Total patterns: {stats['total_patterns']}")
        print(f"High success patterns (>80%): {stats['high_success_patterns']}")
        print(f"Frequently used (>5 times): {stats['frequently_used']}")
        
        print(f"\n📌 Pattern Types Distribution:")
        for ptype, count in sorted(stats['by_type'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {ptype:30s} {count:3d} patterns")
        
        print(f"\n🌐 Top Domains:")
        for domain, count in sorted(stats['by_domain'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {domain:30s} {count:3d} patterns")
    else:
        print("⚠️  No patterns found (store is empty)")
    
    # Test 2: Get patterns by type
    print("\n" + "=" * 80)
    print("📊 Test 2: Get Patterns by Type")
    print("-" * 80)
    
    test_types = ["product_list", "article_extraction", "review_extraction"]
    for pattern_type in test_types:
        patterns = await knowledge_store.get_patterns_by_type(pattern_type, top_k=3)
        print(f"\n🔍 {pattern_type}:")
        if patterns:
            for i, p in enumerate(patterns, 1):
                print(f"  {i}. Domain: {p.get('domain', 'N/A'):20s} "
                      f"Success: {p.get('success_rate', 0):.2f} "
                      f"Frequency: {p.get('frequency', 0):3d}")
                if p.get('extraction_fields'):
                    print(f"     Fields: {', '.join(p['extraction_fields'][:5])}")
        else:
            print(f"  No patterns found")
    
    # Test 3: Best practices
    print("\n" + "=" * 80)
    print("📊 Test 3: Best Practices")
    print("-" * 80)
    
    best = await knowledge_store.get_best_practices(pattern_type="product_list")
    if best:
        print(f"\n✨ Top 5 Product List Best Practices:")
        for i, p in enumerate(best[:5], 1):
            print(f"\n  {i}. {p.get('domain', 'N/A')}")
            print(f"     Type: {p.get('type', 'N/A')}")
            print(f"     Success Rate: {p.get('success_rate', 0):.2%}")
            print(f"     Frequency: {p.get('frequency', 0)} times")
            print(f"     Score: {p.get('best_practice_score', 0):.2f}")
            if p.get('extraction_fields'):
                print(f"     Fields: {', '.join(p['extraction_fields'])}")
    else:
        print("⚠️  No best practices found yet")
    
    # Test 4: Semantic search
    print("\n" + "=" * 80)
    print("📊 Test 4: Semantic Search")
    print("-" * 80)
    
    test_queries = [
        {
            "name": "Product extraction with prices",
            "query": {
                "intent": "extract product information",
                "description": "Get product names and prices from e-commerce site",
                "extraction_fields": ["product_name", "price"],
                "include_related": True
            }
        },
        {
            "name": "Article content extraction",
            "query": {
                "intent": "extract article content",
                "description": "Get news articles with author and date",
                "extraction_fields": ["headline", "author", "published_date"],
                "include_related": True
            }
        }
    ]
    
    for test in test_queries:
        print(f"\n🔍 Query: {test['name']}")
        patterns = await knowledge_store.retrieve_patterns(test['query'], top_k=3)
        
        if patterns:
            print(f"  Found {len(patterns)} matching patterns:")
            for i, p in enumerate(patterns, 1):
                relation = f" ({p['relation']})" if p.get('relation') == 'graph_enriched' else ""
                print(f"    {i}. Type: {p.get('type', 'N/A'):25s} "
                      f"Domain: {p.get('domain', 'N/A'):20s} "
                      f"Score: {p.get('score', 0):.2f}{relation}")
        else:
            print("  No matching patterns found")
    
    # Test 5: Domain patterns (Neo4j)
    print("\n" + "=" * 80)
    print("📊 Test 5: Domain Patterns (from Neo4j Graph)")
    print("-" * 80)
    
    domain_patterns = knowledge_store.get_domain_patterns()
    if domain_patterns:
        print(f"\n📌 Learned patterns for {len(domain_patterns)} domains:")
        for domain, patterns in list(domain_patterns.items())[:5]:
            print(f"\n  🌐 {domain}:")
            for p in patterns[:3]:
                print(f"     - {p.get('type', 'N/A'):25s} "
                      f"(success: {p.get('success_rate', 0):.2f}, "
                      f"freq: {p.get('frequency', 0)})")
    else:
        print("⚠️  No domain patterns found in Neo4j")
    
    # Cleanup
    knowledge_store.close()
    
    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


async def test_sample_pattern_storage():
    """Store sample patterns to test classification"""
    
    print("\n🧪 Storing Sample Patterns for Testing")
    print("=" * 80)
    
    # Initialize
    config = Config()
    config.validate()
    
    gemini_client = GeminiClient(config.gemini)
    rl_controller = RLResourceController(gemini_client, config.training)
    knowledge_store = HybridKnowledgeStore(gemini_client, rl_controller, config.knowledge_store)
    
    # Sample patterns
    sample_patterns = [
        {
            "type": "product_list",
            "domain": "amazon.com",
            "extraction_fields": ["product_name", "price", "rating"],
            "success_rate": 0.95,
            "frequency": 10,
            "description": "Product extraction with prices and ratings",
            "metadata": {
                "user_prompt": "Get all products with prices",
                "items_extracted": 50
            }
        },
        {
            "type": "article_extraction",
            "domain": "nytimes.com",
            "extraction_fields": ["headline", "author", "published_date", "content"],
            "success_rate": 0.92,
            "frequency": 8,
            "description": "News article extraction",
            "metadata": {
                "user_prompt": "Extract news articles",
                "items_extracted": 20
            }
        },
        {
            "type": "review_extraction",
            "domain": "yelp.com",
            "extraction_fields": ["rating", "review_text", "author", "date"],
            "success_rate": 0.88,
            "frequency": 6,
            "description": "Customer reviews extraction",
            "metadata": {
                "user_prompt": "Get customer reviews",
                "items_extracted": 100
            }
        },
        {
            "type": "product_with_reviews",
            "domain": "shopee.vn",
            "extraction_fields": ["product_name", "price", "rating", "review_count"],
            "success_rate": 0.90,
            "frequency": 12,
            "description": "E-commerce product with reviews",
            "metadata": {
                "user_prompt": "Crawl products with ratings",
                "items_extracted": 75
            }
        },
        {
            "type": "contact_info",
            "domain": "yellowpages.com",
            "extraction_fields": ["business_name", "phone", "address", "email"],
            "success_rate": 0.85,
            "frequency": 5,
            "description": "Business contact information",
            "metadata": {
                "user_prompt": "Extract business contacts",
                "items_extracted": 30
            }
        }
    ]
    
    for i, pattern in enumerate(sample_patterns, 1):
        print(f"  {i}. Storing {pattern['type']} pattern for {pattern['domain']}...", end=" ")
        success = await knowledge_store.store_pattern(pattern)
        print("✅" if success else "❌")
    
    print("\n✅ Sample patterns stored!")
    knowledge_store.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test knowledge store pattern classification")
    parser.add_argument("--store-samples", action="store_true", 
                       help="Store sample patterns for testing")
    
    args = parser.parse_args()
    
    if args.store_samples:
        asyncio.run(test_sample_pattern_storage())
    
    asyncio.run(test_pattern_classification())
