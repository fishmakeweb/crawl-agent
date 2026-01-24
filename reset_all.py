#!/usr/bin/env python3
"""
Complete Reset Script for Self-Learning Agent
Clears all learned knowledge from all storage systems
"""

import os
import sys
import json
import shutil
from pathlib import Path

def clear_frozen_resources():
    """Clear frozen resources (learned patterns)"""
    print("\n1️⃣  Clearing frozen resources...")
    
    frozen_dir = Path(__file__).parent / "frozen_resources"
    
    if frozen_dir.exists():
        # Backup first
        backup_dir = Path(__file__).parent / "frozen_resources.backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(frozen_dir, backup_dir)
        print(f"   📦 Backup created: {backup_dir}")
        
        # Clear all JSON files
        for json_file in frozen_dir.glob("*.json"):
            # Reset to empty structure
            with open(json_file, 'w') as f:
                json.dump({
                    "domain_patterns": {},
                    "extraction_prompts": {},
                    "crawl_configs": {},
                    "version": "1.0.0",
                    "last_updated": None,
                    "total_patterns": 0
                }, f, indent=2)
            print(f"   ✓ Reset: {json_file.name}")
    else:
        print(f"   ⚠️  Frozen resources directory not found")
    
    print("   ✅ Frozen resources cleared")


def clear_qdrant():
    """Clear Qdrant vector database"""
    print("\n2️⃣  Clearing Qdrant vector store...")
    
    try:
        from qdrant_client import QdrantClient
        
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        
        # Delete and recreate collection
        collection_name = "crawler_patterns"
        
        try:
            client.delete_collection(collection_name)
            print(f"   ✓ Deleted collection: {collection_name}")
        except Exception as e:
            print(f"   ⚠️  Collection might not exist: {e}")
        
        # Recreate empty collection
        from qdrant_client.models import Distance, VectorParams
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"   ✓ Recreated empty collection: {collection_name}")
        print("   ✅ Qdrant cleared")
        
    except ImportError:
        print("   ⚠️  qdrant-client not installed, skipping...")
    except Exception as e:
        print(f"   ❌ Error clearing Qdrant: {e}")


def clear_neo4j():
    """Clear Neo4j graph database"""
    print("\n3️⃣  Clearing Neo4j graph database...")
    
    try:
        from neo4j import GraphDatabase
        
        # Connect to Neo4j
        uri = "bolt://localhost:7687"
        auth = ("neo4j", os.getenv("NEO4J_PASSWORD", "your_password"))
        
        driver = GraphDatabase.driver(uri, auth=auth)
        
        with driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("   ✓ All nodes and relationships deleted")
        
        driver.close()
        print("   ✅ Neo4j cleared")
        
    except ImportError:
        print("   ⚠️  neo4j driver not installed, skipping...")
    except Exception as e:
        print(f"   ❌ Error clearing Neo4j: {e}")
        print(f"   💡 Make sure NEO4J_PASSWORD env var is set correctly")


def clear_redis():
    """Clear Redis cache"""
    print("\n4️⃣  Clearing Redis cache...")
    
    try:
        import redis
        
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Flush specific pattern keys
        pattern_keys = r.keys("pattern:*")
        domain_keys = r.keys("domain:*")
        training_keys = r.keys("training:*")
        
        all_keys = pattern_keys + domain_keys + training_keys
        
        if all_keys:
            r.delete(*all_keys)
            print(f"   ✓ Deleted {len(all_keys)} keys")
        else:
            print("   ℹ️  No matching keys found")
        
        print("   ✅ Redis cleared")
        
    except ImportError:
        print("   ⚠️  redis package not installed, skipping...")
    except Exception as e:
        print(f"   ❌ Error clearing Redis: {e}")


def verify_reset():
    """Verify all systems are cleared"""
    print("\n5️⃣  Verifying reset...")
    
    checks = []
    
    # Check frozen resources
    frozen_dir = Path(__file__).parent / "frozen_resources"
    if frozen_dir.exists():
        latest = frozen_dir / "latest.json"
        if latest.exists():
            with open(latest) as f:
                data = json.load(f)
                pattern_count = data.get("total_patterns", 0)
                checks.append(("Frozen resources", pattern_count == 0))
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        count = client.count("crawl_patterns")
        checks.append(("Qdrant", count.count == 0))
    except:
        checks.append(("Qdrant", None))
    
    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        uri = "bolt://localhost:7687"
        auth = ("neo4j", os.getenv("NEO4J_PASSWORD", "your_password"))
        driver = GraphDatabase.driver(uri, auth=auth)
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            checks.append(("Neo4j", count == 0))
        driver.close()
    except:
        checks.append(("Neo4j", None))
    
    # Print results
    print("\n   Verification Results:")
    for name, cleared in checks:
        if cleared is None:
            print(f"   ⚠️  {name}: Could not verify")
        elif cleared:
            print(f"   ✅ {name}: Empty")
        else:
            print(f"   ❌ {name}: Still has data")
    
    all_clear = all(c for c in checks if c[1] is not None)
    return all_clear


def main():
    print("=" * 60)
    print("🗑️  COMPLETE RESET - Self-Learning Agent")
    print("=" * 60)
    print("\nThis will clear ALL learned knowledge:")
    print("  • Frozen resources (learned patterns)")
    print("  • Qdrant vector database")
    print("  • Neo4j graph database")
    print("  • Redis cache")
    print("\n⚠️  WARNING: This action cannot be undone!")
    print("   (Backup will be created in frozen_resources.backup)")
    
    # Confirm
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response != "yes":
        print("❌ Reset cancelled")
        sys.exit(0)
    
    # Execute reset
    clear_frozen_resources()
    clear_qdrant()
    clear_neo4j()
    clear_redis()
    
    # Verify
    if verify_reset():
        print("\n" + "=" * 60)
        print("✅ RESET COMPLETE - All knowledge cleared!")
        print("=" * 60)
        print("\n📊 System is ready for fresh training")
        print("💡 Restart training agent: docker-compose restart agent-training")
    else:
        print("\n" + "=" * 60)
        print("⚠️  RESET COMPLETED WITH WARNINGS")
        print("=" * 60)
        print("\n💡 Some systems could not be fully verified")
        print("   Check the logs above for details")


if __name__ == "__main__":
    main()
