#!/usr/bin/env python3
"""
Complete Reset Script for Self-Learning Agent (v2 - Docker Exec)
Clears all learned knowledge from all storage systems using docker exec
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path


def run_docker_command(container_name, python_code):
    """Execute Python code in a Docker container"""
    result = subprocess.run(
        ["docker", "exec", container_name, "python", "-c", python_code],
        capture_output=True,
        text=True
    )
    return result


def find_container(name_filter):
    """Find Docker container by name filter"""
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={name_filter}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True
    )
    container = result.stdout.strip().split('\n')[0] if result.stdout.strip() else None
    return container


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
    return True


def clear_qdrant():
    """Clear Qdrant vector database using docker exec"""
    print("\n2️⃣  Clearing Qdrant vector store...")
    
    try:
        # Find agent container that has qdrant-client
        container = find_container("agent-training")
        
        if not container:
            print("   ⚠️  Agent container not found")
            return False
        
        print(f"   📦 Using container: {container}")
        
        # Clear Qdrant using Python in agent container
        clear_script = """
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="qdrant", port=6333)

# Delete collection
try:
    client.delete_collection("crawl_patterns")
    print("✓ Deleted collection: crawl_patterns")
except Exception as e:
    print(f"⚠ Collection might not exist: {e}")

# Recreate empty collection
client.create_collection(
    collection_name="crawler_patterns",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
print("✓ Recreated empty collection: crawler_patterns")
"""
        
        result = run_docker_command(container, clear_script)
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"   {line}")
            print("   ✅ Qdrant cleared")
            return True
        else:
            print(f"   ❌ Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error clearing Qdrant: {e}")
        return False


def clear_neo4j():
    """Clear Neo4j graph database using docker exec"""
    print("\n3️⃣  Clearing Neo4j graph database...")
    
    try:
        # Find agent container that has neo4j driver
        container = find_container("agent-training")
        
        if not container:
            print("   ⚠️  Agent container not found")
            return False
        
        print(f"   📦 Using container: {container}")
        
        # Clear Neo4j using Python in agent container
        clear_script = """
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    "bolt://neo4j:7687",
    auth=("neo4j", "selflearning_neo4j_pass")
)

with driver.session() as session:
    # Delete all nodes and relationships
    session.run("MATCH (n) DETACH DELETE n")
    print("✓ Deleted all nodes and relationships")
    
    # Verify count
    result = session.run("MATCH (n) RETURN count(n) as count")
    count = result.single()["count"]
    print(f"✓ Remaining nodes: {count}")

driver.close()
"""
        
        result = run_docker_command(container, clear_script)
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"   {line}")
            print("   ✅ Neo4j cleared")
            return True
        else:
            print(f"   ❌ Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error clearing Neo4j: {e}")
        return False


def clear_redis():
    """Clear Redis cache using docker exec"""
    print("\n4️⃣  Clearing Redis cache...")
    
    try:
        # Find agent container that has redis package
        container = find_container("agent-training")
        
        if not container:
            print("   ⚠️  Agent container not found")
            return False
        
        print(f"   📦 Using container: {container}")
        
        # Clear Redis using Python in agent container
        clear_script = """
import redis

r = redis.Redis(host='redis-cache', port=6379, db=0, decode_responses=True)

# Delete pattern-related keys
patterns = ["pattern:*", "domain:*", "training:*", "cache:*"]
total_deleted = 0

for pattern in patterns:
    keys = r.keys(pattern)
    if keys:
        deleted = r.delete(*keys)
        total_deleted += deleted
        print(f"✓ Deleted {deleted} keys matching '{pattern}'")

if total_deleted == 0:
    print("ℹ No matching keys found")
else:
    print(f"✓ Total deleted: {total_deleted} keys")
"""
        
        result = run_docker_command(container, clear_script)
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"   {line}")
            print("   ✅ Redis cleared")
            return True
        else:
            print(f"   ❌ Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error clearing Redis: {e}")
        return False


def verify_reset():
    """Verify all systems are cleared"""
    print("\n5️⃣  Verifying reset...")
    
    results = {}
    
    # Check frozen resources
    try:
        frozen_dir = Path(__file__).parent / "frozen_resources"
        latest = frozen_dir / "latest.json"
        if latest.exists():
            with open(latest) as f:
                data = json.load(f)
                pattern_count = data.get("total_patterns", 0)
                results["frozen_resources"] = (pattern_count, pattern_count == 0)
    except Exception as e:
        results["frozen_resources"] = (None, None)
    
    # Check Qdrant using docker
    try:
        container = find_container("agent-training")
        if container:
            check_script = """
from qdrant_client import QdrantClient
client = QdrantClient(host="qdrant", port=6333)
count = client.count("crawl_patterns")
print(count.count)
"""
            result = run_docker_command(container, check_script)
            if result.returncode == 0:
                count = int(result.stdout.strip())
                results["qdrant"] = (count, count == 0)
            else:
                results["qdrant"] = (None, None)
        else:
            results["qdrant"] = (None, None)
    except Exception as e:
        results["qdrant"] = (None, None)
    
    # Check Neo4j using docker
    try:
        container = find_container("agent-training")
        if container:
            check_script = """
from neo4j import GraphDatabase
driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "selflearning_neo4j_pass"))
with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as count")
    count = result.single()["count"]
    print(count)
driver.close()
"""
            result = run_docker_command(container, check_script)
            if result.returncode == 0:
                count = int(result.stdout.strip())
                results["neo4j"] = (count, count == 0)
            else:
                results["neo4j"] = (None, None)
        else:
            results["neo4j"] = (None, None)
    except Exception as e:
        results["neo4j"] = (None, None)
    
    # Print results
    print("\n   Verification Results:")
    
    for system, (count, is_empty) in results.items():
        if count is None:
            print(f"   ⚠️  {system.replace('_', ' ').title()}: Could not verify")
        elif is_empty:
            print(f"   ✅ {system.replace('_', ' ').title()}: Empty (count: {count})")
        else:
            print(f"   ❌ {system.replace('_', ' ').title()}: Still has {count} items")
    
    # All must be empty (ignore None)
    all_clear = all(is_empty for count, is_empty in results.values() if is_empty is not None)
    return all_clear


def main():
    print("=" * 60)
    print("🗑️  COMPLETE RESET - Self-Learning Agent (v2)")
    print("=" * 60)
    print("\nThis will clear ALL learned knowledge:")
    print("  • Frozen resources (learned patterns)")
    print("  • Qdrant vector database")
    print("  • Neo4j graph database")
    print("  • Redis cache")
    print("\n⚠️  WARNING: This action cannot be undone!")
    print("   (Backup will be created in frozen_resources.backup)")
    print("\n💡 This version uses docker exec to access databases")
    
    # Confirm
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response != "yes":
        print("❌ Reset cancelled")
        sys.exit(0)
    
    # Execute reset
    success = True
    success &= clear_frozen_resources()
    success &= clear_qdrant()
    success &= clear_neo4j()
    success &= clear_redis()
    
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
