"""
Version Manager for Training Resources
View, compare, and manage training resource versions
"""
import json
import os
import glob
import re
from datetime import datetime
from typing import Dict, List, Optional


class TrainingVersionManager:
    def __init__(self, resources_dir: str = "./"):
        self.resources_dir = resources_dir
    
    def list_versions(self) -> List[Dict]:
        """List all available versions with metadata"""
        versions = []
        pattern = os.path.join(self.resources_dir, "training_resources_v*.json")
        
        for filepath in glob.glob(pattern):
            match = re.search(r'training_resources_v(\d+)\.json', filepath)
            if match:
                version = int(match.group(1))
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        versions.append({
                            'version': version,
                            'file': filepath,
                            'frozen_at': data.get('frozen_at'),
                            'domain_count': len(data.get('domain_patterns', {})),
                            'total_patterns': sum(len(p) for p in data.get('domain_patterns', {}).values()),
                            'total_cycles': data.get('total_cycles', 0),
                            'performance_metrics': data.get('performance_metrics', {}),
                            'previous_version': data.get('previous_version'),
                            'incremental_info': data.get('incremental_learning', {})
                        })
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
        
        return sorted(versions, key=lambda x: x['version'])
    
    def get_latest_version(self) -> Optional[Dict]:
        """Get the latest version"""
        versions = self.list_versions()
        return versions[-1] if versions else None
    
    def compare_versions(self, v1: int, v2: int) -> Dict:
        """Compare two versions"""
        file1 = os.path.join(self.resources_dir, f"training_resources_v{v1}.json")
        file2 = os.path.join(self.resources_dir, f"training_resources_v{v2}.json")
        
        with open(file1, 'r') as f:
            data1 = json.load(f)
        with open(file2, 'r') as f:
            data2 = json.load(f)
        
        domains1 = set(data1.get('domain_patterns', {}).keys())
        domains2 = set(data2.get('domain_patterns', {}).keys())
        
        return {
            'v1': v1,
            'v2': v2,
            'new_domains': list(domains2 - domains1),
            'removed_domains': list(domains1 - domains2),
            'common_domains': list(domains1 & domains2),
            'patterns_diff': {
                'v1_total': sum(len(p) for p in data1.get('domain_patterns', {}).values()),
                'v2_total': sum(len(p) for p in data2.get('domain_patterns', {}).values())
            },
            'cycles_diff': {
                'v1': data1.get('total_cycles', 0),
                'v2': data2.get('total_cycles', 0),
                'increase': data2.get('total_cycles', 0) - data1.get('total_cycles', 0)
            }
        }
    
    def get_version_details(self, version: int) -> Optional[Dict]:
        """Get detailed information about a specific version"""
        filepath = os.path.join(self.resources_dir, f"training_resources_v{version}.json")
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return {
            'version': version,
            'frozen_at': data.get('frozen_at'),
            'domains': list(data.get('domain_patterns', {}).keys()),
            'domain_details': {
                domain: {
                    'pattern_count': len(patterns),
                    'pattern_types': self._count_pattern_types(patterns)
                }
                for domain, patterns in data.get('domain_patterns', {}).items()
            },
            'performance_history': data.get('performance_history', []),
            'total_cycles': data.get('total_cycles', 0),
            'performance_metrics': data.get('performance_metrics', {}),
            'incremental_learning': data.get('incremental_learning', {})
        }
    
    def _count_pattern_types(self, patterns: List) -> Dict:
        """Count different types of patterns"""
        types = {}
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            types[pattern_type] = types.get(pattern_type, 0) + 1
        return types
    
    def print_summary(self):
        """Print a summary of all versions"""
        versions = self.list_versions()
        
        if not versions:
            print("📂 No training versions found")
            return
        
        print("\n" + "="*80)
        print("📊 TRAINING RESOURCES VERSION SUMMARY")
        print("="*80)
        
        for v in versions:
            print(f"\n📦 Version {v['version']}")
            print(f"   Frozen at: {v['frozen_at']}")
            print(f"   Domains: {v['domain_count']}")
            print(f"   Total patterns: {v['total_patterns']}")
            print(f"   Training cycles: {v['total_cycles']}")
            
            if v.get('previous_version'):
                print(f"   📈 Built upon: v{v['previous_version']}")
            
            if v.get('incremental_info'):
                info = v['incremental_info']
                print(f"   🔄 Incremental learning:")
                print(f"      - New domains: {info.get('new_domains_added', 0)}")
                print(f"      - New patterns: {info.get('new_patterns_count', 0)}")
            
            if v.get('performance_metrics'):
                metrics = v['performance_metrics']
                print(f"   📈 Performance:")
                for key, value in metrics.items():
                    print(f"      - {key}: {value}")
        
        # Latest version info
        latest = versions[-1]
        print(f"\n{'='*80}")
        print(f"🏆 LATEST VERSION: v{latest['version']}")
        print(f"   Production should use: frozen_resources/latest.json")
        print(f"   Or specifically: training_resources_v{latest['version']}.json")
        print(f"{'='*80}\n")


def main():
    """CLI interface"""
    import sys
    
    manager = TrainingVersionManager()
    
    if len(sys.argv) < 2:
        manager.print_summary()
        return
    
    command = sys.argv[1]
    
    if command == "list":
        manager.print_summary()
    
    elif command == "latest":
        latest = manager.get_latest_version()
        if latest:
            print(json.dumps(latest, indent=2))
        else:
            print("No versions found")
    
    elif command == "details":
        if len(sys.argv) < 3:
            print("Usage: version_manager.py details <version>")
            return
        version = int(sys.argv[2])
        details = manager.get_version_details(version)
        if details:
            print(json.dumps(details, indent=2))
        else:
            print(f"Version {version} not found")
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("Usage: version_manager.py compare <v1> <v2>")
            return
        v1 = int(sys.argv[2])
        v2 = int(sys.argv[3])
        comparison = manager.compare_versions(v1, v2)
        print(json.dumps(comparison, indent=2))
    
    else:
        print(f"Unknown command: {command}")
        print("Available commands: list, latest, details, compare")


if __name__ == "__main__":
    main()
