"""
Data Pattern Detector - Analyze Product Data Structure

Analyzes first N products to detect:
- Field patterns (what fields exist, types, formats)
- Price formats (various currency symbols, separators)
- Data consistency scores
- Missing field rates

Output: DataProfile object with schema + examples for LLM context
This helps LLM generate more accurate parsing code.

Example:
    profiler = DataPatternDetector()
    profile = profiler.analyze(products[:50])
    
    # profile contains:
    # - field_schemas: {"price": {"type": "string", "format_examples": ["$100", "100₫"]}}
    # - consistency_score: 0.85
    # - missing_rates: {"brand": 0.15}
"""
import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FieldSchema:
    """Schema for a single field"""
    field_name: str
    types_found: Set[str] = field(default_factory=set)  # 'str', 'int', 'float', 'dict', 'list'
    format_examples: List[str] = field(default_factory=list)  # Sample values
    missing_count: int = 0
    total_count: int = 0
    
    @property
    def missing_rate(self) -> float:
        """Percentage of products missing this field"""
        return self.missing_count / self.total_count if self.total_count > 0 else 0.0
    
    @property
    def primary_type(self) -> str:
        """Most common type for this field"""
        if not self.types_found:
            return "unknown"
        return sorted(self.types_found)[0]  # Alphabetically first for determinism
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return {
            "field_name": self.field_name,
            "types": list(self.types_found),
            "primary_type": self.primary_type,
            "format_examples": self.format_examples[:5],  # Limit to 5 examples
            "missing_rate": round(self.missing_rate, 3)
        }


@dataclass
class DataProfile:
    """
    Complete data profile for products.
    
    Contains:
    - field_schemas: Dict[field_name, FieldSchema]
    - consistency_score: 0.0-1.0 (1.0 = all products have same fields)
    - price_formats: Detected price format patterns
    - total_products_analyzed: Number of products analyzed
    - recommendations: List of suggestions for LLM code generation
    """
    field_schemas: Dict[str, FieldSchema] = field(default_factory=dict)
    consistency_score: float = 0.0
    price_formats: List[str] = field(default_factory=list)
    total_products_analyzed: int = 0
    recommendations: List[str] = field(default_factory=list)
    
    def get_field(self, field_name: str) -> Optional[FieldSchema]:
        """Get schema for a specific field"""
        return self.field_schemas.get(field_name)
    
    def has_consistent_prices(self) -> bool:
        """Check if price data is consistent enough for calculations"""
        price_fields = ['price', 'price_vnd', 'price_usd', 'salePrice', 'gia', 'giá']
        
        for field_name in price_fields:
            schema = self.get_field(field_name)
            if schema and schema.missing_rate < 0.3:  # Less than 30% missing
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return {
            "total_products_analyzed": self.total_products_analyzed,
            "consistency_score": round(self.consistency_score, 3),
            "field_schemas": {
                name: schema.to_dict() 
                for name, schema in self.field_schemas.items()
            },
            "price_formats": self.price_formats,
            "recommendations": self.recommendations
        }
    
    def to_llm_context(self) -> str:
        """
        Convert to human-readable context for LLM prompt.
        
        Returns:
            Formatted string describing data structure
        """
        lines = []
        lines.append(f"**DATA PROFILE ({self.total_products_analyzed} products analyzed)**")
        lines.append(f"Consistency Score: {self.consistency_score:.2%}")
        lines.append("")
        
        lines.append("**FIELDS DETECTED:**")
        for field_name, schema in sorted(self.field_schemas.items()):
            missing_pct = schema.missing_rate * 100
            lines.append(f"- `{field_name}`: {schema.primary_type} (missing: {missing_pct:.1f}%)")
            if schema.format_examples:
                examples_str = ", ".join(f'"{ex}"' for ex in schema.format_examples[:3])
                lines.append(f"  Examples: {examples_str}")
        
        lines.append("")
        
        if self.price_formats:
            lines.append("**PRICE FORMATS DETECTED:**")
            for fmt in self.price_formats:
                lines.append(f"- {fmt}")
            lines.append("")
        
        if self.recommendations:
            lines.append("**RECOMMENDATIONS:**")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        return "\n".join(lines)


class DataPatternDetector:
    """
    Analyze product data to detect patterns and schemas.
    
    Usage:
        detector = DataPatternDetector(sample_size=50)
        profile = detector.analyze(products)
        print(profile.to_llm_context())
    """
    
    # Common price format patterns
    PRICE_PATTERNS = [
        (r'^\d+$', 'Plain number (e.g., "1000")'),
        (r'^\$\d+', 'Dollar prefix (e.g., "$100")'),
        (r'^\d+₫$', 'Vietnamese dong suffix (e.g., "100₫")'),
        (r'^\d+đ$', 'Vietnamese dong suffix lowercase (e.g., "100đ")'),
        (r'^\d{1,3}(,\d{3})+$', 'Comma-separated thousands (e.g., "1,000")'),
        (r'^\d{1,3}(\.\d{3})+$', 'Dot-separated thousands (e.g., "1.000")'),
        (r'^\d+\.\d{2}$', 'Decimal price (e.g., "99.99")'),
    ]
    
    def __init__(self, sample_size: int = 50):
        """
        Initialize detector.
        
        Args:
            sample_size: Number of products to analyze (default 50)
        """
        self.sample_size = sample_size
    
    def _get_type_name(self, value: Any) -> str:
        """Get simplified type name"""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "dict"
        elif value is None:
            return "null"
        else:
            return "other"
    
    def _detect_price_formats(self, price_samples: List[str]) -> List[str]:
        """
        Detect price format patterns from samples.
        
        Args:
            price_samples: List of price strings
            
        Returns:
            List of detected format descriptions
        """
        detected_formats = []
        
        for pattern, description in self.PRICE_PATTERNS:
            matches = sum(1 for sample in price_samples if re.match(pattern, str(sample)))
            if matches > 0:
                percentage = (matches / len(price_samples)) * 100
                detected_formats.append(f"{description} ({matches}/{len(price_samples)} = {percentage:.1f}%)")
        
        return detected_formats
    
    def analyze(self, products: List[Dict[str, Any]]) -> DataProfile:
        """
        Analyze products to generate data profile.
        
        Args:
            products: List of product dicts
            
        Returns:
            DataProfile with detected patterns
        """
        if not products:
            logger.warning("No products to analyze")
            return DataProfile(total_products_analyzed=0)
        
        # Limit to sample size
        sample = products[:self.sample_size]
        total_analyzed = len(sample)
        
        logger.info(f"Analyzing {total_analyzed} products for data patterns...")
        
        # Collect field information
        field_data: Dict[str, FieldSchema] = {}
        all_fields_per_product: List[Set[str]] = []
        
        price_samples = []
        
        for product in sample:
            if not isinstance(product, dict):
                continue
            
            product_fields = set(product.keys())
            all_fields_per_product.append(product_fields)
            
            for field_name, value in product.items():
                # Initialize field schema if not exists
                if field_name not in field_data:
                    field_data[field_name] = FieldSchema(
                        field_name=field_name,
                        total_count=total_analyzed
                    )
                
                schema = field_data[field_name]
                
                # Record type
                type_name = self._get_type_name(value)
                schema.types_found.add(type_name)
                
                # Collect value examples (limit to 5 unique)
                if value is not None and len(schema.format_examples) < 5:
                    value_str = str(value)
                    if value_str not in schema.format_examples:
                        schema.format_examples.append(value_str)
                
                # Collect price samples
                if field_name in ['price', 'price_vnd', 'price_usd', 'salePrice', 'gia', 'giá']:
                    if value is not None:
                        price_samples.append(str(value))
        
        # Calculate missing counts
        for field_name, schema in field_data.items():
            schema.missing_count = sum(
                1 for product in sample 
                if isinstance(product, dict) and field_name not in product
            )
        
        # Calculate consistency score
        # Consistency = average similarity of fields across products
        if all_fields_per_product:
            # Find most common field set
            all_fields = set()
            for fields in all_fields_per_product:
                all_fields.update(fields)
            
            # Calculate average field presence
            field_presence_sum = 0
            for field in all_fields:
                presence_count = sum(1 for product_fields in all_fields_per_product if field in product_fields)
                field_presence_sum += presence_count / total_analyzed
            
            consistency_score = field_presence_sum / len(all_fields) if all_fields else 0.0
        else:
            consistency_score = 0.0
        
        # Detect price formats
        price_formats = self._detect_price_formats(price_samples) if price_samples else []
        
        # Generate recommendations
        recommendations = []
        
        # Check if prices are consistent
        price_field_schemas = [
            schema for name, schema in field_data.items() 
            if name in ['price', 'price_vnd', 'price_usd', 'salePrice', 'gia', 'giá']
        ]
        
        if price_field_schemas:
            min_missing = min(schema.missing_rate for schema in price_field_schemas)
            if min_missing > 0.5:
                recommendations.append("⚠️ >50% of products lack valid price data. Consider data validation.")
            elif min_missing > 0.3:
                recommendations.append("⚠️ 30-50% of products missing price. Handle missing values carefully.")
            
            # Check for multiple price formats
            if len(price_formats) > 2:
                recommendations.append(f"⚠️ {len(price_formats)} different price formats detected. Need robust parsing.")
        else:
            recommendations.append("❌ No price field detected. Cannot perform price-based calculations.")
        
        # Check for brand field
        brand_schemas = [
            schema for name, schema in field_data.items()
            if name in ['brand', 'Brand', 'thuong_hieu', 'Thương hiệu', 'manufacturer']
        ]
        
        if brand_schemas:
            min_brand_missing = min(schema.missing_rate for schema in brand_schemas)
            if min_brand_missing > 0.3:
                recommendations.append(f"⚠️ {min_brand_missing*100:.1f}% products missing brand data.")
        else:
            recommendations.append("⚠️ No brand field detected.")
        
        # Check consistency
        if consistency_score < 0.5:
            recommendations.append("⚠️ Low data consistency - products have very different field structures.")
        elif consistency_score > 0.8:
            recommendations.append("✅ High data consistency - uniform product structure.")
        
        # Build profile
        profile = DataProfile(
            field_schemas=field_data,
            consistency_score=consistency_score,
            price_formats=price_formats,
            total_products_analyzed=total_analyzed,
            recommendations=recommendations
        )
        
        logger.info(f"Data profile: {len(field_data)} fields, consistency={consistency_score:.2%}")
        
        return profile


# Singleton instance
_detector: Optional[DataPatternDetector] = None

def get_detector(sample_size: int = 50) -> DataPatternDetector:
    """Get or create singleton detector instance"""
    global _detector
    if _detector is None:
        _detector = DataPatternDetector(sample_size=sample_size)
    return _detector
