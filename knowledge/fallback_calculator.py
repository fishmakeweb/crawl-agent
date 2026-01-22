"""
Fallback Calculator - Conservative Regex-Based Calculations

Last resort when code generation fails after retry limit.
Handles common patterns with very conservative parsing:
- Count all products
- Count products matching simple criteria (price < X, brand = Y)
- Sum prices
- Average price
- Min/max price
- Top N by simple field

Trade-off:
- Low accuracy for complex queries
- High reliability for simple patterns
- No execution risk (pure regex)

Usage:
    calculator = FallbackCalculator()
    result = calculator.calculate(query, products)
    if result:
        print(result)  # Dict with numerical results
    else:
        # Fallback also failed → return error to user
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FallbackResult:
    """Result from fallback calculation"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    method: str = "unknown"  # Which fallback method was used
    
    def __str__(self):
        if self.success:
            return f"Fallback {self.method}: {self.result}"
        else:
            return f"Fallback failed ({self.method}): {self.error}"


class FallbackCalculator:
    """
    Conservative regex-based calculator for common query patterns.
    
    Supports:
    - Count all: "bao nhiêu sản phẩm", "how many products"
    - Count filtered: "bao nhiêu sản phẩm dưới 500k", "products under $100"
    - Sum prices: "tổng giá", "total price"
    - Average price: "trung bình giá", "average price"
    - Min/max price: "giá rẻ nhất", "most expensive"
    - Top N brands: "top 5 thương hiệu", "most popular brands"
    
    Limitations:
    - Cannot handle complex multi-step logic
    - Price parsing is conservative (may miss some formats)
    - Only works with simple field names (price, brand, name)
    """
    
    def __init__(self):
        """Initialize fallback calculator"""
        pass
    
    def _extract_price(self, product: Dict[str, Any]) -> Optional[float]:
        """
        Extract price from product dict.
        
        Tries multiple fields and formats:
        - Fields: price, price_vnd, price_usd, salePrice, giá
        - Formats: "100", "$100", "100₫", "100.000đ", "100,000"
        
        Returns:
            Price as float, or None if not found/parseable
        """
        # Try common price fields
        price_fields = ['price', 'price_vnd', 'price_usd', 'salePrice', 'gia', 'giá']
        
        for field in price_fields:
            if field in product:
                price_value = product[field]
                
                # Handle already numeric
                if isinstance(price_value, (int, float)):
                    return float(price_value)
                
                # Handle string
                if isinstance(price_value, str):
                    # Remove common currency symbols and separators
                    cleaned = re.sub(r'[₫$đ,.\s]', '', price_value)
                    
                    # Try to parse as number
                    try:
                        return float(cleaned)
                    except ValueError:
                        continue
        
        return None
    
    def _extract_brand(self, product: Dict[str, Any]) -> Optional[str]:
        """
        Extract brand from product dict.
        
        Returns:
            Brand name as string, or None if not found
        """
        brand_fields = ['brand', 'Brand', 'thuong_hieu', 'Thương hiệu', 'manufacturer']
        
        for field in brand_fields:
            if field in product and product[field]:
                brand = product[field]
                if isinstance(brand, str) and brand.strip():
                    return brand.strip()
        
        return None
    
    def _count_all(self, products: List[Dict[str, Any]]) -> FallbackResult:
        """Count all products"""
        count = len(products)
        return FallbackResult(
            success=True,
            result={"count": count, "total_products": count},
            method="count_all"
        )
    
    def _count_price_filter(
        self,
        products: List[Dict[str, Any]],
        operator: str,
        threshold: float
    ) -> FallbackResult:
        """
        Count products matching price filter.
        
        Args:
            operator: '<', '>', '<=', '>='
            threshold: Price threshold
        """
        valid_products = []
        invalid_count = 0
        
        for product in products:
            price = self._extract_price(product)
            if price is None:
                invalid_count += 1
                continue
            
            match = False
            if operator in ['<', 'dưới', 'duoi', 'under', 'below']:
                match = price < threshold
            elif operator in ['>', 'trên', 'tren', 'over', 'above']:
                match = price > threshold
            elif operator in ['<=']:
                match = price <= threshold
            elif operator in ['>=']:
                match = price >= threshold
            
            if match:
                valid_products.append(product)
        
        if invalid_count > len(products) * 0.5:
            return FallbackResult(
                success=False,
                error=f"Too many products ({invalid_count}/{len(products)}) lack valid price data",
                method="count_price_filter"
            )
        
        return FallbackResult(
            success=True,
            result={
                "count": len(valid_products),
                "threshold": threshold,
                "operator": operator,
                "invalid_count": invalid_count
            },
            method="count_price_filter"
        )
    
    def _sum_prices(self, products: List[Dict[str, Any]]) -> FallbackResult:
        """Sum all prices"""
        total = 0.0
        valid_count = 0
        invalid_count = 0
        
        for product in products:
            price = self._extract_price(product)
            if price is not None:
                total += price
                valid_count += 1
            else:
                invalid_count += 1
        
        if valid_count == 0:
            return FallbackResult(
                success=False,
                error="No valid prices found in any product",
                method="sum_prices"
            )
        
        if invalid_count > len(products) * 0.5:
            return FallbackResult(
                success=False,
                error=f"Too many products ({invalid_count}/{len(products)}) lack valid price data",
                method="sum_prices"
            )
        
        return FallbackResult(
            success=True,
            result={
                "sum": total,
                "count": valid_count,
                "invalid_count": invalid_count
            },
            method="sum_prices"
        )
    
    def _avg_price(self, products: List[Dict[str, Any]]) -> FallbackResult:
        """Calculate average price"""
        sum_result = self._sum_prices(products)
        
        if not sum_result.success:
            return FallbackResult(
                success=False,
                error=sum_result.error,
                method="avg_price"
            )
        
        total = sum_result.result["sum"]
        count = sum_result.result["count"]
        avg = total / count if count > 0 else 0
        
        return FallbackResult(
            success=True,
            result={
                "average": avg,
                "count": count,
                "sum": total,
                "invalid_count": sum_result.result["invalid_count"]
            },
            method="avg_price"
        )
    
    def _min_max_price(
        self,
        products: List[Dict[str, Any]],
        find_min: bool = True
    ) -> FallbackResult:
        """Find min or max price"""
        prices = []
        
        for product in products:
            price = self._extract_price(product)
            if price is not None:
                prices.append((price, product))
        
        if not prices:
            return FallbackResult(
                success=False,
                error="No valid prices found",
                method="min_max_price"
            )
        
        if find_min:
            result_price, result_product = min(prices, key=lambda x: x[0])
            method = "min_price"
        else:
            result_price, result_product = max(prices, key=lambda x: x[0])
            method = "max_price"
        
        return FallbackResult(
            success=True,
            result={
                "price": result_price,
                "product": result_product,
                "total_products_checked": len(products),
                "valid_prices_found": len(prices)
            },
            method=method
        )
    
    def _top_brands(
        self,
        products: List[Dict[str, Any]],
        top_n: int = 5
    ) -> FallbackResult:
        """Count top N brands"""
        brand_counts = {}
        no_brand_count = 0
        
        for product in products:
            brand = self._extract_brand(product)
            if brand:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            else:
                no_brand_count += 1
        
        if not brand_counts:
            return FallbackResult(
                success=False,
                error="No products have brand information",
                method="top_brands"
            )
        
        # Sort by count descending
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        top_brands = sorted_brands[:top_n]
        
        return FallbackResult(
            success=True,
            result={
                "top_brands": dict(top_brands),
                "total_brands": len(brand_counts),
                "no_brand_count": no_brand_count,
                "top_n": top_n
            },
            method="top_brands"
        )
    
    def calculate(
        self,
        query: str,
        products: List[Dict[str, Any]]
    ) -> FallbackResult:
        """
        Attempt conservative calculation based on query patterns.
        
        Args:
            query: User's question
            products: List of product dicts
            
        Returns:
            FallbackResult with success/error and result dict
        """
        query_lower = query.lower()
        
        logger.info(f"🔧 Fallback calculator attempting: {query}")
        
        # Pattern 1: Count all products
        if re.search(r'bao nhiêu|bao nhieu|how many|count|số lượng|so luong', query_lower):
            # Check if filtering by price
            price_match = re.search(r'(\d+)[k|m|tr|triệu|nghìn|nghin]?', query_lower)
            operator_match = re.search(r'[<>]|dưới|duoi|trên|tren|under|over|below|above', query_lower)
            
            if price_match and operator_match:
                # Extract threshold
                num_str = price_match.group(1)
                threshold = float(num_str)
                
                # Handle k/m suffixes
                if 'k' in query_lower or 'nghìn' in query_lower or 'nghin' in query_lower:
                    threshold *= 1000
                elif 'm' in query_lower or 'tr' in query_lower or 'triệu' in query_lower:
                    threshold *= 1000000
                
                # Determine operator
                if any(kw in query_lower for kw in ['dưới', 'duoi', 'under', 'below', '<']):
                    operator = '<'
                else:
                    operator = '>'
                
                return self._count_price_filter(products, operator, threshold)
            else:
                # Simple count all
                return self._count_all(products)
        
        # Pattern 2: Sum prices
        if re.search(r'tổng|tong|sum|total.*giá|gia|price', query_lower):
            return self._sum_prices(products)
        
        # Pattern 3: Average price
        if re.search(r'trung bình|trung binh|average|avg.*giá|gia|price', query_lower):
            return self._avg_price(products)
        
        # Pattern 4: Min/max price
        if re.search(r'rẻ nhất|re nhat|cheapest|lowest|thấp nhất|thap nhat|min', query_lower):
            return self._min_max_price(products, find_min=True)
        
        if re.search(r'đắt nhất|dat nhat|expensive|highest|cao nhất|cao nhat|max', query_lower):
            return self._min_max_price(products, find_min=False)
        
        # Pattern 5: Top brands
        top_match = re.search(r'top\s*(\d+)', query_lower)
        if top_match and any(kw in query_lower for kw in ['thương hiệu', 'thuong hieu', 'brand', 'nhãn hiệu']):
            top_n = int(top_match.group(1))
            return self._top_brands(products, top_n)
        
        # No pattern matched
        return FallbackResult(
            success=False,
            error="Query pattern not recognized by fallback calculator",
            method="unknown"
        )


# Singleton instance
_calculator: Optional[FallbackCalculator] = None

def get_calculator() -> FallbackCalculator:
    """Get or create singleton calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = FallbackCalculator()
    return _calculator
