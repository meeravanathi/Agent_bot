"""
Mock database classes for the e-commerce chatbot
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import random

class MockOrderDatabase:
    """Mock order management system"""
    
    def __init__(self):
        self.orders = {
            "ORD001": {
                "order_id": "ORD001",
                "customer_id": "CUST001",
                "status": "shipped",
                "items": [
                    {"product_id": "PROD001", "name": "Wireless Headphones", "quantity": 1, "price": 99.99}
                ],
                "total": 99.99,
                "order_date": "2024-01-15",
                "shipping_address": "123 Main St, New York, NY",
                "tracking_number": "TRK123456789",
                "can_cancel": False
            },
            "ORD002": {
                "order_id": "ORD002",
                "customer_id": "CUST001",
                "status": "processing",
                "items": [
                    {"product_id": "PROD002", "name": "Smart Watch", "quantity": 1, "price": 249.99},
                    {"product_id": "PROD003", "name": "Phone Case", "quantity": 2, "price": 19.99}
                ],
                "total": 289.97,
                "order_date": "2024-01-20",
                "shipping_address": "123 Main St, New York, NY",
                "tracking_number": None,
                "can_cancel": True
            },
            "ORD003": {
                "order_id": "ORD003",
                "customer_id": "CUST002",
                "status": "delivered",
                "items": [
                    {"product_id": "PROD004", "name": "Laptop Stand", "quantity": 1, "price": 45.99}
                ],
                "total": 45.99,
                "order_date": "2024-01-10",
                "shipping_address": "456 Oak Ave, Los Angeles, CA",
                "tracking_number": "TRK987654321",
                "can_cancel": False
            },
            "ORD004": {
                "order_id": "ORD004",
                "customer_id": "CUST003",
                "status": "processing",
                "items": [
                    {"product_id": "PROD005", "name": "Winter Jacket", "quantity": 1, "price": 89.99}
                ],
                "total": 89.99,
                "order_date": "2024-01-22",
                "shipping_address": "789 Pine Rd, Chicago, IL",
                "tracking_number": None,
                "can_cancel": True
            },
            "ORD005": {
                "order_id": "ORD005",
                "customer_id": "CUST004",
                "status": "delivered",
                "items": [
                    {"product_id": "PROD006", "name": "Gaming Mouse", "quantity": 1, "price": 59.99}
                ],
                "total": 59.99,
                "order_date": "2024-01-18",
                "shipping_address": "321 Elm St, Houston, TX",
                "tracking_number": "TRK2468101214",
                "can_cancel": False
            },
            "ORD006": {
                "order_id": "ORD006",
                "customer_id": "CUST005",
                "status": "processing",
                "items": [
                    {"product_id": "PROD007", "name": "Bluetooth Speaker", "quantity": 2, "price": 34.99}
                ],
                "total": 69.98,
                "order_date": "2024-01-21",
                "shipping_address": "654 Maple Ln, Miami, Florida",
                "tracking_number": None,
                "can_cancel": True
            }
        }
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status by order ID"""
        return self.orders.get(order_id)
    
    def cancel_order(self, order_id: str) -> Dict[str, Union[bool, str]]:
        """Cancel an order if possible"""
        order = self.orders.get(order_id)
        if not order:
            return {"success": False, "message": "Order not found"}
        
        if not order["can_cancel"]:
            return {"success": False, "message": "Order cannot be cancelled (already shipped/delivered)"}
        
        order["status"] = "cancelled"
        order["can_cancel"] = False
        return {"success": True, "message": "Order cancelled successfully"}
    
    def process_return(self, order_id: str, reason: str = "") -> Dict[str, Union[bool, str]]:
        """Process a return request"""
        order = self.orders.get(order_id)
        if not order:
            return {"success": False, "message": "Order not found"}
        
        if order["status"] not in ["delivered"]:
            return {"success": False, "message": "Order must be delivered to process return"}
        
        # Mock return processing
        return {
            "success": True, 
            "message": f"Return request processed. Return ID: RET{random.randint(1000, 9999)}. Please ship items back within 30 days."
        }

class MockProductDatabase:
    """Mock product information system"""
    
    def __init__(self):
        self.products = {
            "PROD001": {
                "product_id": "PROD001",
                "name": "Wireless Headphones",
                "category": "Electronics",
                "price": 99.99,
                "availability": "in_stock",
                "stock_count": 25,
                "description": "High-quality wireless headphones with noise cancellation",
                "rating": 4.5,
                "features": ["Bluetooth 5.0", "30-hour battery", "Active noise cancellation"]
            },
            "PROD002": {
                "product_id": "PROD002",
                "name": "Smart Watch",
                "category": "Electronics",
                "price": 249.99,
                "availability": "in_stock",
                "stock_count": 12,
                "description": "Feature-rich smartwatch with health monitoring",
                "rating": 4.3,
                "features": ["Heart rate monitor", "GPS", "Water resistant", "7-day battery"]
            },
            "PROD003": {
                "product_id": "PROD003",
                "name": "Phone Case",
                "category": "Accessories",
                "price": 19.99,
                "availability": "in_stock",
                "stock_count": 100,
                "description": "Durable protective phone case",
                "rating": 4.1,
                "features": ["Drop protection", "Wireless charging compatible", "Clear design"]
            },
            "PROD004": {
                "product_id": "PROD004",
                "name": "Laptop Stand",
                "category": "Office",
                "price": 45.99,
                "availability": "low_stock",
                "stock_count": 3,
                "description": "Adjustable laptop stand for ergonomic working",
                "rating": 4.7,
                "features": ["Adjustable height", "Foldable", "Heat dissipation", "Universal compatibility"]
            },
            "PROD005": {
                "product_id": "PROD005",
                "name": "Winter Jacket",
                "category": "Clothing",
                "price": 89.99,
                "availability": "in_stock",
                "stock_count": 15,
                "description": "Warm and waterproof winter jacket",
                "rating": 4.4,
                "features": ["Waterproof", "Insulated", "Multiple pockets", "Wind resistant"]
            },
            "PROD006": {
                "product_id": "PROD006",
                "name": "Gaming Mouse",
                "category": "Electronics",
                "price": 59.99,
                "availability": "in_stock",
                "stock_count": 40,
                "description": "Ergonomic gaming mouse with customizable buttons",
                "rating": 4.6,
                "features": ["RGB lighting", "High precision sensor", "Wireless and wired modes"]
            },
            "PROD007": {
                "product_id": "PROD007",
                "name": "Bluetooth Speaker",
                "category": "Electronics",
                "price": 34.99,
                "availability": "in_stock",
                "stock_count": 50,
                "description": "Portable Bluetooth speaker with rich bass",
                "rating": 4.2,
                "features": ["Water resistant", "12-hour battery", "Compact design"]
            },
            "PROD008": {
                "product_id": "PROD008",
                "name": "Desk Lamp",
                "category": "Office",
                "price": 29.99,
                "availability": "in_stock",
                "stock_count": 20,
                "description": "LED desk lamp with adjustable brightness",
                "rating": 4.0,
                "features": ["Adjustable brightness", "Touch control", "Energy efficient"]
            },
            "PROD009": {
                "product_id": "PROD009",
                "name": "Running Shoes",
                "category": "Clothing",
                "price": 75.99,
                "availability": "in_stock",
                "stock_count": 30,
                "description": "Lightweight running shoes for everyday use",
                "rating": 4.3,
                "features": ["Breathable material", "Cushioned sole", "Durable outsole"]
            },
            "PROD010": {
                "product_id": "PROD010",
                "name": "Coffee Mug",
                "category": "Accessories",
                "price": 14.99,
                "availability": "in_stock",
                "stock_count": 60,
                "description": "Ceramic coffee mug with a sleek design",
                "rating": 4.5,
                "features": ["Microwave safe", "Dishwasher safe", "350ml capacity"]
            }
        }
    
    def search_products(self, query: str, category: str = None) -> List[Dict]:
        """Search products by name or category"""
        results = []
        query_lower = query.lower()
        
        for product in self.products.values():
            match = False
            
            # Check name match
            if query_lower in product["name"].lower():
                match = True
            
            # Check category match if specified
            if category and category.lower() != product["category"].lower():
                match = False
            
            if match:
                results.append(product)
        
        return results
    
    def get_product_details(self, product_id: str) -> Optional[Dict]:
        """Get detailed product information"""
        return self.products.get(product_id)
    
    def get_recommendations(self, category: str = None, weather_condition: str = None) -> List[Dict]:
        """Get product recommendations based on category or weather"""
        recommendations = []
        
        if weather_condition:
            if "cold" in weather_condition.lower() or "winter" in weather_condition.lower():
                # Recommend winter items
                recommendations = [p for p in self.products.values() if "winter" in p["name"].lower() or p["category"].lower() == "clothing"]
            elif "rain" in weather_condition.lower():
                # Recommend waterproof items
                recommendations = [p for p in self.products.values() if any("waterproof" in f.lower() for f in p.get("features", []))]
        
        if not recommendations and category:
            recommendations = [p for p in self.products.values() if p["category"].lower() == category.lower()]
        
        if not recommendations:
            # Default recommendations (top rated)
            recommendations = sorted(self.products.values(), key=lambda x: x["rating"], reverse=True)[:3]
        
        return recommendations

class MockCustomerDatabase:
    """Mock customer database"""
    
    def __init__(self):
        self.customers = {
            "CUST001": {
                "customer_id": "CUST001",
                "name": "John Doe",
                "email": "john.doe@email.com",
                "phone": "+1-555-0123",
                "address": "123 Main St, New York, NY",
                "loyalty_points": 1250,
                "tier": "Gold",
                "preferences": {
                    "categories": ["Electronics", "Books"],
                    "brands": ["TechBrand", "BookCorp"],
                    "communication": "email"
                },
                "order_history": ["ORD001", "ORD002"]
            },
            "CUST002": {
                "customer_id": "CUST002",
                "name": "Jane Smith",
                "email": "jane.smith@email.com",
                "phone": "+1-555-0456",
                "address": "456 Oak Ave, Los Angeles, CA",
                "loyalty_points": 750,
                "tier": "Silver",
                "preferences": {
                    "categories": ["Home", "Office"],
                    "brands": ["HomePlus", "OfficeMax"],
                    "communication": "sms"
                },
                "order_history": ["ORD003"]
            },
            "CUST003": {
                "customer_id": "CUST003",
                "name": "Alice Johnson",
                "email": "alice.johnson@email.com",
                "phone": "+1-555-0789",
                "address": "789 Pine Rd, Chicago, Illinois",
                "loyalty_points": 300,
                "tier": "Bronze",
                "preferences": {
                    "categories": ["Clothing", "Accessories"],
                    "brands": ["Fashionista", "Accents"],
                    "communication": "email"
                },
                "order_history": ["ORD004"]
            },
            "CUST004": {
                "customer_id": "CUST004",
                "name": "Bob Brown",
                "email": "bob.brown@email.com",
                "phone": "+1-555-0110",
                "address": "321 Elm St, Houston, Texas",
                "loyalty_points": 980,
                "tier": "Gold",
                "preferences": {
                    "categories": ["Electronics", "Gaming"],
                    "brands": ["GamePro", "TechBrand"],
                    "communication": "sms"
                },
                "order_history": ["ORD005"]
            },
            "CUST005": {
                "customer_id": "CUST005",
                "name": "Carol White",
                "email": "carol.white@email.com",
                "phone": "+1-555-0222",
                "address": "654 Maple Ln, Miami, Florida",
                "loyalty_points": 450,
                "tier": "Silver",
                "preferences": {
                    "categories": ["Electronics", "Home"],
                    "brands": ["HomePlus", "SoundMaster"],
                    "communication": "email"
                },
                "order_history": ["ORD006"]
            }
        }
    
    def get_customer_info(self, customer_id: str) -> Optional[Dict]:
        """Get customer information"""
        return self.customers.get(customer_id)
    
    def get_customer_by_email(self, email: str) -> Optional[Dict]:
        """Get customer by email"""
        for customer in self.customers.values():
            if customer["email"].lower() == email.lower():
                return customer
        return None
