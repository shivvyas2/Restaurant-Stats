"""
One-time script to generate vector embeddings for all MongoDB order records.

This script:
1. Connects to MongoDB
2. Fetches all orders from the collection
3. Creates text representations of each order
4. Generates vector embeddings using sentence-transformers
5. Updates each document with the embedding field

Usage:
    python generate_embeddings.py
"""

import os
import sys
from typing import List, Dict, Any
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    print("Please install it with: pip install sentence-transformers")
    sys.exit(1)

# MongoDB connection
uri = os.getenv("MONGO_URI")
if not uri:
    raise ValueError("MONGO_URI environment variable not set")

client = MongoClient(uri, server_api=ServerApi('1'))

# Database and collection names
DB_NAME = "restaurant_stats"
COLLECTION_NAME = "orders"

# Embedding model (using a lightweight, general-purpose model)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dimensional embeddings


def create_order_text_representation(order: Dict[str, Any]) -> str:
    """
    Create a text representation of an order for embedding.
    
    This combines relevant fields that would be useful for semantic search:
    - Store name and location
    - Products ordered
    - Order status
    - Shipping location
    """
    parts = []
    
    # Store information
    store = order.get("store", {})
    if store.get("name"):
        parts.append(f"Restaurant: {store['name']}")
    
    store_addr = store.get("address", {})
    if store_addr.get("city"):
        parts.append(f"Restaurant location: {store_addr.get('city', '')}, {store_addr.get('region', '')}")
    
    # Products
    products = order.get("products", [])
    if products:
        product_names = [p.get("name", "") for p in products if p.get("name")]
        if product_names:
            parts.append(f"Items ordered: {', '.join(product_names)}")
    
    # Order status
    status = order.get("status", "")
    if status:
        parts.append(f"Status: {status}")
    
    # Shipping location
    shipping = order.get("shipping_address", {})
    if shipping.get("city"):
        parts.append(f"Delivery location: {shipping.get('city', '')}, {shipping.get('region', '')}")
    
    # Price information (optional, but can be useful)
    price = order.get("price", {})
    if price.get("total"):
        parts.append(f"Total: ${price['total']:.2f} {price.get('currency', 'USD')}")
    
    return ". ".join(parts) if parts else "Order"


def get_all_orders(collection: Collection) -> List[Dict[str, Any]]:
    """Fetch all orders from the collection."""
    print("Fetching all orders from MongoDB...")
    orders = list(collection.find({}))
    print(f"Found {len(orders)} orders")
    return orders


def generate_embeddings(orders: List[Dict[str, Any]], model: SentenceTransformer) -> List[List[float]]:
    """
    Generate embeddings for all orders.
    
    Args:
        orders: List of order documents
        model: SentenceTransformer model instance
    
    Returns:
        List of embedding vectors
    """
    print("Creating text representations...")
    texts = [create_order_text_representation(order) for order in orders]
    
    print(f"Generating embeddings using model: {EMBEDDING_MODEL}...")
    print("This may take a few minutes depending on the number of orders...")
    
    # Generate embeddings in batches for efficiency
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings.tolist())
    
    return all_embeddings


def update_orders_with_embeddings(
    collection: Collection,
    orders: List[Dict[str, Any]],
    embeddings: List[List[float]]
) -> int:
    """
    Update each order document with its embedding.
    
    Args:
        collection: MongoDB collection
        orders: List of order documents
        embeddings: List of embedding vectors
    
    Returns:
        Number of successfully updated documents
    """
    print("\nUpdating documents with embeddings...")
    updated_count = 0
    
    for order, embedding in tqdm(zip(orders, embeddings), total=len(orders), desc="Updating documents"):
        try:
            # Update the document with the embedding
            result = collection.update_one(
                {"_id": order["_id"]},
                {"$set": {"embedding": embedding}}
            )
            if result.modified_count > 0:
                updated_count += 1
        except Exception as e:
            print(f"\nWarning: Failed to update order {order.get('order_key', 'unknown')}: {e}")
    
    return updated_count


def create_vector_index(collection: Collection):
    """Create a vector search index on the embedding field (if using MongoDB Atlas vector search)."""
    print("\nNote: To enable vector search, you'll need to create a vector search index in MongoDB Atlas.")
    print("The embedding field has been added to all documents.")
    print("For MongoDB Atlas vector search, create an index with:")
    print("  - Field: 'embedding'")
    print("  - Type: 'vector'")
    print("  - Dimensions: 384 (for all-MiniLM-L6-v2 model)")


def main():
    """Main function to generate and store embeddings."""
    print("=" * 60)
    print("MongoDB Order Embeddings Generation Script")
    print("=" * 60)
    
    # Test MongoDB connection
    try:
        client.admin.command('ping')
        print("✓ Successfully connected to MongoDB!")
    except Exception as e:
        print(f"ERROR: Failed to connect to MongoDB: {e}")
        return
    
    # Get collection
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Check if embeddings already exist
    existing_count = collection.count_documents({"embedding": {"$exists": True}})
    if existing_count > 0:
        response = input(
            f"\nWarning: {existing_count} documents already have embeddings. "
            "Do you want to regenerate them? (yes/no): "
        )
        if response.lower() != "yes":
            print("Aborted.")
            return
    
    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    print("(This may take a moment on first run as it downloads the model)")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}")
        return
    
    # Fetch all orders
    orders = get_all_orders(collection)
    
    if not orders:
        print("No orders found in the collection.")
        return
    
    # Generate embeddings
    embeddings = generate_embeddings(orders, model)
    
    if len(embeddings) != len(orders):
        print(f"ERROR: Mismatch between orders ({len(orders)}) and embeddings ({len(embeddings)})")
        return
    
    # Update documents
    updated_count = update_orders_with_embeddings(collection, orders, embeddings)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total orders processed: {len(orders)}")
    print(f"Successfully updated: {updated_count}")
    print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
    print("=" * 60)
    
    # Note about vector index
    create_vector_index(collection)
    
    # Close connection
    client.close()
    print("\n✓ MongoDB connection closed")
    print("✓ Script completed successfully!")


if __name__ == "__main__":
    main()

