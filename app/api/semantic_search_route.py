"""
API route for semantic search with Claude LLM integration.

This endpoint:
1. Takes a natural language query
2. Converts it to an embedding
3. Performs vector similarity search in MongoDB
4. Retrieves relevant order context
5. Sends query + context to Claude
6. Returns the LLM response
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

from app.mongo.mongo_client import get_orders_collection
from bson import ObjectId

load_dotenv()

router = APIRouter(prefix="/semantic-search", tags=["semantic-search"])

# Embedding model (must match the one used in generate_embeddings.py)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Try to import required libraries
try:
    from sentence_transformers import SentenceTransformer
    import anthropic
except ImportError as e:
    print(f"Warning: Required libraries not installed: {e}")
    print("Please install: pip install sentence-transformers anthropic")
    SentenceTransformer = None
    anthropic = None

# Initialize embedding model (lazy loading)
_embedding_model = None


def get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None and SentenceTransformer:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def vector_search(
    query_embedding: List[float],
    collection,
    limit: int = 5,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Perform vector similarity search in MongoDB.
    
    Args:
        query_embedding: The query embedding vector
        collection: MongoDB collection
        limit: Maximum number of results to return
        min_score: Minimum similarity score threshold
    
    Returns:
        List of matching documents with similarity scores
    """
    # Fetch all documents with embeddings
    # Note: For large collections, you might want to use MongoDB Atlas Vector Search
    # For now, we'll do in-memory similarity search
    all_docs = list(collection.find({"embedding": {"$exists": True}}))
    
    if not all_docs:
        return []
    
    # Calculate similarities
    results = []
    for doc in all_docs:
        if "embedding" not in doc:
            continue
        
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        if similarity >= min_score:
            results.append({
                "document": doc,
                "score": similarity
            })
    
    # Sort by similarity score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top results
    return results[:limit]


def format_order_context(orders: List[Dict[str, Any]]) -> str:
    """
    Format order documents into a readable context string for the LLM.
    
    Args:
        orders: List of order documents with scores
    
    Returns:
        Formatted context string
    """
    if not orders:
        return "No relevant orders found."
    
    context_parts = []
    for i, result in enumerate(orders, 1):
        order = result["document"]
        score = result["score"]
        
        # Extract key information
        store_name = order.get("store", {}).get("name", "Unknown Restaurant")
        store_city = order.get("store", {}).get("address", {}).get("city", "")
        products = order.get("products", [])
        product_names = [p.get("name", "") for p in products if p.get("name")]
        total = order.get("price", {}).get("total", 0)
        order_date = order.get("order_completed_at", "")
        shipping_city = order.get("shipping_address", {}).get("city", "")
        
        context_parts.append(
            f"Order {i} (Relevance: {score:.2f}):\n"
            f"  Restaurant: {store_name}"
            + (f" in {store_city}" if store_city else "") + "\n"
            + (f"  Items: {', '.join(product_names)}" if product_names else "  Items: None") + "\n"
            + (f"  Total: ${total:.2f}" if total else "") + "\n"
            + (f"  Delivery Location: {shipping_city}" if shipping_city else "") + "\n"
            + (f"  Date: {order_date}" if order_date else "") + "\n"
        )
    
    return "\n".join(context_parts)


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str
    max_results: Optional[int] = 5
    min_similarity: Optional[float] = 0.0
    model: Optional[str] = "claude-3-sonnet-20240229"  # Default Claude model (more widely available)
    temperature: Optional[float] = 0.7


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search."""
    answer: str
    relevant_orders: List[Dict[str, Any]]
    query_embedding_dim: int
    num_results: int


@router.post("/query", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Perform semantic search and get LLM response with context.
    
    Example queries:
    - "What restaurants do people order from most in Princeton?"
    - "Show me orders with pizza"
    - "What are the most popular items ordered?"
    - "Find orders delivered to downtown areas"
    """
    if not SentenceTransformer or not anthropic:
        raise HTTPException(
            status_code=500,
            detail="Required libraries not installed. Please install sentence-transformers and anthropic."
        )
    
    # Check Claude API key
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    if not claude_api_key:
        raise HTTPException(
            status_code=500,
            detail="CLAUDE_API_KEY environment variable not set"
        )
    
    try:
        # Step 1: Generate query embedding
        model = get_embedding_model()
        if not model:
            raise HTTPException(status_code=500, detail="Failed to load embedding model")
        
        query_embedding = model.encode(request.query).tolist()
        
        # Step 2: Perform vector search
        collection = get_orders_collection()
        search_results = vector_search(
            query_embedding,
            collection,
            limit=request.max_results,
            min_score=request.min_similarity
        )
        
        if not search_results:
            return SemanticSearchResponse(
                answer="I couldn't find any relevant orders matching your query.",
                relevant_orders=[],
                query_embedding_dim=len(query_embedding),
                num_results=0
            )
        
        # Step 3: Format context
        context = format_order_context(search_results)
        
        # Step 4: Create prompt for Claude
        system_prompt = """You are a helpful assistant that analyzes restaurant order data. 
        Based on the provided order context, answer the user's question accurately and concisely.
        If the context doesn't contain enough information to fully answer the question, say so.
        Focus on the most relevant information from the orders provided."""
        
        user_prompt = f"""User Question: {request.query}

Relevant Orders Found:
{context}

Please answer the user's question based on the order data above. Be specific and reference the relevant orders when possible."""
        
        # Step 5: Call Claude API
        client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Try the requested model, with fallbacks if it fails
        models_to_try = [
            request.model,
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229"
        ]
        
        answer = None
        last_error = None
        
        for model_name in models_to_try:
            try:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=500,
                    temperature=request.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                answer = response.content[0].text
                break  # Success, exit the loop
            except Exception as e:
                last_error = e
                # If it's not a model not found error, raise it immediately
                if "not_found_error" not in str(e).lower() and "404" not in str(e):
                    raise
                # Otherwise, try the next model
                continue
        
        if answer is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to call Claude API with any available model. Last error: {str(last_error)}. Please check your API key and model availability."
            )
        
        # Step 6: Format results (remove embeddings for response)
        relevant_orders = []
        for result in search_results:
            order = result["document"].copy()
            # Remove embedding from response (too large)
            if "embedding" in order:
                del order["embedding"]
            # Convert ObjectId to string
            if "_id" in order and isinstance(order["_id"], ObjectId):
                order["_id"] = str(order["_id"])
            # Add similarity score
            order["similarity_score"] = result["score"]
            relevant_orders.append(order)
        
        return SemanticSearchResponse(
            answer=answer,
            relevant_orders=relevant_orders,
            query_embedding_dim=len(query_embedding),
            num_results=len(relevant_orders)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check for semantic search endpoint."""
    model = get_embedding_model()
    has_claude_key = bool(os.getenv("CLAUDE_API_KEY"))
    
    return {
        "status": "healthy" if (model and has_claude_key) else "degraded",
        "embedding_model_loaded": model is not None,
        "claude_key_configured": has_claude_key,
        "embedding_model": EMBEDDING_MODEL
    }

