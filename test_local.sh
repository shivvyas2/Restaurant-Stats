#!/bin/bash

# Test script for local FastAPI server
# Make sure the server is running on http://localhost:8000

BASE_URL="http://localhost:8000"

echo "🧪 Testing Restaurant Stats API..."
echo ""

echo "1. Testing root endpoint..."
curl -s "$BASE_URL/" | python3 -m json.tool
echo ""

echo "2. Testing health endpoint..."
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

echo "3. Testing mock-order endpoint..."
curl -s "$BASE_URL/mock-order" | python3 -m json.tool | head -20
echo ""

echo "4. Testing Knot session endpoint..."
curl -s -X POST "$BASE_URL/api/knot/session?external_user_id=test123" | python3 -m json.tool
echo ""

echo "✅ Tests complete!"


