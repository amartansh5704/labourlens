# test_api.py
# Tests all FastAPI endpoints
# Uses TestClient so no server needs to be running

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

print("=" * 60)
print("LaborLens - FastAPI Endpoint Tests")
print("=" * 60)

# ── Test 1: Root Endpoint ──────────────────────────────
print("\n[1] Testing root endpoint...")
try:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "LaborLens"
    assert "endpoints" in data
    print(f"    ✅ GET / → {response.status_code}")
    print(f"    ✅ App name: {data['name']}")
    print(f"    ✅ Endpoints listed: {len(data['endpoints'])}")
except Exception as e:
    print(f"    ❌ Root failed: {e}")

# ── Test 2: Health Check ───────────────────────────────
print("\n[2] Testing health endpoint...")
try:
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant" in data
    assert "database" in data
    assert "groq" in data
    print(f"    ✅ GET /api/health → {response.status_code}")
    print(f"    ✅ Overall status: {data['status']}")
    print(f"    ✅ Database:       {data['database']}")
    print(f"    ✅ Qdrant:         {data['qdrant']}")
    print(f"    ✅ Groq:           {data['groq']}")
except Exception as e:
    print(f"    ❌ Health check failed: {e}")

# ── Test 3: Jurisdictions Endpoint ────────────────────
print("\n[3] Testing jurisdictions endpoint...")
try:
    response = client.get("/api/jurisdictions")
    assert response.status_code == 200
    data = response.json()
    assert "jurisdictions" in data
    jurisdictions = data["jurisdictions"]
    assert "Delhi" in jurisdictions
    assert "Central" in jurisdictions
    assert "Maharashtra" in jurisdictions
    print(f"    ✅ GET /api/jurisdictions → {response.status_code}")
    print(f"    ✅ Jurisdictions: {jurisdictions}")
except Exception as e:
    print(f"    ❌ Jurisdictions failed: {e}")

# ── Test 4: Topics Endpoint ───────────────────────────
print("\n[4] Testing topics endpoint...")
try:
    response = client.get("/api/topics")
    assert response.status_code == 200
    data = response.json()
    assert "topics" in data
    assert "minimum_wage" in data["keys"]
    assert "working_hours" in data["keys"]
    print(f"    ✅ GET /api/topics → {response.status_code}")
    print(f"    ✅ Topics: {data['keys']}")
except Exception as e:
    print(f"    ❌ Topics failed: {e}")

# ── Test 5: Stats Endpoint ────────────────────────────
print("\n[5] Testing stats endpoint...")
try:
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "qdrant_vectors" in data
    assert "by_jurisdiction" in data
    print(f"    ✅ GET /api/stats → {response.status_code}")
    print(f"    ✅ Total documents: {data['total_documents']}")
    print(f"    ✅ Qdrant vectors:  {data['qdrant_vectors']}")
    print(f"    ✅ By jurisdiction: {data['by_jurisdiction']}")
except Exception as e:
    print(f"    ❌ Stats failed: {e}")

# ── Test 6: Documents Endpoint ────────────────────────
print("\n[6] Testing documents endpoint...")
try:
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert "page" in data
    print(f"    ✅ GET /api/documents → {response.status_code}")
    print(f"    ✅ Total documents: {data['total']}")
    print(f"    ✅ Page: {data['page']}")

    # test with filters
    response2 = client.get(
        "/api/documents?jurisdiction=Delhi&topic=minimum_wage"
    )
    assert response2.status_code == 200
    print(f"    ✅ GET /api/documents?jurisdiction=Delhi → {response2.status_code}")

except Exception as e:
    print(f"    ❌ Documents failed: {e}")

# ── Test 7: Ask Endpoint - Valid Question ─────────────
print("\n[7] Testing ask endpoint with valid question...")
try:
    response = client.post(
        "/api/ask",
        json={
            "question": "What is the minimum wage for unskilled workers in Delhi?",
            "jurisdiction": "Delhi",
            "topic": "minimum_wage",
            "top_k": 5
        }
    )
    assert response.status_code == 200
    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert "has_results" in data
    assert "disclaimer" in data

    print(f"    ✅ POST /api/ask → {response.status_code}")
    print(f"    ✅ has_results: {data['has_results']}")
    print(f"    ✅ Sources:     {len(data['sources'])}")
    print(f"    ✅ Disclaimer:  {bool(data['disclaimer'])}")
    print(f"    ✅ Answer preview: {data['answer'][:100].strip()}...")

    if data["sources"]:
        src = data["sources"][0]
        print(f"    ✅ Top source: {src['law_name']}")
        print(f"    ✅ Top score:  {src['score']}")

except Exception as e:
    print(f"    ❌ Ask endpoint failed: {e}")
    import traceback
    traceback.print_exc()

# ── Test 8: Ask Endpoint - No Filter ─────────────────
print("\n[8] Testing ask endpoint without filters...")
try:
    response = client.post(
        "/api/ask",
        json={
            "question": "What is the EPF contribution rate?",
        }
    )
    assert response.status_code == 200
    data = response.json()
    print(f"    ✅ POST /api/ask (no filter) → {response.status_code}")
    print(f"    ✅ has_results: {data['has_results']}")
    print(f"    ✅ Sources: {len(data['sources'])}")
except Exception as e:
    print(f"    ❌ Ask no-filter failed: {e}")

# ── Test 9: Ask Endpoint - Validation ─────────────────
print("\n[9] Testing ask endpoint validation...")
try:
    # empty question
    response = client.post(
        "/api/ask",
        json={"question": ""}
    )
    assert response.status_code == 422
    print(f"    ✅ Empty question rejected: {response.status_code}")

    # question too long
    response = client.post(
        "/api/ask",
        json={"question": "x" * 501}
    )
    assert response.status_code == 422
    print(f"    ✅ Too long question rejected: {response.status_code}")

    # invalid jurisdiction
    response = client.post(
        "/api/ask",
        json={
            "question": "What is minimum wage?",
            "jurisdiction": "InvalidState"
        }
    )
    assert response.status_code == 422
    print(f"    ✅ Invalid jurisdiction rejected: {response.status_code}")

except Exception as e:
    print(f"    ❌ Validation tests failed: {e}")

# ── Test 10: Compare Endpoint ─────────────────────────
print("\n[10] Testing compare endpoint...")
try:
    response = client.post(
        "/api/compare",
        json={
            "question": "What is the minimum wage?",
            "jurisdiction1": "Delhi",
            "jurisdiction2": "Maharashtra",
            "topic": "minimum_wage"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "comparison" in data
    assert "Delhi" in data["comparison"]
    assert "Maharashtra" in data["comparison"]
    assert "question" in data

    print(f"    ✅ POST /api/compare → {response.status_code}")
    print(f"    ✅ Delhi result:       has_results={data['comparison']['Delhi']['has_results']}")
    print(f"    ✅ Maharashtra result: has_results={data['comparison']['Maharashtra']['has_results']}")

    # test same jurisdiction error
    response2 = client.post(
        "/api/compare",
        json={
            "question": "test",
            "jurisdiction1": "Delhi",
            "jurisdiction2": "Delhi"
        }
    )
    assert response2.status_code == 400
    print(f"    ✅ Same jurisdiction rejected: {response2.status_code}")

except Exception as e:
    print(f"    ❌ Compare endpoint failed: {e}")
    import traceback
    traceback.print_exc()

# ── Final Result ───────────────────────────────────────
print("\n" + "=" * 60)
print("✅ FastAPI tests complete - Ready for Phase 5")
print("=" * 60)
print("\nNext: Phase 5 - Streamlit Frontend")
print("Start API manually with:")
print("  uvicorn api.main:app --reload --port 8000")
print("Then open: http://localhost:8000/docs")