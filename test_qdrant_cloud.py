# test_qdrant_cloud.py
# Paste your actual Qdrant Cloud details here to test

from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
load_dotenv()

# ─── PASTE YOUR DETAILS HERE ──────────────────────────
# Get these from cloud.qdrant.io dashboard
QDRANT_URL  = os.getenv("QDRANT_HOST", "")
QDRANT_KEY  = os.getenv("QDRANT_API_KEY", "")
# ──────────────────────────────────────────────────────

print("=" * 55)
print("Qdrant Cloud Connection Test")
print("=" * 55)
print(f"URL from .env:     {QDRANT_URL}")
print(f"API Key (first 8): {QDRANT_KEY[:8]}...")
print()

# try different URL formats
formats_to_try = []

if QDRANT_URL:
    clean = QDRANT_URL.replace("https://", "").replace("http://", "").rstrip("/")
    formats_to_try = [
        f"https://{clean}",         # most common
        f"https://{clean}:6333",    # with port
        f"https://{clean}:443",     # HTTPS port
    ]

for url_format in formats_to_try:
    print(f"Trying: {url_format}")
    try:
        client = QdrantClient(
            url=url_format,
            api_key=QDRANT_KEY,
            timeout=10,
            check_compatibility=False,
        )
        collections = client.get_collections()
        print(f"✅ SUCCESS with: {url_format}")
        print(f"   Collections: {[c.name for c in collections.collections]}")
        print()
        print(f"Add to .env:")
        print(f"QDRANT_HOST={clean}")
        print(f"QDRANT_PORT=6333")
        break
    except Exception as e:
        print(f"❌ Failed: {str(e)[:80]}")
        print()