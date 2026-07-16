# test_setup.py
# Run this to verify everything is working

import sys
import os

print("=" * 50)
print("LaborLens Setup Verification")
print("=" * 50)

# ── Test 1: Python Version ─────────────────────────
print("\n[1] Checking Python version...")
version = sys.version_info
if version.major == 3 and version.minor >= 10:
    print(f"    ✅ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"    ❌ Python {version.major}.{version.minor} - Need 3.10+")

# ── Test 2: Environment Variables ──────────────────
print("\n[2] Checking .env file...")
from dotenv import load_dotenv
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if groq_key and groq_key != "paste_your_groq_key_here":
    print(f"    ✅ GROQ_API_KEY found (starts with: {groq_key[:8]}...)")
else:
    print("    ❌ GROQ_API_KEY not set or still placeholder")

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
print(f"    ✅ QDRANT_HOST = {qdrant_host}")

# ── Test 3: Qdrant Connection ──────────────────────
print("\n[3] Checking Qdrant connection...")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )
    collections = client.get_collections()
    print(f"    ✅ Qdrant connected")
    print(f"    ✅ Collections: {len(collections.collections)}")
except Exception as e:
    print(f"    ❌ Qdrant failed: {e}")
    print("    Make sure Docker is running and you did: docker compose up -d qdrant")

# ── Test 4: Groq API ───────────────────────────────
# In test_setup.py
# Find this section and make sure it looks like this

print("\n[4] Checking Groq API...")
try:
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # uses model from .env file
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    response = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say: OK"}],
        max_tokens=10
    )
    answer = response.choices[0].message.content
    print(f"    ✅ Groq API working - model: {model}")
    print(f"    ✅ Response: {answer}")
except Exception as e:
    print(f"    ❌ Groq API failed: {e}")

# ── Test 5: Embedding Model ────────────────────────
print("\n[5] Checking Embedding Model...")
print("    Downloading model first time (130MB)...")
print("    This takes 2-3 minutes on first run...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    test_vec = model.encode(["test legal sentence"])
    print(f"    ✅ Embedding model loaded")
    print(f"    ✅ Vector size: {test_vec.shape[1]} dimensions")
except Exception as e:
    print(f"    ❌ Embedding model failed: {e}")

# ── Test 6: Key Libraries ──────────────────────────
print("\n[6] Checking key libraries...")
libraries = [
    ("scrapy", "Scrapy"),
    ("bs4", "BeautifulSoup4"),
    ("pdfplumber", "pdfplumber"),
    ("fitz", "PyMuPDF"),
    ("fastapi", "FastAPI"),
    ("streamlit", "Streamlit"),
    ("langchain", "LangChain"),
    ("sqlalchemy", "SQLAlchemy"),
    ("loguru", "Loguru"),
    ("ftfy", "ftfy"),
    ("pandas", "pandas"),
    ("pydantic", "Pydantic"),
]

all_good = True
for module, name in libraries:
    try:
        __import__(module)
        print(f"    ✅ {name}")
    except ImportError:
        print(f"    ❌ {name} - run: pip install -r requirements.txt")
        all_good = False

# ── Final Result ───────────────────────────────────
print("\n" + "=" * 50)
if all_good:
    print("🎉 ALL CHECKS PASSED - Ready to build!")
else:
    print("⚠️  Some checks failed - Fix errors above")
print("=" * 50)