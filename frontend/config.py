# frontend/config.py
import os

# On Railway: both run on same machine
# Streamlit calls FastAPI via localhost:8000
API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000/api"
)