#!/bin/bash

# Start FastAPI in background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit on the PORT Railway provides
streamlit run frontend/app.py \
  --server.port $PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false