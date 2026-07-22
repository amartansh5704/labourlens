#!/bin/bash

echo "==============================="
echo "Starting LaborLens"
echo "PORT from Railway: $PORT"
echo "==============================="

# Start FastAPI on port 8000 in background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

echo "FastAPI started on port 8000 (PID: $API_PID)"

# Wait for API to be ready
sleep 5

# Start Streamlit on Railway's PORT
# This is the port Railway exposes to internet
exec streamlit run frontend/app.py \
  --server.port ${PORT:-8501} \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.enableCORS false \
  --server.enableXsrfProtection false