#!/bin/bash
# Start the FastAPI backend in the background
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit frontend
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0