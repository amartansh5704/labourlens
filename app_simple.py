# app_simple.py
# Simplest possible FastAPI to test Railway is working

from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "Railway is working",
        "port": os.getenv("PORT", "not set"),
        "message": "LaborLens API is alive"
    }

@app.get("/health")
def health():
    return {"status": "ok"}