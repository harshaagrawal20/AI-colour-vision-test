from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import your existing backend FastAPI app
from backend.main import app as backend_app

# Create a root FastAPI app
app = FastAPI()

# Mount your backend at /api
app.mount("/api", backend_app)

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)
