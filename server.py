"""
FastAPI server for music mood analysis with WebUI.
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from main import MusicMoodAnalyzer
from data_models import SongAnalysis


# Global temp directory for uploads
UPLOAD_DIR = Path(tempfile.gettempdir()) / "music_analyzer_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    # Startup
    print("Starting Music Mood Analyzer API...")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Music Mood Analyzer API",
    description="API for analyzing music mood with optional LLM commentary",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve the main WebUI."""
    return FileResponse("index.html")


@app.post("/api/analyze")
async def analyze_music(
    audio_file: UploadFile = File(..., description="MP3/audio file to analyze"),
    lyrics_file: Optional[UploadFile] = File(None, description="Optional lyrics file"),
    use_llm: bool = Form(True, description="Enable LLM commentary"),
    use_separation: bool = Form(False, description="Enable source separation (slower but more detailed)"),
    llm_provider: str = Form("gemini", description="LLM provider: claude or gemini"),
    llm_model: Optional[str] = Form(None, description="LLM model to use"),
    api_key: Optional[str] = Form(None, description="API key for LLM provider")
):
    """
    Analyze a music file with optional lyrics and LLM commentary.

    Returns:
        SongAnalysis object with complete analysis
    """
    # Generate unique ID for this analysis
    analysis_id = str(uuid.uuid4())

    # Create temporary directory for this analysis
    temp_dir = UPLOAD_DIR / analysis_id
    temp_dir.mkdir(exist_ok=True)

    try:
        # Save uploaded audio file
        audio_path = temp_dir / audio_file.filename
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # Save lyrics file if provided
        lyrics_path = None
        if lyrics_file:
            lyrics_path = temp_dir / lyrics_file.filename
            with open(lyrics_path, "wb") as f:
                content = await lyrics_file.read()
                f.write(content)

        # Get API key from environment if not provided
        if use_llm and not api_key:
            if llm_provider == "gemini":
                api_key = os.getenv("OPENROUTER_API_KEY")
            elif llm_provider == "claude":
                api_key = os.getenv("ANTHROPIC_API_KEY")

        # Create analyzer
        analyzer = MusicMoodAnalyzer(
            audio_path=str(audio_path),
            lyrics_path=str(lyrics_path) if lyrics_path else None,
            use_llm=use_llm,
            use_separation=use_separation,
            llm_provider=llm_provider,
            llm_model=llm_model,
            api_key=api_key
        )

        # Run analysis
        analysis = analyzer.analyze()

        # Cleanup analyzer
        analyzer.cleanup()

        # Convert to dict for JSON response
        result = analysis.model_dump(exclude_none=False)

        return JSONResponse(content=result)

    except Exception as e:
        # Cleanup on error
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files after analysis
        # (Keep them for a bit in case we want to implement result caching)
        pass


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


@app.get("/api/providers")
async def get_providers():
    """Get available LLM providers."""
    return {
        "providers": [
            {
                "id": "gemini",
                "name": "Gemini (via OpenRouter)",
                "default_model": "google/gemini-2.0-flash-001:free",
                "requires_api_key": True,
                "env_var": "OPENROUTER_API_KEY"
            },
            {
                "id": "claude",
                "name": "Claude (Anthropic)",
                "default_model": "claude-3-5-haiku-20241022",
                "requires_api_key": True,
                "env_var": "ANTHROPIC_API_KEY"
            }
        ]
    }


def main():
    """Run the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Music Mood Analyzer Web Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
