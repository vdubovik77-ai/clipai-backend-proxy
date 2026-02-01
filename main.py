"""
ClipAI Backend - FastAPI Proxy Server
Handles Gemini API requests and serves frontend
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import httpx
import os
from typing import Optional, List, Dict, Any

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD_dxDs44sxpzWnlsWh-y5yUrxXaNVDNLs")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

app = FastAPI(
    title="ClipAI Backend",
    description="Proxy server for ClipAI Music Video Platform",
    version="1.0.0"
)

# CORS - Allow all origins for mobile/desktop access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class GenerateContentRequest(BaseModel):
    contents: List[Dict[str, Any]]
    generationConfig: Optional[Dict[str, Any]] = None

class ScriptRequest(BaseModel):
    title: str
    genre: str = "Unknown"
    style: str = "cinematic"
    mood: str = "Various"
    tempo: str = "Medium"
    duration: int = 180
    sceneDuration: int = 8
    numScenes: int = 6
    description: str = ""
    lyrics: str = ""

class ProjectData(BaseModel):
    id: str
    title: str
    style: str
    generator: str
    status: str
    audio_url: Optional[str] = None
    audio_file: Optional[Dict] = None
    audio_duration: int = 180
    song_description: str = ""
    genre: str = ""
    mood: str = ""
    tempo: str = ""
    lyrics: str = ""
    script: List[Dict] = []
    script_approved: bool = False
    final_video_url: Optional[str] = None
    progress: int = 0

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Check if backend is running"""
    return {"status": "ok", "service": "ClipAI Backend"}

# Gemini API Proxy - List Models
@app.get("/api/gemini/models")
async def list_models():
    """Proxy request to Gemini API to list available models"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{GEMINI_BASE_URL}/models?key={GEMINI_API_KEY}&pageSize=20"
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gemini API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gemini API Proxy - Generate Content
@app.post("/api/gemini/generate")
async def generate_content(request: GenerateContentRequest):
    """Proxy request to Gemini API for content generation"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GEMINI_BASE_URL}/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
                json=request.dict(exclude_none=True),
                headers={"Content-Type": "application/json"}
            )
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gemini API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate Script Endpoint
@app.post("/api/script/generate")
async def generate_script(request: ScriptRequest):
    """Generate music video script using Gemini API"""
    
    prompt = f"""You are an expert music video director. Create a detailed scene-by-scene script.

SONG: "{request.title}"
Genre: {request.genre}
Style: {request.style}
Mood: {request.mood}
Tempo: {request.tempo}
Duration: {request.duration}s
Scene Duration: {request.sceneDuration}s
Number of Scenes: {request.numScenes}

{request.description and f"Description: {request.description}" or ""}
{request.lyrics and f"Lyrics: {request.lyrics}" or ""}

Create {request.numScenes} scenes. For each scene provide:
- scene_number (integer)
- timestamp (MM:SS format)
- duration (seconds)
- type (intro/verse/chorus/bridge/outro)
- description (2-3 sentences, vivid visual description)
- visual_prompt (detailed prompt for AI image generation, 60-100 words, include style, lighting, colors, composition)
- video_prompt (detailed prompt for AI video generation, 60-100 words, include camera movement, motion, effects)
- camera_movement (e.g., "slow push in", "static wide", "tracking shot", "drone aerial", "dolly zoom")
- mood (emotional tone of the scene)

Respond ONLY with a valid JSON array. Example format:
[
  {{
    "scene_number": 1,
    "timestamp": "0:00",
    "duration": 8,
    "type": "intro",
    "description": "Dark silhouette against neon city lights",
    "visual_prompt": "Cinematic cyberpunk scene, silhouette of person against neon-lit city skyline, rain falling, reflections on wet pavement, purple and blue lighting, 8K quality",
    "video_prompt": "Slow push in on silhouette, rain particles falling, neon lights flickering subtly, atmospheric haze",
    "camera_movement": "slow push in",
    "mood": "mysterious"
  }}
]
"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GEMINI_BASE_URL}/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.8,
                        "maxOutputTokens": 8000
                    }
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                error_data = response.json()
                return JSONResponse(
                    content={"error": "Gemini API error", "details": error_data},
                    status_code=response.status_code
                )
            
            data = response.json()
            content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Extract JSON from response
            import json
            import re
            
            try:
                # Try to find JSON array in response
                json_match = re.search(r'\[[\s\S]*\]', content)
                if json_match:
                    script = json.loads(json_match.group(0))
                else:
                    script = json.loads(content)
                
                # Add IDs and initialize fields
                for i, scene in enumerate(script):
                    scene["id"] = f"scene_{int(__import__('time').time() * 1000)}_{i}"
                    scene["status"] = "pending"
                    scene["image_url"] = None
                    scene["video_url"] = None
                    scene["generating_image"] = False
                
                return {"script": script, "source": "gemini"}
                
            except json.JSONDecodeError as e:
                return JSONResponse(
                    content={"error": "Failed to parse script", "raw_content": content[:500]},
                    status_code=500
                )
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Gemini API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Fallback Script Generation (when Gemini is unavailable)
@app.post("/api/script/fallback")
async def fallback_script(request: ScriptRequest):
    """Generate fallback script locally when Gemini API fails"""
    
    types = ['intro', 'verse', 'chorus', 'bridge', 'outro']
    cameras = ['slow push in', 'static wide', 'tracking shot', 'drone aerial', 'dolly zoom', 'handheld', 'crane up']
    moods = ['melancholic', 'energetic', 'mysterious', 'romantic', 'intense', 'dreamy', 'epic']
    
    script = []
    for i in range(request.numScenes):
        start_sec = i * request.sceneDuration
        mins = start_sec // 60
        secs = start_sec % 60
        type_idx = 0 if i == 0 else 4 if i == request.numScenes - 1 else min((i * 3 // request.numScenes) + 1, 3)
        mood = moods[i % len(moods)]
        camera = cameras[i % len(cameras)]
        
        script.append({
            "id": f"scene_{int(__import__('time').time() * 1000)}_{i}",
            "scene_number": i + 1,
            "timestamp": f"{mins}:{secs:02d}",
            "duration": request.sceneDuration,
            "type": types[type_idx],
            "description": f"{types[type_idx].capitalize()} scene with {mood} atmosphere. {camera} movement captures the emotional essence. Rich colors enhance the {request.style} aesthetic.",
            "visual_prompt": f"{request.style} {request.genre} music video, {mood} atmosphere, dramatic cinematic lighting, professional 8K cinematography, film grain, detailed textures, {camera} composition, artistic, high quality",
            "video_prompt": f"{request.style} video with {camera} camera movement, smooth motion, {mood} mood, professional 4K quality, atmospheric effects, dynamic composition",
            "camera_movement": camera,
            "mood": mood,
            "status": "pending",
            "image_url": None,
            "video_url": None,
            "generating_image": False
        })
    
    return {"script": script, "source": "fallback"}

# Image Generation Proxy
@app.get("/api/image/generate")
async def generate_image(prompt: str, seed: int = 0, width: int = 1024, height: int = 576):
    """Proxy to Pollinations AI for image generation"""
    try:
        encoded_prompt = prompt[:400].replace(' ', '%20')
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&seed={seed}&nologo=true&enhance=true"
        return {"image_url": image_url, "source": "pollinations"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve Frontend - Mount static files
@app.get("/")
async def serve_index():
    """Serve main index.html"""
    index_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

# Catch-all for frontend routes (SPA support)
@app.get("/{path:path}")
async def serve_spa(path: str):
    """Serve index.html for all routes (SPA)"""
    # API routes should be handled before this
    if path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    index_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="index.html not found")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
