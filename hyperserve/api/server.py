from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from hyperserve.serving.engine import HyperEngine
import structlog

# Initialize JSON Logging
structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()

app = FastAPI(title="HyperServe: Disaggregated Inference Engine")
engine = HyperEngine()

class GenerateRequest(BaseModel):
    prompt_ids: List[int] # Sending tokens directly for simplicity

@app.get("/health")
async def health():
    return {"status": "operational", "vram_blocks_free": len(engine.allocator.free_blocks)}

@app.post("/v1/chat/completions")
async def generate(req: GenerateRequest):
    try:
        result = await engine.generate(req.prompt_ids)
        return result
    except Exception as e:
        logger.error("inference_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Engine Error")