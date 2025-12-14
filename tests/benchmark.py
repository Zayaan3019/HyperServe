import asyncio
import aiohttp
import time
import random
import pandas as pd
import structlog
import os
import sys

# Configure structured logging
structlog.configure(processors=[structlog.processors.JSONRenderer()])
logger = structlog.get_logger()

SERVER_URL = "http://127.0.0.1:8000/v1/chat/completions"
SYSTEM_PREFIX = [101, 102, 103]

async def send_request(session, req_id, tokens, req_type):
    start = time.perf_counter()
    try:
        async with session.post(SERVER_URL, json={"prompt_ids": tokens}) as resp:
            data = await resp.json()
            end = time.perf_counter()
            
            # Safe metrics extraction
            metrics = data.get("metrics", {})
            return {
                "id": req_id,
                "type": req_type,
                "latency_ms": (end - start) * 1000,
                "hit_rate": metrics.get("cache_hit_rate", 0),
                "status": resp.status
            }
    except Exception as e:
        # Fail gracefully
        return {
            "id": req_id, "type": req_type, "latency_ms": 0, "hit_rate": 0, "status": 500, "error": str(e)
        }

async def run_benchmark(num_requests=100):
    """
    Main entry point called directly by the dashboard.
    """
    logger.info("benchmark_started", requests=num_requests)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_requests):
            # 50/50 Split for Cache Hit/Miss simulation
            if random.random() > 0.5:
                # Warm Request (Shares Prefix)
                tokens = SYSTEM_PREFIX + [random.randint(200, 999)]
                req_type = "Warm (Cached)"
            else:
                # Cold Request (Random)
                tokens = [random.randint(1000, 2000) for _ in range(4)]
                req_type = "Cold (Uncached)"
            
            tasks.append(send_request(session, i, tokens, req_type))
        
        # Run all requests concurrently
        results = await asyncio.gather(*tasks)
        return results

if __name__ == "__main__":
    # Allow standalone execution for debugging
    results = asyncio.run(run_benchmark(100))
    print(pd.DataFrame(results))