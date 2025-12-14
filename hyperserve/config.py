from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    LOG_LEVEL: str = "INFO"
    MAX_GPU_BLOCKS: int = 1024  # Simulated VRAM slots
    BLOCK_SIZE: int = 16        # Tokens per block (vLLM standard)
    
    class Config:
        env_file = ".env"

settings = Settings()