from pydantic import BaseSettings, List


class Settings(BaseSettings):
    API_KEY: str = "sk-scp-..."
    MAX_TOKENS: int = 2048
    DEFAULT_MODEL: str = "scp-1"
    ALLOWED_MODELS: List[str] = ["scp-1"]
    
    class Config:
        env_file = ".env"