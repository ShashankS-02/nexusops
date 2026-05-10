"""
NexusOps Configuration — Central settings loaded from environment variables.
All modules import from here instead of reading env vars directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # LangSmith
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "nexusops")

    # Tavily
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://nexusops:nexusops@localhost:5432/nexusops",
    )
    DATABASE_URL_SYNC: str = os.getenv(
        "DATABASE_URL_SYNC",
        "postgresql://nexusops:nexusops@localhost:5432/nexusops",
    )

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

    # App
    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    SENTINEL_ANOMALY_THRESHOLD: float = float(
        os.getenv("SENTINEL_ANOMALY_THRESHOLD", "0.85")
    )
    HUMAN_APPROVAL_REQUIRED: bool = (
        os.getenv("HUMAN_APPROVAL_REQUIRED", "true").lower() == "true"
    )

    # LLM Provider (openai | groq)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # LangGraph checkpoint (SQLite — free, local, no extra service needed)
    CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "nexusops_checkpoints.db")

    # Qdrant (local Docker)
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "nexusops_incidents")

    # Model paths
    PYTORCH_MODEL_PATH: str = os.getenv(
        "PYTORCH_MODEL_PATH", "ml/pytorch/saved_models/lstm_autoencoder.pt"
    )
    TF_MODEL_PATH: str = os.getenv(
        "TF_MODEL_PATH", "ml/tensorflow/saved_models/log_classifier"
    )


settings = Settings()
