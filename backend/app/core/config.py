from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API keys
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    langsmith_api_key: str | None = Field(None, env="LANGSMITH_API_KEY")

    langsmith_tracing: str = Field("true", env="LANGSMITH_TRACING")
    langsmith_project: str = Field("ResMe-Fullstack", env="LANGSMITH_PROJECT")
    # App environment
    env: str = Field("development", env="ENV")

    # Google AI (Gemini)
    google_api_key: str | None = Field(None, env="GOOGLE_API_KEY")

    # Model configuration
    model_analyst: str = Field("llama-3.3-70b-versatile", env="MODEL_ANALYST")
    model_editor: str = Field("llama-3.1-8b-instant", env="MODEL_EDITOR")
    model_gemini: str = Field("gemini-2.0-flash", env="MODEL_GEMINI")

    # Supabase
    supabase_url: str | None = Field(None, env="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(None, env="SUPABASE_SERVICE_ROLE_KEY")

    # Council of Agents
    enable_council_mode: bool = Field(True, env="ENABLE_COUNCIL_MODE")
    council_editor_count: int = Field(3, env="COUNCIL_EDITOR_COUNT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
