from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # API keys
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")
    langsmith_api_key: str | None = Field(None, env="LANGSMITH_API_KEY")

    # App environment
    env: str = Field("development", env="ENV")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
