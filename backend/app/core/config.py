from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # Application
    PROJECT_NAME: str = "ChefGenius"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/chefgenius"
    MONGODB_URL: str = "mongodb://localhost:27017/chefgenius"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Search and Indexing
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # AI Models
    RECIPE_MODEL_PATH: str = "mistralai/Mistral-7B-Instruct-v0.1"
    RECIPE_MODEL_LOCAL_PATH: str = "models/recipe_generation"
    SUBSTITUTION_MODEL_PATH: str = "models/substitution"
    NUTRITION_MODEL_PATH: str = "models/nutrition"
    VISION_MODEL_PATH: str = "models/vision"
    
    # External API Keys
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # File Storage
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"
    
    # Task Queue (Celery)
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Email Configuration
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAIL_FROM: str = "noreply@chefgenius.com"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 100
    
    # Caching
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000
    
    # Model Configuration
    MAX_RECIPE_GENERATION_LENGTH: int = 2048
    BATCH_PREDICTION_SIZE: int = 8
    MODEL_CACHE_SIZE: int = 3
    USE_4BIT_QUANTIZATION: bool = True
    TORCH_DTYPE: str = "bfloat16"
    
    @property
    def database_url_async(self) -> str:
        """Get async database URL for SQLAlchemy."""
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def log_file_path(self) -> Optional[Path]:
        """Get full path for log file."""
        if self.LOG_FILE:
            return Path(self.LOG_FILE)
        return None
    
    @property
    def upload_dir_path(self) -> Path:
        """Get full path for upload directory."""
        upload_path = Path(self.UPLOAD_DIR)
        upload_path.mkdir(parents=True, exist_ok=True)
        return upload_path
    
    def get_model_path(self, model_type: str) -> Path:
        """Get full path for a specific model."""
        model_paths = {
            "recipe": self.RECIPE_MODEL_PATH,
            "substitution": self.SUBSTITUTION_MODEL_PATH,
            "nutrition": self.NUTRITION_MODEL_PATH,
            "vision": self.VISION_MODEL_PATH,
        }
        
        if model_type not in model_paths:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return Path(model_paths[model_type])
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "forbid"  # Prevent extra fields
        
        @classmethod
        def prepare_field(cls, field) -> None:
            if 'alias' in field.field_info:
                return
        
        # Environment variable aliases
        fields = {
            'DATABASE_URL': {
                'env': ['DATABASE_URL', 'POSTGRES_URL', 'DB_URL']
            },
            'SECRET_KEY': {
                'env': ['SECRET_KEY', 'JWT_SECRET', 'APP_SECRET']
            },
            'REDIS_URL': {
                'env': ['REDIS_URL', 'CACHE_URL']
            }
        }

# Global settings instance
settings = Settings()

# Validation on import
if settings.is_production() and settings.SECRET_KEY == "your-super-secret-key-change-in-production":
    raise ValueError(
        "SECRET_KEY must be changed in production environment. "
        "Please set a secure SECRET_KEY in your environment variables."
    )