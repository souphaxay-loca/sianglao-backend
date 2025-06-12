#!/usr/bin/env python3
"""
Configuration settings for Sianglao Backend Service
"""

import os
from pathlib import Path

class Config:
    """Main configuration class for the sianglao-backend service"""
    
    # ================================
    # Flask Configuration
    # ================================
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 5005
    FLASK_DEBUG = False
    
    # ================================
    # ML Service Configuration
    # ================================
    ML_SERVICE_CONFIG = {
        "base_url": os.getenv('ML_SERVICE_URL', "http://localhost:8000"),
        "endpoints": {
            "predict_all": "/predict",
            "predict_single": "/predict/{model_name}",
            "health": "/health",
            "models": "/models"
        },
        "timeout": 120,  # 2 minutes for ML inference
        "retry_attempts": 2,
        "retry_delay": 1.0  # seconds
    }
    
    # ================================
    # Audio Processing Configuration
    # ================================
    AUDIO_CONFIG = {
        "sample_rate": 16000,
        "channels": 1,  # Mono
        "format": "WAV",
        "subtype": "PCM_16",
        "supported_input_formats": [".wav", ".mp3", ".webm"],
        "max_duration_seconds": 30,
        "min_duration_seconds": 0.5,
        "max_file_size_mb": 10,
        "quality_validation": True
    }
    
    # ================================
    # Audio Quality Validation
    # ================================
    QUALITY_CONFIG = {
        "silence_threshold": 0.01,      # Below this is considered silent
        "clipping_threshold": 0.99,     # Above this is considered clipped
        "max_clipping_ratio": 0.1,      # Max 10% of samples can be clipped
        "check_corruption": True,       # Check for NaN/Inf values
        "normalize_audio": False        # Whether to normalize audio levels
    }
    
    # ================================
    # File Storage Configuration
    # ================================
    STORAGE_CONFIG = {
        "uploads_dir": "./uploads",
        "temp_dir": "./temp",
        "logs_dir": "./logs",
        "cleanup_after_hours": (1/60),      # Delete files after 1 minute
        "max_concurrent_requests": 50,  # Limit concurrent processing
        "request_id_length": 12         # Length of generated request IDs
    }
    
    # ================================
    # Ensemble Configuration
    # ================================
    ENSEMBLE_CONFIG = {
        "strategy": "best_model",  # confidence_weighted, best_model, voting
        "model_weights": {
            "xls-r": 0.45,      # Best performance (15.14% CER)
            "xlsr-53": 0.35,    # Second best (16.22% CER)  
            "hubert": 0.20      # Baseline (25.37% CER)
        },
        "confidence_threshold": 0.5,    # Minimum confidence to trust prediction
        "fallback_to_best": True,       # Use best model if ensemble fails
        "include_individual_results": True  # Return individual model results
    }
    
    # ================================
    # Request Processing Configuration
    # ================================
    PROCESSING_CONFIG = {
        "async_processing": True,       # Enable background processing
        "status_check_interval": 1.0,   # Seconds between status updates
        "max_processing_time": 300,     # 5 minutes max per request
        "enable_progress_updates": True, # Detailed progress tracking
        "cleanup_on_completion": False,  # Keep files for download
        "auto_cleanup_interval": 90   # Cleanup check every hour
    }
    
    # ================================
    # Response Configuration
    # ================================
    RESPONSE_CONFIG = {
        "include_confidence": True,
        "include_processing_time": True,
        "include_audio_info": True,
        "include_model_details": True,
        "decimal_precision": 4,
        "return_raw_predictions": False,  # Include uncleaned predictions
        "enable_debug_info": False       # Extra debugging information
    }
    
    # ================================
    # Error Handling Configuration
    # ================================
    ERROR_CONFIG = {
        "log_errors": True,
        "include_error_details": True,
        "enable_error_notifications": False,
        "max_retries": 3,
        "retry_backoff": 2.0,
        "graceful_degradation": True,
        "fallback_responses": True
    }
    
    # ================================
    # Security Configuration
    # ================================
    SECURITY_CONFIG = {
        "enable_cors": True,
        "allowed_origins": ["http://localhost:3000"],  # testing
        "max_upload_size": 10 * 1024 * 1024,  # 10MB in bytes
        "allowed_file_types": [".wav", ".mp3", ".webm"],
        "validate_file_headers": True,
        "sanitize_filenames": True,
        "rate_limiting": False  # Can be enabled later
    }
    
    # ================================
    # Logging Configuration
    # ================================
    LOGGING_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file_logging": True,
        "console_logging": True,
        "log_file": "logs/sianglao_backend.log",
        "max_log_size": "10MB",
        "backup_count": 5
    }


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    FLASK_DEBUG = True
    
    ML_SERVICE_CONFIG = {
        **Config.ML_SERVICE_CONFIG,
        "timeout": 180,  # Longer timeout for development
    }
    
    ERROR_CONFIG = {
        **Config.ERROR_CONFIG,
        "include_error_details": True,
    }
    
    RESPONSE_CONFIG = {
        **Config.RESPONSE_CONFIG,
        "enable_debug_info": True,
        "return_raw_predictions": True,
    }
    
    LOGGING_CONFIG = {
        **Config.LOGGING_CONFIG,
        "level": "DEBUG",
    }


class ProductionConfig(Config):
    """Production-specific configuration"""
    FLASK_DEBUG = False
    
    ERROR_CONFIG = {
        **Config.ERROR_CONFIG,
        "include_error_details": False,
    }
    
    RESPONSE_CONFIG = {
        **Config.RESPONSE_CONFIG,
        "enable_debug_info": False,
        "return_raw_predictions": False,
    }
    
    SECURITY_CONFIG = {
        **Config.SECURITY_CONFIG,
        "allowed_origins": ["https://yourdomain.com"],  # Update for production
    }


class TestingConfig(Config):
    """Testing-specific configuration"""
    FLASK_DEBUG = True
    
    STORAGE_CONFIG = {
        **Config.STORAGE_CONFIG,
        "uploads_dir": "./test_uploads",
        "cleanup_after_hours": 1,  # Quick cleanup for tests
    }
    
    PROCESSING_CONFIG = {
        **Config.PROCESSING_CONFIG,
        "max_processing_time": 60,  # Shorter timeout for tests
    }


# ================================
# Configuration Selection
# ================================
def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()


# ================================
# Utility Functions
# ================================
def create_directories():
    """Create necessary directories"""
    config = get_config()
    
    directories = [
        config.STORAGE_CONFIG["uploads_dir"],
        config.STORAGE_CONFIG["temp_dir"], 
        config.STORAGE_CONFIG["logs_dir"]
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def get_ml_service_url(endpoint_name: str, **kwargs) -> str:
    """Build ML service URL for given endpoint"""
    config = get_config()
    base_url = config.ML_SERVICE_CONFIG["base_url"]
    endpoint = config.ML_SERVICE_CONFIG["endpoints"][endpoint_name]
    
    # Format endpoint with any provided parameters
    if kwargs:
        endpoint = endpoint.format(**kwargs)
    
    return f"{base_url}{endpoint}"


def validate_audio_file_type(filename: str) -> bool:
    """Check if file type is supported"""
    config = get_config()
    file_ext = Path(filename).suffix.lower()
    return file_ext in config.AUDIO_CONFIG["supported_input_formats"]


def get_max_file_size_bytes() -> int:
    """Get maximum file size in bytes"""
    config = get_config()
    return config.SECURITY_CONFIG["max_upload_size"]


# ================================
# Export default config
# ================================
config = get_config()

# For easy imports
AUDIO_CONFIG = config.AUDIO_CONFIG
ENSEMBLE_CONFIG = config.ENSEMBLE_CONFIG
STORAGE_CONFIG = config.STORAGE_CONFIG
ML_SERVICE_CONFIG = config.ML_SERVICE_CONFIG