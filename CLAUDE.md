# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Flask-based backend service for Lao ASR (Automatic Speech Recognition) called "Sianglao Backend". It processes audio files through multiple ML models and provides ensemble-based transcription results in Lao language.

## Architecture

### Core Components

- **app.py** - Main Flask application with REST API endpoints
- **asr_service.py** - Business logic service layer handling audio processing and ML integration  
- **models.py** - Data models (ASRRequest, AudioInput, Transcription) with lifecycle management
- **config.py** - Comprehensive configuration management with environment-specific settings
- **utils.py** - Utility functions (currently empty)

### Request Flow

1. Audio upload via `/api/upload` → creates ASRRequest with file validation
2. Transcription start via `/api/transcribe/<request_id>` → initiates background processing
3. Status monitoring via `/api/status/<request_id>` → tracks progress
4. Result retrieval via `/api/result/<request_id>` → returns Lao text with confidence scores

### Key Design Patterns

- **Async Processing**: Background threads handle ML inference while API returns immediately
- **Ensemble Strategy**: Combines predictions from multiple models (xls-r, xlsr-53, hubert) using configurable weights
- **Request Lifecycle**: PENDING → PROCESSING → COMPLETED/FAILED with automatic cleanup
- **Storage Management**: Per-request directories with configurable expiration

## Development Commands

### Running the Application

```bash
# Start the Flask server
python app.py

# With environment variables
FLASK_ENV=development python app.py
FLASK_ENV=production python app.py
```

### Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Key dependencies: Flask, librosa, soundfile, requests, numpy
```

### Configuration

The application uses environment-based configuration:
- Development: `DevelopmentConfig` (default)
- Production: `ProductionConfig` 
- Testing: `TestingConfig`

Set `FLASK_ENV` environment variable to control which config is used.

## API Endpoints

### Primary Workflow
1. `POST /api/upload` - Upload audio file
2. `POST /api/transcribe/<request_id>` - Start transcription  
3. `GET /api/status/<request_id>` - Check progress
4. `GET /api/result/<request_id>` - Get transcription results

### Additional Endpoints
- `GET /` - Service info and available endpoints
- `GET /health` - Health check
- `GET /api/audio/<request_id>` - Stream audio for playback
- `DELETE /api/cleanup/<request_id>` - Manual cleanup

## ML Service Integration

The backend expects an external ML service running on `http://localhost:8000` with:
- `/predict` endpoint accepting audio files
- Returns ensemble predictions from multiple ASR models
- Configurable timeout (120s default) and retry logic

## Storage and Cleanup

- Audio files stored in `./uploads/<request_id>/` directories  
- Automatic cleanup after 1 minute (configurable)
- Background cleanup scheduler runs every 90 seconds
- Manual cleanup available via API endpoint

## Error Handling

- Standardized JSON error responses with timestamps
- Request-level error tracking in ASRRequest.errorInfo
- Graceful degradation with fallback strategies
- Debug information in development mode

## Security Notes

- File type validation for audio uploads
- Filename sanitization
- CORS enabled for development (configurable origins)
- 10MB upload size limit
- Audio quality validation (silence, clipping, corruption checks)

## Testing

No formal test framework is currently configured. The codebase includes a TestingConfig for future test implementation.