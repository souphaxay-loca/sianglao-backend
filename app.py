#!/usr/bin/env python3
"""
Sianglao Backend - Flask Application
APIHandler implementation with all endpoints from UML design
"""

import os
import time
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path

# Import our services and models
from config import config, create_directories, validate_audio_file_type, get_max_file_size_bytes
from asr_service import asr_service
from models import ASRRequest, RequestStatus, cleanup_expired_requests

# ================================
# Flask App Initialization
# ================================
app = Flask(__name__)

# Configure CORS
if config.SECURITY_CONFIG["enable_cors"]:
    CORS(app, origins=config.SECURITY_CONFIG["allowed_origins"])

# Configure upload limits
app.config['MAX_CONTENT_LENGTH'] = get_max_file_size_bytes()

# ================================
# Service Information
# ================================
SERVICE_INFO = {
    "service": "Sianglao Backend",
    "description": "Lao ASR Backend with Ensemble Processing",
    "version": "1.0.0",
    "ml_service": config.ML_SERVICE_CONFIG["base_url"],
    "supported_formats": config.AUDIO_CONFIG["supported_input_formats"],
    "max_duration": config.AUDIO_CONFIG["max_duration_seconds"],
    "ensemble_strategy": config.ENSEMBLE_CONFIG["strategy"]
}

STARTUP_TIME = None

# ================================
# APIHandler Implementation
# ================================

@app.route('/', methods=['GET'])
def root():
    """Service information endpoint"""
    uptime = time.time() - STARTUP_TIME if STARTUP_TIME else 0
    
    return jsonify({
        **SERVICE_INFO,
        "status": "running",
        "uptime_seconds": round(uptime, 1),
        "active_requests": len(asr_service.activeRequests),
        "endpoints": {
            "upload": "POST /api/upload",
            "record": "POST /api/record", 
            "transcribe": "POST /api/transcribe/<request_id>",
            "status": "GET /api/status/<request_id>",
            "result": "GET /api/result/<request_id>",
            "audio": "GET /api/audio/<request_id>",
            "download": "GET /api/download/<request_id>",
            "cleanup": "DELETE /api/cleanup/<request_id>"
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    uptime = time.time() - STARTUP_TIME if STARTUP_TIME else 0
    
    return jsonify({
        "status": "healthy",
        "uptime_seconds": round(uptime, 1),
        "active_requests": len(asr_service.activeRequests),
        "ml_service_url": config.ML_SERVICE_CONFIG["base_url"],
        "timestamp": datetime.now().isoformat()
    }), 200


# ================================
# Audio Upload Endpoints
# ================================

@app.route('/api/upload', methods=['POST'])
def submitUploadedAudio():
    """
    Upload audio file and create transcription request
    Returns requestID immediately for async processing
    """
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return create_error_response("No audio file provided", 400)
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return create_error_response("No audio file selected", 400)
        
        # Validate file type
        if not validate_audio_file_type(audio_file.filename):
            supported = ", ".join(config.AUDIO_CONFIG["supported_input_formats"])
            return create_error_response(f"Unsupported file type. Supported: {supported}", 400)
        
        # Create ASR request
        asr_request = ASRRequest()
        request_id = asr_request.requestID
        
        # Add to active requests
        asr_service.activeRequests[request_id] = asr_request
        
        # Process uploaded audio (sync validation)
        success = asr_service.processUploadedAudio(
            request_id, 
            audio_file, 
            secure_filename(audio_file.filename)
        )
        
        if not success:
            error_msg = asr_request.errorInfo or "Audio processing failed"
            return create_error_response(error_msg, 400)
        
        # Return requestID for async processing
        response_data = {
            "requestId": request_id,
            "status": asr_request.status,
            "message": "Audio uploaded successfully. Use /api/transcribe to start processing.",
            "filename": audio_file.filename
        }
        
        return create_success_response(response_data), 201
        
    except Exception as e:
        error_msg = f"Upload failed: {str(e)}"
        print(f"❌ {error_msg}")
        return create_error_response(error_msg, 500)


# @app.route('/api/record', methods=['POST'])
# def submitRecordedAudio():
#     """
#     Submit recorded audio data (future implementation)
#     """
#     return create_error_response("Recorded audio submission not yet implemented", 501)


# ================================
# Transcription Processing Endpoints
# ================================

@app.route('/api/transcribe/<request_id>', methods=['POST'])
def startTranscription(request_id):
    """
    Start transcription processing for uploaded audio
    Initiates background processing
    """
    try:
        if request_id not in asr_service.activeRequests:
            return create_error_response("Request not found", 404)
        
        asr_request = asr_service.activeRequests[request_id]
        
        # Check if already completed or processing
        if asr_request.status == RequestStatus.COMPLETED.value:
            return create_error_response("Request already completed", 400)
        
        if asr_request.status == RequestStatus.PROCESSING.value:
            return create_error_response("Request already processing", 400)
        
        if asr_request.status == RequestStatus.FAILED.value:
            return create_error_response(f"Request failed: {asr_request.errorInfo}", 400)
        
        # Start transcription
        success = asr_service.initiateTranscription(request_id)
        
        if not success:
            error_msg = asr_request.errorInfo or "Failed to start transcription"
            return create_error_response(error_msg, 500)
        
        response_data = {
            "requestId": request_id,
            "status": asr_request.status,
            "message": "Transcription started. Use /api/status to check progress."
        }
        
        return create_success_response(response_data), 200
        
    except Exception as e:
        error_msg = f"Transcription start failed: {str(e)}"
        print(f"❌ {error_msg}")
        return create_error_response(error_msg, 500)


# ================================
# Status and Results Endpoints
# ================================

@app.route('/api/status/<request_id>', methods=['GET'])
def checkStatus(request_id):
    """
    Check status of transcription request
    Returns current status and progress information
    """
    try:
        status_info = asr_service.getJobStatus(request_id)
        
        if not status_info:
            return create_error_response("Request not found", 404)
        
        # Add helpful messages based on status
        status = status_info.get("status", "unknown")
        if status == RequestStatus.COMPLETED.value:
            status_info["message"] = "Transcription completed. Use /api/result to get results."
        elif status == RequestStatus.PROCESSING.value:
            status_info["message"] = "Transcription in progress..."
        elif status == RequestStatus.FAILED.value:
            status_info["message"] = f"Transcription failed: {status_info.get('errorInfo', 'Unknown error')}"
        else:
            status_info["message"] = "Request pending. Use /api/transcribe to start processing."
        
        return create_success_response(status_info), 200
        
    except Exception as e:
        error_msg = f"Status check failed: {str(e)}"
        print(f"❌ {error_msg}")
        return create_error_response(error_msg, 500)


@app.route('/api/result/<request_id>', methods=['GET'])
def getTranscriptionText(request_id):
    """
    Get transcription results
    Returns Lao text and detailed model information
    """
    try:
        transcription = asr_service.getResultText(request_id)
        
        if not transcription:
            # Check if request exists but not completed
            if request_id in asr_service.activeRequests:
                asr_request = asr_service.activeRequests[request_id]
                if asr_request.status == RequestStatus.PROCESSING.value:
                    return create_error_response("Transcription still processing. Check /api/status", 202)
                elif asr_request.status == RequestStatus.FAILED.value:
                    return create_error_response(f"Transcription failed: {asr_request.errorInfo}", 400)
                else:
                    return create_error_response("Transcription not started. Use /api/transcribe", 400)
            else:
                return create_error_response("Request not found", 404)
        
        # Return detailed results
        result_data = transcription.getDetailedResult()
        
        return create_success_response(result_data), 200
        
    except Exception as e:
        error_msg = f"Result retrieval failed: {str(e)}"
        print(f"❌ {error_msg}")
        return create_error_response(error_msg, 500)


# ================================
# Audio and Download Endpoints
# ================================

@app.route('/api/audio/<request_id>', methods=['GET'])
def getAudioForPlayback(request_id):
    """
    Get audio file for playback
    Returns the processed audio file
    """
    try:
        audio_path = asr_service.retrieveAudioPath(request_id)
        
        if not audio_path or not os.path.exists(audio_path):
            return create_error_response("Audio file not found", 404)
        
        return send_file(
            audio_path,
            as_attachment=False,
            download_name=f"audio_{request_id}.wav",
            mimetype="audio/wav"
        )
        
    except Exception as e:
        error_msg = f"Audio retrieval failed: {str(e)}"
        print(f"❌ {error_msg}")
        return create_error_response(error_msg, 500)


# @app.route('/api/download/<request_id>', methods=['GET'])
# def downloadTranscription(request_id):
#     """
#     Download transcription as text file
#     """
#     try:
#         transcription = asr_service.getResultText(request_id)
        
#         if not transcription:
#             return create_error_response("Transcription not found or not completed", 404)
        
#         # Create temporary text file
#         request_dir = Path(config.STORAGE_CONFIG["uploads_dir"]) / request_id
#         text_file = request_dir / f"transcription_{request_id}.txt"
        
#         with open(text_file, 'w', encoding='utf-8') as f:
#             f.write(transcription.exportAsText())
        
#         return send_file(
#             str(text_file),
#             as_attachment=True,
#             download_name=f"lao_transcription_{request_id}.txt",
#             mimetype="text/plain"
#         )
        
#     except Exception as e:
#         error_msg = f"Download failed: {str(e)}"
#         print(f"❌ {error_msg}")
#         return create_error_response(error_msg, 500)


# ================================
# Cleanup Endpoints
# ================================

@app.route('/api/cleanup/<request_id>', methods=['DELETE'])
def cleanupSession(request_id):
    """
    Clean up request and associated files
    """
    try:
        success = asr_service.cleanupRequest(request_id)
        
        if success:
            response_data = {
                "requestId": request_id,
                "message": "Request cleaned up successfully"
            }
            return create_success_response(response_data), 200
        else:
            return create_error_response("Request not found or cleanup failed", 404)
        
    except Exception as e:
        error_msg = f"Cleanup failed: {str(e)}"
        print(f"❌ {error_msg}")
        return create_error_response(error_msg, 500)


# ================================
# Utility Functions
# ================================

def create_error_response(message: str, status_code: int = 400) -> tuple:
    """Create standardized error response"""
    response = {
        "success": False,
        "error": message,
        "timestamp": datetime.now().isoformat(),
        "service": SERVICE_INFO["service"]
    }
    
    if config.FLASK_DEBUG and status_code >= 500:
        response["traceback"] = traceback.format_exc()
    
    return jsonify(response), status_code


def create_success_response(data: dict) -> dict:
    """Create standardized success response"""
    response = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "service": SERVICE_INFO["service"],
        **data
    }
    
    return jsonify(response)


# ================================
# Error Handlers
# ================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return create_error_response("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return create_error_response("Method not allowed", 405)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    max_size = get_max_file_size_bytes() // (1024 * 1024)  # Convert to MB
    return create_error_response(f"File too large. Maximum size: {max_size}MB", 413)


@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors"""
    return create_error_response("Internal server error", 500)


# ================================
# Background Tasks
# ================================

def cleanup_expired_requests_task():
    """Background task to clean up expired requests"""
    try:
        cleanup_expired_requests(asr_service.activeRequests)
    except Exception as e:
        print(f"⚠️  Error in cleanup task: {e}")
        
def start_cleanup_scheduler():
    """Start background cleanup scheduler"""
    def cleanup_loop():
        while True:
            try:
                time.sleep(config.STORAGE_CONFIG["auto_cleanup_interval"])  # 1 hour
                cleanup_expired_requests_task()
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
    
    # Start background thread
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    print("🧹 Automatic cleanup scheduler started")


# ================================
# Application Startup
# ================================

def initialize_service():
    """Initialize the backend service"""
    global STARTUP_TIME
    
    print("🚀 Initializing Sianglao Backend Service")
    print("=" * 60)
    print(f"Service: {SERVICE_INFO['service']}")
    print(f"Version: {SERVICE_INFO['version']}")
    print(f"ML Service: {SERVICE_INFO['ml_service']}")
    print("=" * 60)
    
    STARTUP_TIME = time.time()
    
    # Create necessary directories
    create_directories()
    print("✅ Directories created")
    
    # Check ML service connectivity (optional)
    try:
        import requests
        ml_health_url = f"{config.ML_SERVICE_CONFIG['base_url']}/health"
        response = requests.get(ml_health_url, timeout=5)
        if response.status_code == 200:
            print("✅ ML service connectivity verified")
        else:
            print("⚠️  ML service not responding (will retry during requests)")
    except Exception:
        print("⚠️  Cannot connect to ML service (will retry during requests)")
    
    startup_time = time.time() - STARTUP_TIME
    print(f"\n🎉 Backend service initialized in {startup_time:.1f}s")
    print(f"🌐 Ready to serve requests on {config.FLASK_HOST}:{config.FLASK_PORT}")
    
    return True


# ================================
# Application Entry Point
# ================================

if __name__ == '__main__':
    try:
        # Initialize service
        if initialize_service():
            print(f"\n🔥 Starting Flask server...")
            print(f"📍 URL: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
            print(f"📤 Upload endpoint: POST /api/upload")
            print(f"🎯 Transcribe endpoint: POST /api/transcribe/<request_id>")
            print(f"📊 Status endpoint: GET /api/status/<request_id>")
            print(f"📝 Result endpoint: GET /api/result/<request_id>")
            print("\n" + "=" * 60)
            
            # Start Flask app
            app.run(
                host=config.FLASK_HOST,
                port=config.FLASK_PORT,
                debug=config.FLASK_DEBUG,
                threaded=True
            )
        else:
            print("❌ Failed to initialize service. Exiting.")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Service interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        exit(1)