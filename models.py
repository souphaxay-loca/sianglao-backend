#!/usr/bin/env python3
"""
Data models for Sianglao Backend Service
Contains ASRRequest, AudioInput, and Transcription classes
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from config import config, STORAGE_CONFIG


class RequestStatus(Enum):
    """Enumeration for ASR request status"""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ASRRequest:
    """
    Represents an ASR transcription request
    Tracks the lifecycle of a transcription job
    """
    
    def __init__(self, request_id: Optional[str] = None):
        # Core attributes from UML
        self.requestID = request_id or self._generate_request_id()
        self.status = RequestStatus.PENDING.value
        self.errorInfo = ""
        self.submissionTime = datetime.now()
        self.completionTime: Optional[datetime] = None
        self.associatedAudio: Optional[str] = None  # Path to audio file
        self.result: Optional['Transcription'] = None
        
        # Additional tracking attributes
        self.processing_start_time: Optional[datetime] = None
        self.processing_duration: Optional[float] = None
        self.ml_service_response: Optional[Dict] = None
        self.progress_info: Dict = {}
        
        print(f"📝 Created ASRRequest: {self.requestID}")
    
    @staticmethod
    def _generate_request_id() -> str:
        """Generate unique request ID"""
        # Use timestamp + random UUID for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"req_{timestamp}_{unique_id}"
    
    def setResult(self, transcription: 'Transcription') -> None:
        """Set transcription result and mark as completed"""
        self.result = transcription
        self.status = RequestStatus.COMPLETED.value
        self.completionTime = datetime.now()
        
        # Calculate processing duration
        if self.processing_start_time:
            self.processing_duration = (
                self.completionTime - self.processing_start_time
            ).total_seconds()
        
        print(f"✅ ASRRequest {self.requestID} completed in {self.processing_duration:.2f}s")
    
    def setError(self, error_message: str) -> None:
        """Set error status and message"""
        self.status = RequestStatus.FAILED.value
        self.errorInfo = error_message
        self.completionTime = datetime.now()
        
        print(f"❌ ASRRequest {self.requestID} failed: {error_message}")
    
    def setStatus(self, status: str, progress_info: Optional[Dict] = None) -> None:
        """Update request status with optional progress information"""
        old_status = self.status
        self.status = status
        
        if progress_info:
            self.progress_info.update(progress_info)
        
        # Mark processing start time
        if status == RequestStatus.PROCESSING.value and old_status == RequestStatus.PENDING.value:
            self.processing_start_time = datetime.now()
        
        print(f"🔄 ASRRequest {self.requestID}: {old_status} → {status}")
    
    def cleanup(self) -> None:
        """Clean up associated files and resources"""
        try:
            # Clean up audio file
            if self.associatedAudio and os.path.exists(self.associatedAudio):
                os.remove(self.associatedAudio)
                print(f"🗑️  Deleted audio file: {self.associatedAudio}")
            
            # Clean up request directory
            request_dir = self.get_request_directory()
            if request_dir and request_dir.exists():
                # Remove directory and all contents
                import shutil
                shutil.rmtree(request_dir)
                print(f"🗑️  Deleted request directory: {request_dir}")
            
            print(f"🧹 Cleaned up ASRRequest: {self.requestID}")
            
        except Exception as e:
            print(f"⚠️  Error cleaning up ASRRequest {self.requestID}: {e}")
    
    def get_request_directory(self) -> Optional[Path]:
        """Get the directory path for this request"""
        if not self.requestID:
            return None
        return Path(STORAGE_CONFIG["uploads_dir"]) / self.requestID
    
    def is_expired(self) -> bool:
        """Check if request has expired based on configuration"""
        if not self.submissionTime:
            return False
        
        expiry_hours = STORAGE_CONFIG["cleanup_after_hours"]
        expiry_time = self.submissionTime + timedelta(hours=expiry_hours)
        return datetime.now() > expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary for JSON serialization"""
        return {
            "requestID": self.requestID,
            "status": self.status,
            "errorInfo": self.errorInfo,
            "submissionTime": self.submissionTime.isoformat() if self.submissionTime else None,
            "completionTime": self.completionTime.isoformat() if self.completionTime else None,
            "processing_duration": self.processing_duration,
            "progress_info": self.progress_info,
            "has_result": self.result is not None,
            "audio_file": self.associatedAudio
        }


class AudioInput:
    """
    Represents uploaded audio input file
    Handles audio file storage and metadata
    """
    
    def __init__(self, request_id: str, original_filename: str):
        # Core attributes from UML
        self.audioID = self._generate_audio_id()
        self.filePath: Optional[str] = None
        self.originalClientFilename = original_filename
        self.mediaType = ""
        self.fileSize = 0
        
        # Additional attributes
        self.request_id = request_id
        self.processed_file_path: Optional[str] = None
        self.audio_duration: Optional[float] = None
        self.sample_rate: Optional[int] = None
        self.channels: Optional[int] = None
        self.created_time = datetime.now()
        self.validation_status = "pending"
        self.validation_message = ""
        
        print(f"🎵 Created AudioInput: {self.audioID} for request {request_id}")
    
    @staticmethod
    def _generate_audio_id() -> str:
        """Generate unique audio ID"""
        return f"audio_{uuid.uuid4().hex[:12]}"
    
    def setFilePath(self, file_path: str) -> None:
        """Set the file path and extract file metadata"""
        self.filePath = file_path
        
        try:
            # Get file size
            if os.path.exists(file_path):
                self.fileSize = os.path.getsize(file_path)
                
                # Extract media type from extension
                file_ext = Path(file_path).suffix.lower()
                self.mediaType = self._get_media_type(file_ext)
                
                print(f"📁 AudioInput {self.audioID}: {self.fileSize} bytes, {self.mediaType}")
            
        except Exception as e:
            print(f"⚠️  Error setting file path for AudioInput {self.audioID}: {e}")
    
    @staticmethod
    def _get_media_type(file_extension: str) -> str:
        """Get media type from file extension"""
        media_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac'
        }
        return media_types.get(file_extension.lower(), 'audio/unknown')
    
    def getFilePath(self) -> Optional[str]:
        """Get the current file path"""
        return self.filePath
    
    def getProcessedFilePath(self) -> Optional[str]:
        """Get the processed (WAV) file path"""
        return self.processed_file_path or self.filePath
    
    def setAudioMetadata(self, duration: float, sample_rate: int, channels: int) -> None:
        """Set audio metadata after processing"""
        self.audio_duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        
        print(f"🎶 AudioInput {self.audioID}: {duration:.2f}s, {sample_rate}Hz, {channels}ch")
    
    def setValidationResult(self, is_valid: bool, message: str = "") -> None:
        """Set audio validation result"""
        self.validation_status = "valid" if is_valid else "invalid"
        self.validation_message = message
        
        status_icon = "✅" if is_valid else "❌"
        print(f"{status_icon} AudioInput {self.audioID} validation: {message or self.validation_status}")
    
    def cleanup(self) -> None:
        """Clean up audio files"""
        files_to_clean = [self.filePath, self.processed_file_path]
        
        for file_path in files_to_clean:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️  Deleted audio file: {file_path}")
                except Exception as e:
                    print(f"⚠️  Error deleting {file_path}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "audioID": self.audioID,
            "originalFilename": self.originalClientFilename,
            "mediaType": self.mediaType,
            "fileSize": self.fileSize,
            "duration": self.audio_duration,
            "sampleRate": self.sample_rate,
            "channels": self.channels,
            "validationStatus": self.validation_status,
            "validationMessage": self.validation_message,
            "createdTime": self.created_time.isoformat()
        }


class Transcription:
    """
    Represents the final transcription result
    Contains Lao text output and confidence information
    """
    
    def __init__(self, lao_text: str, overall_confidence: float = 0.0):
        # Core attributes from UML  
        self.transcriptionID = self._generate_transcription_id()
        self.laoText = lao_text
        self.overallConfidence = overall_confidence
        
        # Additional attributes for ensemble results
        self.individual_results: Dict[str, Dict] = {}
        self.ensemble_method = ""
        self.processing_time: Optional[float] = None
        self.model_weights: Optional[Dict[str, float]] = None
        self.created_time = datetime.now()
        self.quality_score: Optional[float] = None
        
        print(f"📜 Created Transcription: {self.transcriptionID}")
        print(f"   Text: {lao_text[:50]}{'...' if len(lao_text) > 50 else ''}")
        print(f"   Confidence: {overall_confidence:.3f}")
    
    @staticmethod
    def _generate_transcription_id() -> str:
        """Generate unique transcription ID"""
        return f"trans_{uuid.uuid4().hex[:12]}"
    
    def setIndividualResults(self, results: Dict[str, Dict]) -> None:
        """Set individual model results from ML service"""
        self.individual_results = results
        print('---- results ---', results)
        
        print(f"🔍 Transcription {self.transcriptionID}: {len(results)} model results")
        for model, result in results.items():
            conf = result.get('confidence', 0)
            print(f"   {model}: {conf:.3f} confidence")
    
    def setEnsembleInfo(self, method: str, weights: Optional[Dict[str, float]] = None) -> None:
        """Set ensemble method and weights used"""
        self.ensemble_method = method
        self.model_weights = weights
        
        print(f"🧠 Transcription {self.transcriptionID}: {method} ensemble")
        if weights:
            for model, weight in weights.items():
                print(f"   {model}: {weight:.2f} weight")
    
    def exportAsText(self) -> str:
        """Export transcription as plain text"""
        return self.laoText
    
    def getLaoText(self) -> str:
        """Get the Lao text result"""
        return self.laoText
    
    def exportAsJson(self) -> str:
        """Export full transcription data as JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "transcriptionID": self.transcriptionID,
            "laoText": self.laoText,
            "overallConfidence": self.overallConfidence,
            "individualResults": self.individual_results,
            "ensembleMethod": self.ensemble_method,
            "modelWeights": self.model_weights,
            "processingTime": self.processing_time,
            "qualityScore": self.quality_score,
            "createdTime": self.created_time.isoformat()
        }
    
    def getDetailedResult(self) -> Dict[str, Any]:
        """Get detailed result including all model outputs"""
        return {
            "transcription": self.to_dict(),
            "summary": {
                "text": self.laoText,
                "confidence": self.overallConfidence,
                "method": self.ensemble_method,
                "models_used": list(self.individual_results.keys()) if self.individual_results else [],
                "processing_time": self.processing_time
            }
        }


# ================================
# Utility Functions
# ================================
def create_request_directory(request_id: str) -> Path:
    """Create directory for request files"""
    request_dir = Path(STORAGE_CONFIG["uploads_dir"]) / request_id
    request_dir.mkdir(parents=True, exist_ok=True)
    return request_dir


def save_uploaded_file(request_id: str, uploaded_file, original_filename: str) -> AudioInput:
    """Save uploaded file and create AudioInput object"""
    # Create AudioInput object
    audio_input = AudioInput(request_id, original_filename)
    
    # Create request directory
    request_dir = create_request_directory(request_id)
    
    # Generate safe filename
    safe_filename = f"original_{audio_input.audioID}{Path(original_filename).suffix}"
    file_path = request_dir / safe_filename
    
    # Save file
    uploaded_file.save(str(file_path))
    audio_input.setFilePath(str(file_path))
    
    return audio_input


def cleanup_expired_requests(active_requests: Dict[str, ASRRequest]) -> None:
    """Clean up expired requests"""
    expired_ids = [
        req_id for req_id, request in active_requests.items()
        if request.is_expired()
    ]
    
    for req_id in expired_ids:
        try:
            active_requests[req_id].cleanup()
            del active_requests[req_id]
            print(f"🧹 Cleaned up expired request: {req_id}")
        except Exception as e:
            print(f"⚠️  Error cleaning up expired request {req_id}: {e}")