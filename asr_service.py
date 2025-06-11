#!/usr/bin/env python3
"""
ASRService - Business logic layer for Sianglao Backend
Handles audio processing, ML service integration, and ensemble logic
"""

import os
import time
import threading
import requests
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

# Import our models and config
from models import ASRRequest, AudioInput, Transcription, RequestStatus, create_request_directory
from config import config, AUDIO_CONFIG, ENSEMBLE_CONFIG, ML_SERVICE_CONFIG, get_ml_service_url

class ASRService:
    """
    Business logic service for ASR processing
    Orchestrates audio processing, ML service calls, and ensemble results
    """
    
    def __init__(self):
        # Core attributes from UML
        self.activeRequests: Dict[str, ASRRequest] = {}
        self.temporaryStorageLocation = config.STORAGE_CONFIG["uploads_dir"]
        self.mlService = get_ml_service_url("predict_all")
        
        # Additional service attributes
        self.processing_threads: Dict[str, threading.Thread] = {}
        self.max_concurrent_requests = config.STORAGE_CONFIG["max_concurrent_requests"]
        
        print(f"🚀 ASRService initialized")
        print(f"📁 Storage: {self.temporaryStorageLocation}")
        print(f"🤖 ML Service: {self.mlService}")
        
        # Ensure storage directory exists
        Path(self.temporaryStorageLocation).mkdir(parents=True, exist_ok=True)
    
    def processAudio(self, request_id: str, audio_source, filename: str, source_type: str = "file") -> bool:
        """
        Unified audio processing for both file uploads and live recordings
        """
        try:
            print(f"📤 Processing audio for request {request_id} (type: {source_type})")
            
            # Get the request object
            if request_id not in self.activeRequests:
                print(f"❌ Request {request_id} not found")
                return False
            
            request = self.activeRequests[request_id]
            
            # Step 1: Save to disk (works for both files and blobs)
            audio_input = self._save_audio_data(request_id, audio_source, filename)
            if not audio_input:
                request.setError("Failed to save audio data")
                return False
            
            # Step 2: Validate audio file
            is_valid, message = self._validate_audio_file(audio_input)
            if not is_valid:
                request.setError(f"Audio validation failed: {message}")
                return False
                
            # Step 3: Convert to standard WAV format
            processed_path = self._convert_to_wav(audio_input) 
            if not processed_path:
                request.setError("Audio format conversion failed")
                return False
            
            # Step 4: Associate processed audio with request
            request.associatedAudio = processed_path
            
            print(f"✅ Audio processing completed for request {request_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error processing audio for {request_id}: {e}")
            if request_id in self.activeRequests:
                self.activeRequests[request_id].setError(f"Audio processing error: {str(e)}")
            return False
    
    def initiateTranscription(self, request_id: str) -> bool:
        """
        Start background transcription process
        Returns immediately, actual processing happens in background thread
        """
        try:
            if request_id not in self.activeRequests:
                print(f"❌ Request {request_id} not found for transcription")
                return False
            
            request = self.activeRequests[request_id]
            
            # Check if already processing
            if request.status == RequestStatus.PROCESSING.value:
                print(f"⚠️  Request {request_id} already processing")
                return True
            
            # Check concurrent request limit
            active_processing = sum(
                1 for r in self.activeRequests.values() 
                if r.status == RequestStatus.PROCESSING.value
            )
            
            if active_processing >= self.max_concurrent_requests:
                request.setError("Server busy - too many concurrent requests")
                return False
            
            # Start background processing
            request.setStatus(RequestStatus.PROCESSING.value, {
                "step": "initiating",
                "message": "Starting transcription process"
            })
            
            # Create and start background thread
            thread = threading.Thread(
                target=self._background_transcription,
                args=(request_id,),
                daemon=True
            )
            
            self.processing_threads[request_id] = thread
            thread.start()
            
            print(f"🚀 Initiated background transcription for {request_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error initiating transcription for {request_id}: {e}")
            if request_id in self.activeRequests:
                self.activeRequests[request_id].setError(f"Transcription initiation error: {str(e)}")
            return False
    
    def _background_transcription(self, request_id: str) -> None:
        """
        Background transcription processing (runs in separate thread)
        """
        try:
            request = self.activeRequests[request_id]
            
            # Step 1: Load and preprocess audio
            request.setStatus(RequestStatus.PROCESSING.value, {
                "step": "loading_audio",
                "message": "Loading and preprocessing audio"
            })
            
            audio_data, sample_rate = self._load_audio_for_ml_service(request.associatedAudio)
            if audio_data is None:
                request.setError("Failed to load audio for ML processing")
                return
            
            # Step 2: Call ML service
            request.setStatus(RequestStatus.PROCESSING.value, {
                "step": "ml_inference", 
                "message": "Running ML model inference"
            })
            
            ml_response = self._call_ml_service(request.associatedAudio)
            # print('---- ml_response ---', ml_response)
            if not ml_response:
                request.setError("ML service call failed")
                return
            
            request.ml_service_response = ml_response
            
            # Step 3: Ensemble processing
            request.setStatus(RequestStatus.PROCESSING.value, {
                "step": "ensemble",
                "message": "Combining model predictions"
            })
            
            final_transcription = self._create_ensemble_transcription(ml_response)
            if not final_transcription:
                request.setError("Ensemble processing failed")
                return
            
            # Step 4: Complete
            request.setResult(final_transcription)
            print(f"🎉 Transcription completed for {request_id}")
            
        except Exception as e:
            print(f"❌ Background transcription error for {request_id}: {e}")
            if request_id in self.activeRequests:
                self.activeRequests[request_id].setError(f"Transcription processing error: {str(e)}")
        
        finally:
            # Clean up thread reference
            if request_id in self.processing_threads:
                del self.processing_threads[request_id]
    
    def getJobStatus(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of transcription job"""
        if request_id not in self.activeRequests:
            return None
        
        request = self.activeRequests[request_id]
        status_info = request.to_dict()
        
        # Add progress information
        if request.status == RequestStatus.PROCESSING.value:
            status_info["progress"] = request.progress_info
        
        return status_info
    
    
    def getResultText(self, request_id: str) -> Optional[Transcription]:
        """Get transcription result"""
        if request_id not in self.activeRequests:
            return None
        
        request = self.activeRequests[request_id]
        return request.result
    
    def cleanupRequest(self, request_id: str) -> bool:
        """Clean up request and associated files"""
        try:
            if request_id in self.activeRequests:
                request = self.activeRequests[request_id]
                
                # Stop processing thread if running
                if request_id in self.processing_threads:
                    # Note: We can't forcefully stop threads in Python
                    # The thread will finish naturally
                    del self.processing_threads[request_id]
                
                # Clean up files
                request.cleanup()
                
                # Remove from active requests
                del self.activeRequests[request_id]
                
                print(f"🧹 Cleaned up request {request_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error cleaning up request {request_id}: {e}")
            return False
    
    # ================================
    # Helper Methods
    # ================================
    
    def _save_audio_data(self, request_id: str, audio_source, original_filename: str) -> AudioInput:
        """Save audio data (file or blob) to request directory"""
        audio_input = AudioInput(request_id, original_filename)
        
        # Create request directory
        request_dir = create_request_directory(request_id)
        
        # Generate safe filename
        safe_filename = f"original_{audio_input.audioID}{Path(original_filename).suffix}"
        file_path = request_dir / safe_filename
        
        # Save audio data (works for both FileStorage and blob data)
        audio_source.save(str(file_path))
        audio_input.setFilePath(str(file_path))
        
        return audio_input
    
    def _validate_audio_file(self, audio_input: AudioInput) -> Tuple[bool, str]:
        """Validate audio file quality and format"""
        try:
            file_path = audio_input.getFilePath()
            
            # Check file exists and has content
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                return False, "Empty or missing audio file"
            
            # Load audio for validation
            try:
                audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)
            except Exception as e:
                return False, f"Cannot read audio file: {str(e)}"
            
            # Duration check
            duration = len(audio_data) / sample_rate
            if duration < AUDIO_CONFIG["min_duration_seconds"]:
                return False, f"Audio too short: {duration:.1f}s (min: {AUDIO_CONFIG['min_duration_seconds']}s)"
            
            if duration > AUDIO_CONFIG["max_duration_seconds"]:
                return False, f"Audio too long: {duration:.1f}s (max: {AUDIO_CONFIG['max_duration_seconds']}s)"
            
            # Quality checks if enabled
            if config.QUALITY_CONFIG["check_corruption"]:
                if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                    return False, "Audio file appears corrupted"
            
            # Silence check
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude < config.QUALITY_CONFIG["silence_threshold"]:
                return False, "Audio appears to be silent"
            
            # Clipping check
            clipped_samples = np.sum(np.abs(audio_data) > config.QUALITY_CONFIG["clipping_threshold"])
            clipping_ratio = clipped_samples / len(audio_data.flatten())
            if clipping_ratio > config.QUALITY_CONFIG["max_clipping_ratio"]:
                return False, f"Audio is clipped ({clipping_ratio*100:.1f}% of samples)"
            
            # Set audio metadata
            audio_input.setAudioMetadata(duration, sample_rate, 1 if len(audio_data.shape) == 1 else audio_data.shape[0])
            
            return True, "Audio validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _convert_to_wav(self, audio_input: AudioInput) -> Optional[str]:
        """Convert audio to standard WAV format for ML service"""
        try:
            input_path = audio_input.getFilePath()
            
            # Load with target settings
            audio_data, _ = librosa.load(
                input_path,
                sr=AUDIO_CONFIG["sample_rate"],
                mono=True
            )
            
            # Generate output path
            request_dir = Path(input_path).parent
            wav_filename = f"processed_{audio_input.audioID}.wav"
            output_path = request_dir / wav_filename
            
            # Save as WAV
            sf.write(
                str(output_path),
                audio_data,
                AUDIO_CONFIG["sample_rate"],
                subtype=AUDIO_CONFIG["subtype"]
            )
            
            print(f"🔄 Converted audio to WAV: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"❌ Audio conversion failed: {e}")
            return None
    
    def _load_audio_for_ml_service(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load processed audio for ML service"""
        try:
            # Use processed WAV file if available, otherwise use original
            if not os.path.exists(audio_path):
                print(f"❌ Audio file not found: {audio_path}")
                return None, None
            
            audio_data, sample_rate = librosa.load(
                audio_path,
                sr=AUDIO_CONFIG["sample_rate"],
                mono=True
            )
            
            return audio_data, sample_rate
            
        except Exception as e:
            print(f"❌ Error loading audio for ML service: {e}")
            return None, None
    
    def _call_ml_service(self, audio_path: str) -> Optional[Dict]:
        """Call ML service for transcription"""
        try:
            print(f"🤖 Calling ML service with audio: {audio_path}")
            
            # Prepare request
            url = get_ml_service_url("predict_all")
            
            with open(audio_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                
                response = requests.post(
                    url,
                    files=files,
                    timeout=ML_SERVICE_CONFIG["timeout"]
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ ML service response received")
                return result
            else:
                print(f"❌ ML service error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"❌ ML service timeout after {ML_SERVICE_CONFIG['timeout']}s")
            return None
        except Exception as e:
            print(f"❌ ML service call failed: {e}")
            return None
    
    def _create_ensemble_transcription(self, ml_response: Dict) -> Optional[Transcription]:
        """Create ensemble transcription from ML service response"""
        try:
            if not ml_response.get("success", False):
                print(f"❌ ML service returned unsuccessful response")
                return None
            
            results = ml_response.get("results", {})
            if not results:
                print(f"❌ No results in ML service response")
                return None
            
            # Apply ensemble strategy
            strategy = ENSEMBLE_CONFIG["strategy"]
            
            if strategy == "confidence_weighted":
                final_text, final_confidence = self._confidence_weighted_ensemble(results)
                final_text = final_text.replace("<unk>", " ")
            elif strategy == "best_model":
                final_text, final_confidence = self._best_model_ensemble(results)
                final_text = final_text.replace("<unk>", " ")
            else:
                # Fallback to best model
                final_text, final_confidence = self._best_model_ensemble(results)
                final_text = final_text.replace("<unk>", " ")
            # Create transcription object
            transcription = Transcription(final_text, final_confidence)
            transcription.setIndividualResults(results)
            transcription.setEnsembleInfo(strategy, ENSEMBLE_CONFIG.get("model_weights"))
            
            if "total_processing_time" in ml_response.get("summary", {}):
                transcription.processing_time = ml_response["summary"]["total_processing_time"]
            
            return transcription
            
        except Exception as e:
            print(f"❌ Ensemble processing failed: {e}")
            return None
    
    def _confidence_weighted_ensemble(self, results: Dict) -> Tuple[str, float]:
        """Confidence-weighted ensemble strategy"""
        weights = ENSEMBLE_CONFIG["model_weights"]
        total_weighted_confidence = 0.0
        best_text = ""
        best_weighted_score = 0.0
        
        for model_name, result in results.items():
            if not result.get("success", False):
                continue
            
            confidence = result.get("confidence", 0.0)
            prediction = result.get("raw_prediction", "")
            weight = weights.get(model_name, 0.1)  # Default small weight
            
            weighted_score = confidence * weight
            total_weighted_confidence += weighted_score
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_text = prediction
        
        return best_text, total_weighted_confidence
    
    def _best_model_ensemble(self, results: Dict) -> Tuple[str, float]:
        """Best model ensemble strategy (highest confidence)"""
        best_confidence = 0.0
        best_text = ""
        
        for model_name, result in results.items():
            if not result.get("success", False):
                continue
            
            confidence = result.get("confidence", 0.0)
            if confidence > best_confidence:
                best_confidence = confidence
                best_text = result.get("raw_prediction", "")
        
        return best_text, best_confidence


# ================================
# Global ASRService Instance
# ================================
asr_service = ASRService()