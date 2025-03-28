"""
API endpoints for monitoring training processes
"""
import os
import json
import time
import logging
import threading
from typing import Dict, List, Any, Optional
from flask import Blueprint, jsonify, request

# Training status storage
TRAINING_JOBS = {}
TRAINING_JOBS_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

monitor_bp = Blueprint('monitor', __name__)

@monitor_bp.route('/api/monitor/jobs', methods=['GET'])
def list_training_jobs():
    """
    List all active training jobs
    """
    with TRAINING_JOBS_LOCK:
        return jsonify({
            "jobs": list(TRAINING_JOBS.values())
        })

@monitor_bp.route('/api/monitor/jobs/<job_id>', methods=['GET'])
def get_training_job(job_id):
    """
    Get status of a specific training job
    """
    with TRAINING_JOBS_LOCK:
        if job_id not in TRAINING_JOBS:
            return jsonify({
                "error": f"Training job {job_id} not found"
            }), 404
        
        return jsonify(TRAINING_JOBS[job_id])

@monitor_bp.route('/api/monitor/jobs/<job_id>', methods=['DELETE'])
def delete_training_job(job_id):
    """
    Delete a training job from monitoring
    """
    with TRAINING_JOBS_LOCK:
        if job_id not in TRAINING_JOBS:
            return jsonify({
                "error": f"Training job {job_id} not found"
            }), 404
        
        del TRAINING_JOBS[job_id]
        
        return jsonify({
            "message": f"Training job {job_id} deleted"
        })

@monitor_bp.route('/api/monitor/jobs/<job_id>/stop', methods=['POST'])
def stop_training_job(job_id):
    """
    Request to stop a training job
    """
    with TRAINING_JOBS_LOCK:
        if job_id not in TRAINING_JOBS:
            return jsonify({
                "error": f"Training job {job_id} not found"
            }), 404
        
        TRAINING_JOBS[job_id]["status"] = "stopping"
        
        return jsonify({
            "message": f"Request to stop training job {job_id} has been sent"
        })

# Helper functions for other modules to interact with the training monitor

def register_training_job(job_id: str, model_id: str, dataset: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register a new training job for monitoring
    
    Args:
        job_id: Unique identifier for the job
        model_id: Identifier for the model being trained
        dataset: Name of the dataset used for training
        config: Training configuration
        
    Returns:
        The newly created job record
    """
    timestamp = int(time.time())
    
    job = {
        "job_id": job_id,
        "model_id": model_id,
        "dataset": dataset,
        "config": config,
        "start_time": timestamp,
        "status": "initializing",
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": config.get("num_train_epochs", 0),
        "current_step": 0,
        "total_steps": config.get("max_steps", 0),
        "metrics": {},
        "logs": [
            {"timestamp": timestamp, "message": "Training job registered"}
        ],
        "error": None
    }
    
    with TRAINING_JOBS_LOCK:
        TRAINING_JOBS[job_id] = job
    
    return job

def update_training_status(job_id: str, 
                          status: Optional[str] = None,
                          progress: Optional[float] = None,
                          current_epoch: Optional[int] = None,
                          current_step: Optional[int] = None,
                          metrics: Optional[Dict[str, float]] = None,
                          error: Optional[str] = None) -> Dict[str, Any]:
    """
    Update the status of a training job
    
    Args:
        job_id: Unique identifier for the job
        status: New status for the job (training, evaluating, completed, error, stopping, stopped)
        progress: Overall progress (0-100)
        current_epoch: Current training epoch
        current_step: Current training step
        metrics: Dictionary of metric values
        error: Error message (if any)
        
    Returns:
        The updated job record
    """
    timestamp = int(time.time())
    
    with TRAINING_JOBS_LOCK:
        if job_id not in TRAINING_JOBS:
            logger.warning(f"Attempted to update non-existent training job: {job_id}")
            return None
        
        job = TRAINING_JOBS[job_id]
        
        if status is not None:
            job["status"] = status
            job["logs"].append({
                "timestamp": timestamp,
                "message": f"Status changed to: {status}"
            })
        
        if progress is not None:
            job["progress"] = progress
        
        if current_epoch is not None:
            job["current_epoch"] = current_epoch
        
        if current_step is not None:
            job["current_step"] = current_step
        
        if metrics is not None:
            for key, value in metrics.items():
                job["metrics"][key] = value
        
        if error is not None:
            job["error"] = error
            job["status"] = "error"
            job["logs"].append({
                "timestamp": timestamp,
                "message": f"Error: {error}"
            })
        
        # If job is completed, set progress to 100%
        if status == "completed":
            job["progress"] = 100.0
        
        return job

def add_log_message(job_id: str, message: str) -> bool:
    """
    Add a log message to a training job
    
    Args:
        job_id: Unique identifier for the job
        message: Log message to add
        
    Returns:
        True if the message was added successfully, False otherwise
    """
    timestamp = int(time.time())
    
    with TRAINING_JOBS_LOCK:
        if job_id not in TRAINING_JOBS:
            logger.warning(f"Attempted to add log to non-existent training job: {job_id}")
            return False
        
        TRAINING_JOBS[job_id]["logs"].append({
            "timestamp": timestamp,
            "message": message
        })
        
        return True

def check_stop_requested(job_id: str) -> bool:
    """
    Check if a stop has been requested for a training job
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        True if stop was requested, False otherwise
    """
    with TRAINING_JOBS_LOCK:
        if job_id not in TRAINING_JOBS:
            return False
        
        return TRAINING_JOBS[job_id]["status"] == "stopping"

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the current status of a training job
    
    Args:
        job_id: Unique identifier for the job
        
    Returns:
        Job record, or None if job not found
    """
    with TRAINING_JOBS_LOCK:
        if job_id not in TRAINING_JOBS:
            return None
        
        return TRAINING_JOBS[job_id]