"""
API endpoints for model training
"""
import os
import json
import time
import logging
import threading
from flask import Blueprint, request, jsonify, current_app, send_from_directory
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional, Union

# Import dependency checker
from utils.dependency_checker import import_optional_dependency

# Use dependency checker to import and potentially install pandas
pd = import_optional_dependency("pandas")
pandas_available = pd is not None

from training.dataset_handler import DatasetHandler
from training.config import TrainingConfig
from training.trainer import create_trainer
from training.huggingface import HuggingFaceAPI
from models.config import ModelConfig

logger = logging.getLogger(__name__)

# Create Blueprint
training_bp = Blueprint("training", __name__)

# Configure file upload settings
UPLOAD_FOLDER = "uploads/datasets"
ALLOWED_EXTENSIONS = {"txt", "csv", "json"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@training_bp.route("/api/training/config", methods=["GET"])
def get_training_config():
    """
    Get training configuration
    """
    try:
        # Get learning types and their default configs
        learning_types = list(TrainingConfig.DEFAULT_CONFIGS.keys())
        default_configs = {}
        
        for learning_type in learning_types:
            default_configs[learning_type] = TrainingConfig.get_default_config(learning_type)
        
        # Get any custom saved config
        saved_config = TrainingConfig.load_config()
        
        return jsonify({
            "learning_types": learning_types,
            "default_configs": default_configs,
            "saved_config": saved_config
        })
    
    except Exception as e:
        logger.error(f"Error getting training config: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/config", methods=["POST"])
def save_training_config():
    """
    Save training configuration
    """
    try:
        config_data = request.json
        
        if not config_data:
            return jsonify({
                "error": "No configuration data provided"
            }), 400
        
        # Save configuration
        success = TrainingConfig.save_config(config_data)
        
        if not success:
            return jsonify({
                "error": "Error saving configuration"
            }), 500
        
        return jsonify({
            "status": "success",
            "message": "Configuration saved successfully"
        })
    
    except Exception as e:
        logger.error(f"Error saving training config: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/huggingface/search", methods=["GET"])
def search_huggingface_models():
    """
    Search for models on Hugging Face
    """
    try:
        query = request.args.get("query", "")
        task = request.args.get("task", None)
        limit = int(request.args.get("limit", 20))
        
        filter_by = {}
        if task:
            filter_by["task"] = task
        
        hf_api = HuggingFaceAPI()
        models = hf_api.search_models(query, filter_by, limit)
        
        return jsonify({
            "models": models
        })
    
    except Exception as e:
        logger.error(f"Error searching Hugging Face models: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/huggingface/tasks", methods=["GET"])
def get_huggingface_tasks():
    """
    Get available tasks on Hugging Face
    """
    try:
        hf_api = HuggingFaceAPI()
        tasks = hf_api.get_available_tasks()
        
        return jsonify({
            "tasks": tasks
        })
    
    except Exception as e:
        logger.error(f"Error getting Hugging Face tasks: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/huggingface/model/<model_id>", methods=["GET"])
def get_huggingface_model_info(model_id):
    """
    Get information about a specific Hugging Face model
    """
    try:
        hf_api = HuggingFaceAPI()
        model_info = hf_api.get_model_info(model_id)
        
        return jsonify({
            "model_info": model_info
        })
    
    except Exception as e:
        logger.error(f"Error getting Hugging Face model info: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/huggingface/download", methods=["POST"])
def download_huggingface_model():
    """
    Download a model from Hugging Face
    """
    try:
        data = request.json
        
        if not data or "model_id" not in data:
            return jsonify({
                "error": "No model ID provided"
            }), 400
        
        model_id = data["model_id"]
        custom_id = data.get("custom_id", model_id.replace("/", "-"))
        model_class = data.get("model_class", "sequence-classification")
        
        # Download model
        hf_api = HuggingFaceAPI()
        download_path = f"models/downloaded/{custom_id}"
        local_path = hf_api.download_model(model_id, download_path, model_class)
        
        # Update model config
        model_config = ModelConfig.get_config()
        if "models" not in model_config:
            model_config["models"] = {}
        
        model_config["models"][custom_id] = {
            "path": local_path,
            "type": data.get("model_type", "transformer"),
            "source": "huggingface",
            "original_id": model_id
        }
        
        ModelConfig.save_config(model_config)
        
        return jsonify({
            "status": "success",
            "model_id": custom_id,
            "local_path": local_path
        })
    
    except Exception as e:
        logger.error(f"Error downloading Hugging Face model: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/upload-dataset", methods=["POST"])
def upload_dataset():
    """
    Upload a dataset file
    """
    try:
        # Check if pandas is available
        if not pandas_available:
            return jsonify({
                "error": "Pandas is not installed. Dataset functionality is limited."
            }), 500
            
        # Check if file part is in the request
        if "file" not in request.files:
            return jsonify({
                "error": "No file part in the request"
            }), 400
        
        file = request.files["file"]
        
        # Check if a file was selected
        if file.filename == "":
            return jsonify({
                "error": "No file selected"
            }), 400
        
        # Check if file type is allowed
        if not file or not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not allowed. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the file
        file.save(file_path)
        
        # Determine file format
        _, ext = os.path.splitext(filename)
        file_format = ext[1:].lower()  # Remove the dot
        
        # Initialize dataset handler
        dataset_handler = DatasetHandler(file_path, file_format)
        
        # Load dataset to validate it
        if not dataset_handler.load_dataset():
            return jsonify({
                "error": "Error loading dataset"
            }), 500
        
        # Get column information for the UI
        processed_data = dataset_handler.get_processed_data()
        
        # Check if processed data is available
        if processed_data is None:
            return jsonify({
                "status": "warning",
                "filename": filename,
                "file_path": file_path,
                "format": file_format,
                "message": "Dataset was uploaded but could not be processed. Missing required dependencies."
            })
            
        columns = list(processed_data.columns) if hasattr(processed_data, 'columns') else []
        
        # Get sample data (first 5 rows) if possible
        try:
            sample_data = processed_data.head(5).to_dict(orient="records") if hasattr(processed_data, 'head') else []
        except Exception:
            sample_data = []
        
        return jsonify({
            "status": "success",
            "filename": filename,
            "file_path": file_path,
            "format": file_format,
            "columns": columns,
            "sample_data": sample_data
        })
    
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/datasets", methods=["GET"])
def list_datasets():
    """
    List available datasets
    """
    try:
        datasets = []
        
        # List files in the UPLOAD_FOLDER
        for filename in os.listdir(UPLOAD_FOLDER):
            if allowed_file(filename):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                
                # Get file info
                file_info = os.stat(file_path)
                file_size = file_info.st_size
                modified_time = file_info.st_mtime
                
                # Get file format
                _, ext = os.path.splitext(filename)
                file_format = ext[1:].lower()  # Remove the dot
                
                datasets.append({
                    "filename": filename,
                    "file_path": file_path,
                    "format": file_format,
                    "size": file_size,
                    "modified": modified_time
                })
        
        return jsonify({
            "datasets": datasets
        })
    
    except Exception as e:
        logger.error(f"Error listing datasets: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/dataset-info/<filename>", methods=["GET"])
def get_dataset_info(filename):
    """
    Get information about a specific dataset
    """
    try:
        # Check if pandas is available
        if not pandas_available:
            return jsonify({
                "error": "Pandas is not installed. Dataset functionality is limited."
            }), 500
            
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({
                "error": "Dataset file not found"
            }), 404
        
        # Determine file format
        _, ext = os.path.splitext(filename)
        file_format = ext[1:].lower()  # Remove the dot
        
        # Initialize dataset handler
        dataset_handler = DatasetHandler(file_path, file_format)
        
        # Load dataset
        if not dataset_handler.load_dataset():
            return jsonify({
                "error": "Error loading dataset"
            }), 500
        
        # Get information about the dataset
        processed_data = dataset_handler.get_processed_data()
        
        # Check if processed data is available
        if processed_data is None:
            return jsonify({
                "status": "warning",
                "filename": filename,
                "file_path": file_path,
                "format": file_format,
                "message": "Dataset could not be processed. Missing required dependencies."
            })
        
        columns = list(processed_data.columns) if hasattr(processed_data, 'columns') else []
        row_count = len(processed_data) if processed_data is not None else 0
        
        # Get sample data (first 5 rows) if possible
        try:
            sample_data = processed_data.head(5).to_dict(orient="records") if hasattr(processed_data, 'head') else []
        except Exception:
            sample_data = []
        
        return jsonify({
            "filename": filename,
            "file_path": file_path,
            "format": file_format,
            "columns": columns,
            "row_count": row_count,
            "sample_data": sample_data
        })
    
    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/dataset/<filename>", methods=["DELETE"])
def delete_dataset(filename):
    """
    Delete a dataset file
    """
    try:
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({
                "error": "Dataset file not found"
            }), 404
        
        # Delete the file
        os.remove(file_path)
        
        return jsonify({
            "status": "success",
            "message": f"Dataset {filename} deleted successfully"
        })
    
    except Exception as e:
        logger.error(f"Error deleting dataset: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/train", methods=["POST"])
def train_model():
    """
    Train a model
    """
    try:
        # Check if pandas is available
        if not pandas_available:
            return jsonify({
                "error": "Training functionality requires pandas, which is not installed."
            }), 500
            
        data = request.json
        
        if not data:
            return jsonify({
                "error": "No training data provided"
            }), 400
        
        # Required parameters
        required_params = ["model_id", "dataset_filename", "learning_type", "input_field"]
        for param in required_params:
            if param not in data:
                return jsonify({
                    "error": f"Missing required parameter: {param}"
                }), 400
        
        model_id = data["model_id"]
        dataset_filename = data["dataset_filename"]
        learning_type = data["learning_type"]
        input_field = data["input_field"]
        output_field = data.get("output_field")  # Optional for some learning types
        
        # Check if output field is required but not provided
        if learning_type == "supervised" and not output_field:
            return jsonify({
                "error": "Output field is required for supervised learning"
            }), 400
        
        # Load model configuration
        model_config = ModelConfig.get_config()
        if "models" not in model_config or model_id not in model_config["models"]:
            return jsonify({
                "error": f"Model {model_id} not found in configuration"
            }), 404
        
        # Load dataset
        dataset_path = os.path.join(UPLOAD_FOLDER, secure_filename(dataset_filename))
        if not os.path.exists(dataset_path):
            return jsonify({
                "error": f"Dataset file {dataset_filename} not found"
            }), 404
        
        # Determine dataset format
        _, ext = os.path.splitext(dataset_filename)
        dataset_format = ext[1:].lower()  # Remove the dot
        
        # Initialize dataset handler
        dataset_handler = DatasetHandler(dataset_path, dataset_format)
        
        # Load dataset
        if not dataset_handler.load_dataset():
            return jsonify({
                "error": "Error loading dataset"
            }), 500
        
        # Check if the dataset was actually processed (might be None due to missing dependencies)
        if dataset_handler.get_processed_data() is None:
            return jsonify({
                "error": "Dataset could not be processed. Missing required dependencies."
            }), 500
            
        try:
            # Create trainer
            trainer = create_trainer(learning_type, model_id, model_config["models"][model_id])
        except ImportError as e:
            logger.error(f"Missing dependencies for training: {str(e)}")
            return jsonify({
                "error": f"Missing dependencies for training: {str(e)}"
            }), 500
            
        # Prepare training configuration
        training_config = data.get("training_config", {})
        training_config["learning_type"] = learning_type
        
        try:
            trainer.prepare_training_config(training_config)
            
            # Validate training configuration
            validation_errors = trainer.validate_training_config()
            if validation_errors:
                return jsonify({
                    "error": "Invalid training configuration",
                    "validation_errors": validation_errors
                }), 400
            
            # Create a unique job_id for the training process
            job_id = f"job_{int(time.time())}_{model_id}"
            
            # Register the training job with the monitoring system
            from api.monitor import register_training_job
            register_training_job(
                job_id=job_id,
                model_id=model_id,
                dataset=dataset_filename,
                config=training_config
            )
            
            # Start the training in a background thread
            def train_in_background():
                try:
                    from api.monitor import update_training_status, add_log_message, check_stop_requested
                    
                    # Update status to training
                    update_training_status(job_id, status="training", progress=0)
                    
                    # Set the job_id in the trainer so it can report progress
                    trainer.job_id = job_id
                    
                    # Train model based on learning type
                    if learning_type == "supervised":
                        result = trainer.train(dataset_handler, input_field, output_field)
                    else:
                        result = trainer.train(dataset_handler, input_field)
                    
                    # Update status based on result
                    if result.get("status") == "error":
                        update_training_status(
                            job_id, 
                            status="error", 
                            error=result.get("error", "Unknown error during training")
                        )
                    elif result.get("status") == "stopped":
                        # Training was stopped by user
                        update_training_status(
                            job_id, 
                            status="stopped", 
                            progress=result.get("progress", 0)
                        )
                        add_log_message(job_id, result.get("message", "Training stopped by user request"))
                    else:
                        # Training completed successfully
                        update_training_status(
                            job_id, 
                            status="completed", 
                            progress=100
                        )
                        add_log_message(job_id, "Training completed successfully!")
                        
                except Exception as e:
                    # Handle any other exceptions
                    from api.monitor import update_training_status
                    logger.error(f"Error in training thread: {str(e)}")
                    update_training_status(
                        job_id, 
                        status="error", 
                        error=str(e)
                    )
            
            # Start the background thread for training
            training_thread = threading.Thread(target=train_in_background)
            training_thread.daemon = True
            training_thread.start()
            
            # Return success with job_id for monitoring
            return jsonify({
                "status": "success",
                "message": "Training started successfully",
                "job_id": job_id
            })
            
        except ImportError as e:
            logger.error(f"Missing dependencies for training: {str(e)}")
            return jsonify({
                "error": f"Missing dependencies for training: {str(e)}"
            }), 500
    
    except NotImplementedError as e:
        logger.error(f"Training method not implemented: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 501
    
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

@training_bp.route("/api/training/browse-dirs", methods=["GET"])
def browse_directories():
    """
    Browse directories for model selection
    """
    try:
        base_path = request.args.get("path", "models")
        
        # Ensure the path is within the permitted directory
        if not os.path.abspath(base_path).startswith(os.path.abspath(".")):
            return jsonify({
                "error": "Path is outside permitted directory"
            }), 403
        
        # Create base directory if it doesn't exist
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)
        
        # Get directory contents
        dirs = []
        files = []
        
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            
            if os.path.isdir(item_path):
                dirs.append({
                    "name": item,
                    "path": item_path,
                    "type": "directory"
                })
            else:
                files.append({
                    "name": item,
                    "path": item_path,
                    "type": "file",
                    "size": os.path.getsize(item_path)
                })
        
        # Sort directories and files by name
        dirs.sort(key=lambda x: x["name"])
        files.sort(key=lambda x: x["name"])
        
        # Combine directories and files
        contents = dirs + files
        
        return jsonify({
            "path": base_path,
            "contents": contents,
            "parent": os.path.dirname(base_path) if base_path != "." else None
        })
    
    except Exception as e:
        logger.error(f"Error browsing directories: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500