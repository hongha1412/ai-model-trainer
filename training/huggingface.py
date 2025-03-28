"""
Module for Hugging Face integration
"""
import os
import logging
from typing import Dict, List, Any, Optional, Union
import json
import requests

# Try to import transformers, but provide a fallback
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    transformers_available = True
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    AutoModelForSequenceClassification = None
    transformers_available = False
    print("WARNING: Transformers is not installed. HuggingFace model functionality will be limited.")

logger = logging.getLogger(__name__)

class HuggingFaceAPI:
    """
    Class for interacting with the Hugging Face API
    """
    
    API_URL = "https://huggingface.co/api"
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the Hugging Face API client
        
        Args:
            api_token: Hugging Face API token
        """
        self.api_token = api_token or os.environ.get("HUGGINGFACE_API_TOKEN")
        self.headers = {}
        
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"
    
    def search_models(self, query: str, filter_by: Optional[Dict[str, Any]] = None, 
                     limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for models on Hugging Face
        
        Args:
            query: Search query
            filter_by: Filter criteria (e.g., {"task": "text-classification"})
            limit: Maximum number of results
            
        Returns:
            List of model information dictionaries
        """
        try:
            params = {
                "search": query,
                "limit": limit,
            }
            
            if filter_by:
                for key, value in filter_by.items():
                    params[key] = value
            
            response = requests.get(
                f"{self.API_URL}/models",
                headers=self.headers,
                params=params
            )
            
            response.raise_for_status()
            
            models = response.json()
            
            # Format the response for easier use
            formatted_models = []
            for model in models:
                formatted_models.append({
                    "id": model.get("modelId"),
                    "name": model.get("modelId").split("/")[-1] if model.get("modelId") else "",
                    "author": model.get("modelId").split("/")[0] if model.get("modelId") else "",
                    "tags": model.get("tags", []),
                    "downloads": model.get("downloads", 0),
                    "likes": model.get("likes", 0),
                    "task": next((tag for tag in model.get("tags", []) if tag.startswith("task:")), ""),
                    "url": f"https://huggingface.co/{model.get('modelId')}",
                })
            
            return formatted_models
        
        except Exception as e:
            logger.error(f"Error searching for models: {str(e)}")
            return []
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_id: Hugging Face model ID
            
        Returns:
            Model information dictionary
        """
        try:
            response = requests.get(
                f"{self.API_URL}/models/{model_id}",
                headers=self.headers
            )
            
            response.raise_for_status()
            
            model_info = response.json()
            
            # Format the response for easier use
            formatted_info = {
                "id": model_info.get("modelId"),
                "name": model_info.get("modelId").split("/")[-1] if model_info.get("modelId") else "",
                "author": model_info.get("modelId").split("/")[0] if model_info.get("modelId") else "",
                "tags": model_info.get("tags", []),
                "downloads": model_info.get("downloads", 0),
                "likes": model_info.get("likes", 0),
                "task": next((tag for tag in model_info.get("tags", []) if tag.startswith("task:")), ""),
                "url": f"https://huggingface.co/{model_info.get('modelId')}",
                "pipeline_tag": model_info.get("pipeline_tag"),
                "siblings": [sibling.get("rfilename") for sibling in model_info.get("siblings", [])],
            }
            
            return formatted_info
        
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {}
    
    def download_model(self, model_id: str, local_dir: str, model_class: str = "sequence-classification") -> str:
        """
        Download model from Hugging Face to local directory
        
        Args:
            model_id: Hugging Face model ID
            local_dir: Local directory to save model
            model_class: Type of model to download (e.g., "sequence-classification")
            
        Returns:
            Path to the downloaded model
        """
        try:
            # Check if transformers is available
            if not transformers_available:
                error_msg = "Transformers library is not installed. Cannot download model."
                logger.error(error_msg)
                
                # Create directory and save model info with error
                os.makedirs(local_dir, exist_ok=True)
                model_info = self.get_model_info(model_id)
                model_info["error"] = error_msg
                with open(os.path.join(local_dir, "model_info.json"), "w") as f:
                    json.dump(model_info, f, indent=2)
                
                # We still return the path, but the model won't be usable
                return local_dir
            
            os.makedirs(local_dir, exist_ok=True)
            
            logger.info(f"Downloading model {model_id} to {local_dir}")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(local_dir)
            
            # Download model
            if model_class == "sequence-classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
            else:
                model = AutoModel.from_pretrained(model_id)
            
            model.save_pretrained(local_dir)
            
            # Save model info
            model_info = self.get_model_info(model_id)
            with open(os.path.join(local_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"Model downloaded successfully to {local_dir}")
            
            return local_dir
        
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            
            # Create directory and save model info with error
            try:
                os.makedirs(local_dir, exist_ok=True)
                model_info = self.get_model_info(model_id)
                model_info["error"] = str(e)
                with open(os.path.join(local_dir, "model_info.json"), "w") as f:
                    json.dump(model_info, f, indent=2)
            except Exception as inner_e:
                logger.error(f"Error saving model info: {str(inner_e)}")
            
            raise
    
    def get_available_tasks(self) -> List[str]:
        """
        Get list of available tasks on Hugging Face
        
        Returns:
            List of task names
        """
        return [
            "text-classification",
            "token-classification",
            "question-answering",
            "translation",
            "summarization",
            "text-generation",
            "fill-mask",
            "sentence-similarity",
            "feature-extraction",
            "text-to-image",
            "image-classification",
            "object-detection",
            "image-segmentation",
            "audio-classification",
            "automatic-speech-recognition",
            "text-to-speech"
        ]