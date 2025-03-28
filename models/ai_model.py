import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Global model registry for loaded models
loaded_models: Dict[str, Any] = {}
model_info: Dict[str, Dict[str, Any]] = {}

def register_model(model_id: str, model_instance: Any, model_type: str) -> None:
    """
    Register a model in the global registry
    
    Args:
        model_id: Unique identifier for the model
        model_instance: The loaded model instance
        model_type: Type of the model (e.g., 't5', 'gpt2')
    """
    loaded_models[model_id] = model_instance
    model_info[model_id] = {
        'id': model_id,
        'type': model_type,
        'loaded': True,
        'status': 'active'
    }
    logger.info(f"Model {model_id} registered successfully")

def unregister_model(model_id: str) -> None:
    """
    Unregister a model from the global registry
    
    Args:
        model_id: Unique identifier for the model
    """
    if model_id in loaded_models:
        del loaded_models[model_id]
    
    if model_id in model_info:
        model_info[model_id]['loaded'] = False
        model_info[model_id]['status'] = 'unloaded'
    
    logger.info(f"Model {model_id} unregistered")

def get_model(model_id: str) -> Optional[Any]:
    """
    Get a model from the registry
    
    Args:
        model_id: Unique identifier for the model
    
    Returns:
        The model instance or None if not found
    """
    return loaded_models.get(model_id)

def get_loaded_model_ids() -> List[str]:
    """
    Get IDs of all loaded models
    
    Returns:
        List of model IDs
    """
    return list(loaded_models.keys())

def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific model
    
    Args:
        model_id: Unique identifier for the model
    
    Returns:
        Dictionary with model information or None if not found
    """
    if model_id in model_info:
        return model_info[model_id]
    
    # Check if model exists in config but is not loaded
    from models.config import ModelConfig
    config = ModelConfig.get_config()
    if 'models' in config and model_id in config['models']:
        model_config = config['models'][model_id]
        return {
            'id': model_id,
            'type': model_config['type'],
            'path': model_config['path'],
            'loaded': False,
            'status': 'not_loaded'
        }
    
    return None

def run_inference(model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Run inference on a model
    
    Args:
        model_id: Unique identifier for the model
        prompt: The input text prompt
        **kwargs: Additional parameters for inference
    
    Returns:
        Dictionary containing inference results
    
    Raises:
        ValueError: If model is not found or inference fails
    """
    model = get_model(model_id)
    if not model:
        raise ValueError(f"Model {model_id} not found or not loaded")
    
    model_type = model_info[model_id]['type']
    
    try:
        if model_type == 't5':
            # T5 model inference
            max_length = kwargs.get('max_length', 100)
            temperature = kwargs.get('temperature', 1.0)
            top_p = kwargs.get('top_p', 1.0)
            
            # Process the input
            inputs = model.tokenizer(prompt, return_tensors="pt")
            
            # Generate outputs
            outputs = model.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            
            # Decode the response
            response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # For demo purposes, augment the response with information about the prompt
            if len(response) < 10:  # If dummy response is too short
                words = prompt.split()
                if words:
                    response += f" Your input contained {len(words)} words."
            
            # Format the response in OpenAI-compatible format
            return {
                'id': f"cmpl-{model_id}",
                'model': model_id,
                'choices': [
                    {
                        'text': response,
                        'index': 0,
                        'finish_reason': 'stop'
                    }
                ],
                'usage': {
                    'prompt_tokens': len(inputs["input_ids"][0]),
                    'completion_tokens': len(outputs[0]),
                    'total_tokens': len(inputs["input_ids"][0]) + len(outputs[0])
                }
            }
        else:
            # Default generic inference
            raise ValueError(f"Model type {model_type} inference not implemented")
    
    except Exception as e:
        logger.error(f"Inference error for model {model_id}: {str(e)}")
        raise ValueError(f"Inference failed: {str(e)}")
