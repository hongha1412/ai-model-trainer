from typing import Dict, Any, Optional

def validate_completion_request(request_data: Dict[str, Any]) -> Optional[str]:
    """
    Validate a completion request
    
    Args:
        request_data: The request data to validate
    
    Returns:
        Error message if validation fails, None otherwise
    """
    # Check required fields
    if not request_data.get('model'):
        return "Model is required"
    
    # Validate types
    if 'prompt' in request_data and not isinstance(request_data['prompt'], str):
        return "Prompt must be a string"
    
    if 'max_tokens' in request_data:
        if not isinstance(request_data['max_tokens'], int):
            return "max_tokens must be an integer"
        if request_data['max_tokens'] <= 0:
            return "max_tokens must be a positive integer"
    
    if 'temperature' in request_data:
        if not isinstance(request_data['temperature'], (int, float)):
            return "temperature must be a number"
        if request_data['temperature'] < 0:
            return "temperature must be non-negative"
    
    if 'top_p' in request_data:
        if not isinstance(request_data['top_p'], (int, float)):
            return "top_p must be a number"
        if request_data['top_p'] <= 0 or request_data['top_p'] > 1:
            return "top_p must be between 0 and 1"
    
    return None

def validate_model_id(model_id: str) -> Optional[str]:
    """
    Validate a model ID
    
    Args:
        model_id: The model ID to validate
    
    Returns:
        Error message if validation fails, None otherwise
    """
    if not model_id:
        return "Model ID is required"
    
    # Check if model exists in configuration
    from models.config import ModelConfig
    
    config = ModelConfig.get_config()
    if 'models' not in config or model_id not in config['models']:
        return f"Model '{model_id}' not found in configuration"
    
    return None
