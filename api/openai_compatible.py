from flask import Blueprint, request, jsonify
import logging
import json
import time
from typing import Dict, Any, Optional

from models.ai_model import run_inference, get_model, get_model_info
from utils.validation import validate_completion_request, validate_model_id

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

@api_bp.route('/models', methods=['GET'])
def list_models():
    """
    API endpoint to list available models
    Compatible with OpenAI's /models endpoint
    """
    from models.config import ModelConfig
    
    config = ModelConfig.get_config()
    models_list = []
    
    for model_id, model_data in config['models'].items():
        model_info = get_model_info(model_id)
        models_list.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "user",
            "permission": [{"id": "modelperm-" + model_id, "object": "model_permission", "created": int(time.time()), "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": True, "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False, "organization": "*", "group": None, "is_blocking": False}],
            "root": model_id,
            "parent": None,
            "status": "active" if model_info and model_info.get('loaded', False) else "not_loaded"
        })
    
    return jsonify({
        "object": "list",
        "data": models_list
    })

@api_bp.route('/models/<model_id>', methods=['GET'])
def retrieve_model(model_id):
    """
    API endpoint to retrieve a specific model's information
    Compatible with OpenAI's /models/{model_id} endpoint
    """
    model_info = get_model_info(model_id)
    
    if not model_info:
        return jsonify({
            "error": {
                "message": f"Model '{model_id}' not found",
                "type": "invalid_request_error",
                "code": "model_not_found"
            }
        }), 404
    
    return jsonify({
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "user",
        "permission": [{"id": "modelperm-" + model_id, "object": "model_permission", "created": int(time.time()), "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": True, "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False, "organization": "*", "group": None, "is_blocking": False}],
        "root": model_id,
        "parent": None,
        "status": "active" if model_info.get('loaded', False) else "not_loaded"
    })

@api_bp.route('/completions', methods=['POST'])
def create_completion():
    """
    API endpoint for text completion
    Compatible with OpenAI's /completions endpoint
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                "error": {
                    "message": "Request body is required",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }), 400
        
        validation_error = validate_completion_request(request_data)
        if validation_error:
            return jsonify({
                "error": {
                    "message": validation_error,
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }), 400
        
        # Extract parameters
        model_id = request_data.get('model')
        prompt = request_data.get('prompt', '')
        max_tokens = request_data.get('max_tokens', 100)
        temperature = request_data.get('temperature', 1.0)
        top_p = request_data.get('top_p', 1.0)
        
        # Validate model
        model_validation_error = validate_model_id(model_id)
        if model_validation_error:
            return jsonify({
                "error": {
                    "message": model_validation_error,
                    "type": "invalid_request_error",
                    "code": "model_not_found"
                }
            }), 404
        
        # Check if model is loaded
        model = get_model(model_id)
        if not model:
            return jsonify({
                "error": {
                    "message": f"Model {model_id} is not loaded. Load the model first.",
                    "type": "invalid_request_error",
                    "code": "model_not_loaded"
                }
            }), 400
        
        # Run inference
        result = run_inference(
            model_id=model_id,
            prompt=prompt,
            max_length=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Add OpenAI compatible fields
        result.update({
            "object": "text_completion",
            "created": int(time.time())
        })
        
        return jsonify(result)
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            "error": {
                "message": str(e),
                "type": "invalid_request_error",
                "code": "invalid_request"
            }
        }), 400
    
    except Exception as e:
        logger.error(f"Completion error: {str(e)}")
        return jsonify({
            "error": {
                "message": f"An error occurred during completion: {str(e)}",
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500

@api_bp.route('/chat/completions', methods=['POST'])
def create_chat_completion():
    """
    API endpoint for chat completion
    Compatible with OpenAI's /chat/completions endpoint
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                "error": {
                    "message": "Request body is required",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }), 400
        
        # Extract parameters
        model_id = request_data.get('model')
        messages = request_data.get('messages', [])
        max_tokens = request_data.get('max_tokens', 100)
        temperature = request_data.get('temperature', 1.0)
        top_p = request_data.get('top_p', 1.0)
        
        # Validate model
        if not model_id:
            return jsonify({
                "error": {
                    "message": "Model ID is required",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }), 400
        
        # Validate messages
        if not messages or not isinstance(messages, list) or len(messages) == 0:
            return jsonify({
                "error": {
                    "message": "Messages are required and must be a non-empty array",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }), 400
        
        # Check if model is loaded
        model = get_model(model_id)
        if not model:
            return jsonify({
                "error": {
                    "message": f"Model {model_id} is not loaded. Load the model first.",
                    "type": "invalid_request_error",
                    "code": "model_not_loaded"
                }
            }), 400
        
        # Format messages into a prompt
        # This is a simple implementation that concatenates messages
        prompt = ""
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            prompt += f"{role}: {content}\n"
        
        # Run inference
        completion_result = run_inference(
            model_id=model_id,
            prompt=prompt,
            max_length=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Format result as a chat completion
        result = {
            "id": completion_result['id'].replace('cmpl-', 'chatcmpl-'),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_result['choices'][0]['text']
                    },
                    "finish_reason": completion_result['choices'][0]['finish_reason']
                }
            ],
            "usage": completion_result['usage']
        }
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        return jsonify({
            "error": {
                "message": f"An error occurred during chat completion: {str(e)}",
                "type": "server_error",
                "code": "internal_error"
            }
        }), 500
