import os
import json
from flask import Blueprint, jsonify, render_template

openapi_bp = Blueprint('openapi', __name__)

@openapi_bp.route('/openapi.json', methods=['GET'])
def get_openapi_spec():
    """Return the OpenAPI specification as JSON"""
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "AI Model Server API",
            "description": "OpenAI-compatible API endpoints for custom T5 model inference",
            "version": "1.0.0"
        },
        "servers": [
            {
                "url": "/api/v1",
                "description": "Main API"
            }
        ],
        "paths": {
            "/models": {
                "get": {
                    "summary": "List available models",
                    "description": "Returns a list of available models",
                    "operationId": "listModels",
                    "responses": {
                        "200": {
                            "description": "List of models",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ModelList"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/models/{model_id}": {
                "get": {
                    "summary": "Get model information",
                    "description": "Returns information about a specific model",
                    "operationId": "retrieveModel",
                    "parameters": [
                        {
                            "name": "model_id",
                            "in": "path",
                            "description": "The ID of the model to retrieve",
                            "required": True,
                            "schema": {
                                "type": "string"
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Model information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Model"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Model not found",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/completions": {
                "post": {
                    "summary": "Create a completion",
                    "description": "Creates a completion for the provided prompt and parameters",
                    "operationId": "createCompletion",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/CompletionRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Completion response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/CompletionResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Model not found",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/chat/completions": {
                "post": {
                    "summary": "Create a chat completion",
                    "description": "Creates a completion for the chat messages",
                    "operationId": "createChatCompletion",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ChatCompletionRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Chat completion response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ChatCompletionResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Model not found",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/Error"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "ModelList": {
                    "type": "object",
                    "properties": {
                        "object": {
                            "type": "string",
                            "enum": ["list"]
                        },
                        "data": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/Model"
                            }
                        }
                    }
                },
                "Model": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Model identifier"
                        },
                        "object": {
                            "type": "string",
                            "enum": ["model"]
                        },
                        "created": {
                            "type": "integer",
                            "description": "Unix timestamp (in seconds) when this model was created"
                        },
                        "owned_by": {
                            "type": "string",
                            "description": "Who owns the model"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["active", "not_loaded"],
                            "description": "Current status of the model"
                        }
                    }
                },
                "CompletionRequest": {
                    "type": "object",
                    "required": ["model"],
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "ID of the model to use"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to generate completions for"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "The maximum number of tokens to generate",
                            "default": 100
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature to use",
                            "default": 1.0,
                            "minimum": 0,
                            "maximum": 2
                        },
                        "top_p": {
                            "type": "number",
                            "description": "Nucleus sampling parameter",
                            "default": 1.0,
                            "minimum": 0,
                            "maximum": 1
                        }
                    }
                },
                "CompletionResponse": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for this completion"
                        },
                        "object": {
                            "type": "string",
                            "enum": ["text_completion"]
                        },
                        "created": {
                            "type": "integer",
                            "description": "Unix timestamp (in seconds) when this completion was created"
                        },
                        "model": {
                            "type": "string",
                            "description": "The model used for this completion"
                        },
                        "choices": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "The generated text"
                                    },
                                    "index": {
                                        "type": "integer",
                                        "description": "The index of this choice"
                                    },
                                    "finish_reason": {
                                        "type": "string",
                                        "description": "The reason why generation stopped"
                                    }
                                }
                            }
                        },
                        "usage": {
                            "type": "object",
                            "properties": {
                                "prompt_tokens": {
                                    "type": "integer",
                                    "description": "Number of tokens in the prompt"
                                },
                                "completion_tokens": {
                                    "type": "integer",
                                    "description": "Number of tokens in the completion"
                                },
                                "total_tokens": {
                                    "type": "integer",
                                    "description": "Total number of tokens used"
                                }
                            }
                        }
                    }
                },
                "ChatCompletionRequest": {
                    "type": "object",
                    "required": ["model", "messages"],
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "ID of the model to use"
                        },
                        "messages": {
                            "type": "array",
                            "description": "A list of messages comprising the conversation so far",
                            "items": {
                                "type": "object",
                                "required": ["role", "content"],
                                "properties": {
                                    "role": {
                                        "type": "string",
                                        "description": "The role of the message author",
                                        "enum": ["system", "user", "assistant"]
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The content of the message"
                                    }
                                }
                            }
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "The maximum number of tokens to generate",
                            "default": 100
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature to use",
                            "default": 1.0,
                            "minimum": 0,
                            "maximum": 2
                        },
                        "top_p": {
                            "type": "number",
                            "description": "Nucleus sampling parameter",
                            "default": 1.0,
                            "minimum": 0,
                            "maximum": 1
                        }
                    }
                },
                "ChatCompletionResponse": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for this chat completion"
                        },
                        "object": {
                            "type": "string",
                            "enum": ["chat.completion"]
                        },
                        "created": {
                            "type": "integer",
                            "description": "Unix timestamp (in seconds) when this completion was created"
                        },
                        "model": {
                            "type": "string",
                            "description": "The model used for this completion"
                        },
                        "choices": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "message": {
                                        "type": "object",
                                        "properties": {
                                            "role": {
                                                "type": "string",
                                                "description": "The role of the message author",
                                                "enum": ["assistant"]
                                            },
                                            "content": {
                                                "type": "string",
                                                "description": "The content of the message"
                                            }
                                        }
                                    },
                                    "index": {
                                        "type": "integer",
                                        "description": "The index of this choice"
                                    },
                                    "finish_reason": {
                                        "type": "string",
                                        "description": "The reason why generation stopped"
                                    }
                                }
                            }
                        },
                        "usage": {
                            "type": "object",
                            "properties": {
                                "prompt_tokens": {
                                    "type": "integer",
                                    "description": "Number of tokens in the prompt"
                                },
                                "completion_tokens": {
                                    "type": "integer",
                                    "description": "Number of tokens in the completion"
                                },
                                "total_tokens": {
                                    "type": "integer",
                                    "description": "Total number of tokens used"
                                }
                            }
                        }
                    }
                },
                "Error": {
                    "type": "object",
                    "properties": {
                        "error": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "Error message"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "Error type"
                                },
                                "code": {
                                    "type": "string",
                                    "description": "Error code"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return jsonify(spec)

@openapi_bp.route('/docs', methods=['GET'])
def docs():
    """Render OpenAPI documentation page"""
    return render_template('openapi.html')