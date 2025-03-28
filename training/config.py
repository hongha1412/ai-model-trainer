"""
Module for training configuration management
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class TrainingConfig:
    """
    Class to manage training configurations
    """
    
    CONFIG_PATH = "config/training_config.json"
    
    # Default configs for different learning types
    DEFAULT_CONFIGS = {
        "supervised": {
            "batch_size": 8,
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "max_seq_length": 128,
            "save_steps": 10000,
            "evaluation_strategy": "epoch"
        },
        "unsupervised": {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_train_epochs": 5,
            "weight_decay": 0.1,
            "warmup_steps": 1000,
            "max_seq_length": 256,
            "save_steps": 10000
        },
        "reinforcement": {
            "batch_size": 64,
            "learning_rate": 5e-5,
            "discount_factor": 0.99,
            "target_update_interval": 1000,
            "replay_buffer_size": 10000,
            "exploration_rate": 0.1,
            "max_steps": 100000
        },
        "semi_supervised": {
            "batch_size": 12,
            "learning_rate": 3e-5,
            "num_train_epochs": 4,
            "weight_decay": 0.05,
            "warmup_steps": 750,
            "max_seq_length": 192,
            "save_steps": 10000,
            "unlabeled_weight": 0.5
        },
        "self_supervised": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_train_epochs": 10,
            "weight_decay": 0.1,
            "warmup_steps": 2000,
            "max_seq_length": 512,
            "save_steps": 5000
        },
        "online": {
            "learning_rate": 1e-3,
            "window_size": 1000,
            "forget_factor": 0.1,
            "update_interval": 100,
            "max_samples": 100000
        },
        "federated": {
            "num_clients": 10,
            "client_fraction": 0.2,
            "local_epochs": 2,
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_rounds": 100,
            "aggregation": "fedavg"
        }
    }
    
    @classmethod
    def save_config(cls, config: Dict[str, Any]) -> bool:
        """
        Save training configuration to file
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(cls.CONFIG_PATH), exist_ok=True)
            
            with open(cls.CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving training configuration: {str(e)}")
            return False
    
    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """
        Load training configuration from file
        
        Returns:
            Training configuration dictionary
        """
        try:
            if not os.path.exists(cls.CONFIG_PATH):
                logger.warning(f"Training configuration file does not exist: {cls.CONFIG_PATH}")
                return {}
            
            with open(cls.CONFIG_PATH, 'r') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading training configuration: {str(e)}")
            return {}
    
    @classmethod
    def get_default_config(cls, learning_type: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific learning type
        
        Args:
            learning_type: Type of learning (supervised, unsupervised, etc.)
            
        Returns:
            Default configuration dictionary for the specified learning type
        """
        if learning_type not in cls.DEFAULT_CONFIGS:
            logger.warning(f"No default configuration available for learning type: {learning_type}")
            return {}
        
        return cls.DEFAULT_CONFIGS[learning_type].copy()
    
    @classmethod
    def merge_with_defaults(cls, config: Dict[str, Any], learning_type: str) -> Dict[str, Any]:
        """
        Merge custom configuration with default configuration
        
        Args:
            config: Custom configuration dictionary
            learning_type: Type of learning (supervised, unsupervised, etc.)
            
        Returns:
            Merged configuration dictionary
        """
        default_config = cls.get_default_config(learning_type)
        merged_config = default_config.copy()
        merged_config.update(config)
        return merged_config
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], learning_type: str) -> List[str]:
        """
        Validate configuration for a specific learning type
        
        Args:
            config: Configuration dictionary to validate
            learning_type: Type of learning (supervised, unsupervised, etc.)
            
        Returns:
            List of validation error messages (empty if validation successful)
        """
        default_config = cls.get_default_config(learning_type)
        errors = []
        
        # Check required parameters for each learning type
        if learning_type == "supervised":
            required_params = ["batch_size", "learning_rate", "num_train_epochs"]
        elif learning_type == "unsupervised":
            required_params = ["batch_size", "learning_rate", "num_train_epochs"]
        elif learning_type == "reinforcement":
            required_params = ["learning_rate", "discount_factor", "exploration_rate"]
        elif learning_type == "semi_supervised":
            required_params = ["batch_size", "learning_rate", "num_train_epochs", "unlabeled_weight"]
        elif learning_type == "self_supervised":
            required_params = ["batch_size", "learning_rate", "num_train_epochs"]
        elif learning_type == "online":
            required_params = ["learning_rate", "window_size", "update_interval"]
        elif learning_type == "federated":
            required_params = ["num_clients", "client_fraction", "local_epochs", "num_rounds"]
        else:
            errors.append(f"Unsupported learning type: {learning_type}")
            return errors
        
        # Check if required parameters are present
        for param in required_params:
            if param not in config:
                errors.append(f"Missing required parameter for {learning_type} learning: {param}")
        
        # Validate parameter types and ranges
        for param, value in config.items():
            if param == "batch_size" and (not isinstance(value, int) or value <= 0):
                errors.append(f"batch_size must be a positive integer: {value}")
            elif param == "learning_rate" and (not isinstance(value, (int, float)) or value <= 0):
                errors.append(f"learning_rate must be a positive number: {value}")
            elif param == "num_train_epochs" and (not isinstance(value, int) or value <= 0):
                errors.append(f"num_train_epochs must be a positive integer: {value}")
            elif param == "discount_factor" and (not isinstance(value, (int, float)) or value < 0 or value > 1):
                errors.append(f"discount_factor must be between 0 and 1: {value}")
            elif param == "exploration_rate" and (not isinstance(value, (int, float)) or value < 0 or value > 1):
                errors.append(f"exploration_rate must be between 0 and 1: {value}")
        
        return errors