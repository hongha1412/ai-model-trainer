import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelConfig:
    """Class to manage model configurations"""
    
    CONFIG_PATH = "config/model_config.json"
    _config: Dict[str, Any] = None
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """
        Get the current configuration
        
        Returns:
            Dictionary containing configuration
        """
        if cls._config is None:
            cls.load_default_config()
        return cls._config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any]) -> None:
        """
        Save configuration to file
        
        Args:
            config: Configuration dictionary to save
        """
        cls._config = config
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.CONFIG_PATH), exist_ok=True)
        
        try:
            with open(cls.CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration saved to {cls.CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """
        Load configuration from file
        
        Returns:
            Dictionary containing configuration
        """
        try:
            if os.path.exists(cls.CONFIG_PATH):
                with open(cls.CONFIG_PATH, 'r') as f:
                    cls._config = json.load(f)
                logger.info(f"Configuration loaded from {cls.CONFIG_PATH}")
            else:
                cls.load_default_config()
                logger.info("Default configuration loaded")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            cls.load_default_config()
        
        return cls._config
    
    @classmethod
    def load_default_config(cls) -> None:
        """Load and save default configuration"""
        default_config = {
            "api_version": "v1",
            "models": {},
            "default_model": None,
            "server": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": True
            }
        }
        
        cls._config = default_config
        
        # Save default config if config file doesn't exist
        if not os.path.exists(cls.CONFIG_PATH):
            cls.save_config(default_config)
