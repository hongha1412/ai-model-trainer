"""
Utility module for checking and installing dependencies
"""
import importlib
import logging
import subprocess
import sys
import os
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Environment variable to disable automatic installation
DISABLE_AUTO_INSTALL = os.environ.get("DISABLE_AUTO_INSTALL", "").lower() in ("1", "true", "yes")

DEPENDENCIES = {
    "pandas": {
        "import_name": "pandas",
        "package_name": "pandas",
        "functions": []
    },
    "sklearn": {
        "import_name": "sklearn",
        "package_name": "scikit-learn",
        "functions": ["model_selection"]
    },
    "torch": {
        "import_name": "torch",
        "package_name": "torch",
        "functions": [],
        "optional": True  # Mark torch as optional due to size
    },
    "transformers": {
        "import_name": "transformers",
        "package_name": "transformers",
        "functions": ["AutoTokenizer", "AutoModelForSeq2SeqLM", "TrainingArguments", "Trainer"],
        "optional": True  # Mark transformers as optional due to size
    }
}

def check_and_install_dependency(dependency_key: str) -> bool:
    """
    Checks if a dependency is installed, and if not, tries to install it
    
    Args:
        dependency_key: Key for the dependency in the DEPENDENCIES dictionary
    
    Returns:
        True if dependency is available (was already installed or was successfully installed), 
        False otherwise
    """
    dependency = DEPENDENCIES.get(dependency_key)
    if not dependency:
        logger.error(f"Unknown dependency: {dependency_key}")
        return False
    
    import_name = dependency["import_name"]
    package_name = dependency["package_name"]
    required_functions = dependency["functions"]
    is_optional = dependency.get("optional", False)
    
    # Skip installation attempt for PyTorch and Transformers 
    # due to disk space constraints on Replit
    if package_name in ["torch", "transformers"]:
        logger.info(f"Skipping {package_name} installation check due to disk space constraints.")
        if is_optional:
            logger.warning(f"{package_name} is marked as optional. Continuing without it.")
            return False
        return False
    
    try:
        # Try to import the module
        module = importlib.import_module(import_name)
        
        # Check for required functions/classes
        for func in required_functions:
            try:
                if "." in func:
                    # Handle nested imports like "model_selection" in sklearn
                    parts = func.split(".")
                    sub_module = module
                    for part in parts:
                        sub_module = getattr(sub_module, part)
                else:
                    # Direct attribute access
                    getattr(module, func)
            except (AttributeError, ImportError):
                logger.warning(f"Function {func} not found in {import_name}")
                raise ImportError(f"Function {func} not found in {import_name}")
        
        return True
                
    except (ImportError, ModuleNotFoundError):
        if DISABLE_AUTO_INSTALL:
            logger.info(f"{package_name} is not installed and auto-install is disabled.")
            return False
            
        logger.info(f"{package_name} is not installed. Attempting to install...")
        
        try:
            # Installation for other dependencies
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            logger.info(f"Successfully installed {package_name}")
            
            # Verify installation
            try:
                module = importlib.import_module(import_name)
                for func in required_functions:
                    if "." in func:
                        parts = func.split(".")
                        sub_module = module
                        for part in parts:
                            sub_module = getattr(sub_module, part)
                    else:
                        getattr(module, func)
                return True
            except (ImportError, AttributeError):
                logger.error(f"Installed {package_name} but still can't import {import_name} or its functions")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_name}: {str(e)}")
            if is_optional:
                logger.warning(f"{package_name} is marked as optional. Continuing without it.")
            return False
        except OSError as e:
            if "Disk quota exceeded" in str(e):
                logger.error(f"Failed to install {package_name} due to disk quota exceeded.")
                if is_optional:
                    logger.warning(f"{package_name} is marked as optional. Continuing without it.")
                logger.info("To disable automatic installation, set DISABLE_AUTO_INSTALL=1")
            else:
                logger.error(f"OS error installing {package_name}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error installing {package_name}: {str(e)}")
            return False

def import_optional_dependency(dependency_key: str) -> Optional[Any]:
    """
    Import an optional dependency, install if possible, return None if not available
    
    Args:
        dependency_key: Key for the dependency in the DEPENDENCIES dictionary
    
    Returns:
        Imported module or None if unavailable
    """
    try:
        # Try to ensure dependency is installed
        if check_and_install_dependency(dependency_key):
            # Import and return the module
            dependency = DEPENDENCIES.get(dependency_key)
            if dependency:
                return importlib.import_module(dependency["import_name"])
        
        return None
    except Exception as e:
        logger.error(f"Error importing {dependency_key}: {str(e)}")
        return None