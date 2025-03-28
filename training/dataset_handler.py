"""
Module for handling different types of datasets for model training
"""
import os
import json
import csv
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

# Try to import pandas, but provide a fallback
try:
    import pandas as pd
except ImportError:
    pd = None
    print("WARNING: pandas is not installed. Some dataset functionality will be limited.")

# Try to import sklearn, but provide a fallback
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None
    print("WARNING: scikit-learn is not installed. Train-test split functionality will be limited.")

logger = logging.getLogger(__name__)

class DatasetHandler:
    """
    Class to handle various types of datasets for model training
    Supports plaintext, JSON, and CSV formats
    """
    
    SUPPORTED_FORMATS = ["text", "json", "csv"]
    
    def __init__(self, dataset_path: str = None, dataset_format: str = None, dataset_content: Any = None):
        """
        Initialize dataset handler
        
        Args:
            dataset_path: Path to the dataset file
            dataset_format: Format of the dataset (text, json, csv)
            dataset_content: Raw content of the dataset
        """
        self.dataset_path = dataset_path
        self.dataset_format = dataset_format
        self.dataset_content = dataset_content
        self.processed_data = None
        
        # Auto-detect format if not specified
        if dataset_path and not dataset_format:
            self.dataset_format = self._detect_format(dataset_path)
        
    def _detect_format(self, file_path: str) -> str:
        """
        Detect format of the dataset based on file extension
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Detected format (text, json, csv)
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.json':
            return 'json'
        elif ext == '.csv':
            return 'csv'
        elif ext in ['.txt', '.text']:
            return 'text'
        else:
            # Default to text
            return 'text'
    
    def load_dataset(self) -> bool:
        """
        Load dataset from file or content
        
        Returns:
            True if dataset was loaded successfully, False otherwise
        """
        try:
            # Check if pandas is required but not available
            if self.dataset_format == 'csv' and pd is None:
                logger.error("Pandas is required for CSV dataset processing")
                return False
                
            if self.dataset_content is not None:
                return self._process_content()
            
            if not self.dataset_path or not os.path.exists(self.dataset_path):
                logger.error(f"Dataset path does not exist: {self.dataset_path}")
                return False
            
            if self.dataset_format == 'json':
                with open(self.dataset_path, 'r') as f:
                    self.dataset_content = json.load(f)
            elif self.dataset_format == 'csv':
                if pd is None:
                    logger.error("Pandas is required for CSV dataset processing")
                    return False
                self.dataset_content = pd.read_csv(self.dataset_path)
            elif self.dataset_format == 'text':
                with open(self.dataset_path, 'r') as f:
                    self.dataset_content = f.read()
            else:
                logger.error(f"Unsupported dataset format: {self.dataset_format}")
                return False
            
            return self._process_content()
        
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
    
    def _process_content(self) -> bool:
        """
        Process the loaded content based on its format
        
        Returns:
            True if content was processed successfully, False otherwise
        """
        try:
            # Check if pandas is available
            if pd is None:
                logger.error("Pandas is required for dataset processing")
                self.processed_data = None
                return False
                
            if self.dataset_format == 'json':
                # Convert to pandas DataFrame for easier processing
                if isinstance(self.dataset_content, list):
                    self.processed_data = pd.DataFrame(self.dataset_content)
                elif isinstance(self.dataset_content, dict):
                    # If it's a dict with records, convert to DataFrame
                    if 'data' in self.dataset_content and isinstance(self.dataset_content['data'], list):
                        self.processed_data = pd.DataFrame(self.dataset_content['data'])
                    else:
                        # Dictionary of lists/values
                        self.processed_data = pd.DataFrame(self.dataset_content)
            
            elif self.dataset_format == 'csv':
                # Already in DataFrame format if read with pandas
                if isinstance(self.dataset_content, pd.DataFrame):
                    self.processed_data = self.dataset_content
                else:
                    # If it's raw content, parse it
                    import io
                    self.processed_data = pd.read_csv(io.StringIO(self.dataset_content))
            
            elif self.dataset_format == 'text':
                # For text data, split by lines and create a simple DataFrame
                if isinstance(self.dataset_content, str):
                    lines = self.dataset_content.strip().split('\n')
                    self.processed_data = pd.DataFrame({'text': lines})
            
            return True
        
        except Exception as e:
            logger.error(f"Error processing dataset content: {str(e)}")
            return False
    
    def get_processed_data(self) -> Any:
        """
        Get the processed dataset
        
        Returns:
            Processed dataset as pandas DataFrame or None if processing failed
        """
        return self.processed_data
    
    def get_train_test_split(self, test_ratio: float = 0.2) -> Tuple[Any, Any]:
        """
        Split dataset into training and testing sets
        
        Args:
            test_ratio: Ratio of test data (0.0 to 1.0)
            
        Returns:
            Tuple of (train_data, test_data) as pandas DataFrames
        """
        if self.processed_data is None:
            logger.error("Dataset not processed yet")
            return None, None
            
        # Check if required dependencies are available
        if train_test_split is None:
            logger.error("scikit-learn is required for train-test split functionality")
            return None, None
        
        try:
            train_data, test_data = train_test_split(self.processed_data, test_size=test_ratio, random_state=42)
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error performing train-test split: {str(e)}")
            return None, None
    
    def convert_to_training_format(self, input_field: str, output_field: Optional[str] = None, 
                                   learning_type: str = "supervised") -> Dict[str, Any]:
        """
        Convert processed data to format suitable for model training
        
        Args:
            input_field: Column name for input data
            output_field: Column name for output data (for supervised learning)
            learning_type: Type of learning (supervised, unsupervised, etc.)
            
        Returns:
            Dictionary with data formatted for training
        """
        if self.processed_data is None:
            logger.error("Dataset not processed yet")
            return {}
        
        result = {}
        
        if learning_type == "supervised":
            if not output_field:
                logger.error("Output field required for supervised learning")
                return {}
            
            if input_field not in self.processed_data.columns:
                logger.error(f"Input field '{input_field}' not found in dataset")
                return {}
                
            if output_field not in self.processed_data.columns:
                logger.error(f"Output field '{output_field}' not found in dataset")
                return {}
            
            result["inputs"] = self.processed_data[input_field].tolist()
            result["outputs"] = self.processed_data[output_field].tolist()
            
        elif learning_type == "unsupervised":
            if input_field not in self.processed_data.columns:
                logger.error(f"Input field '{input_field}' not found in dataset")
                return {}
                
            result["inputs"] = self.processed_data[input_field].tolist()
            
        elif learning_type == "reinforcement":
            # For reinforcement learning, we typically need state, action, reward
            required_fields = ["state", "action", "reward", "next_state"]
            for field in required_fields:
                if field not in self.processed_data.columns:
                    logger.error(f"Required field '{field}' for reinforcement learning not found in dataset")
                    return {}
            
            result["states"] = self.processed_data["state"].tolist()
            result["actions"] = self.processed_data["action"].tolist()
            result["rewards"] = self.processed_data["reward"].tolist()
            result["next_states"] = self.processed_data["next_state"].tolist()
            
        else:
            logger.error(f"Unsupported learning type: {learning_type}")
            return {}
        
        return result