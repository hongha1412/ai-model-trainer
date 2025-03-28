"""
Module for model training with different learning approaches
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import time

# Import dependency checker
from utils.dependency_checker import import_optional_dependency

# Use dependency checker to import and potentially install pandas
pd = import_optional_dependency("pandas")

# Use dependency checker to import and potentially install torch
torch = import_optional_dependency("torch")
torch_available = torch is not None

# Use dependency checker to import and potentially install transformers
transformers = import_optional_dependency("transformers")
transformers_available = transformers is not None

AutoModelForSequenceClassification = None
AutoTokenizer = None
TrainingArguments = None
Trainer = None

if transformers_available:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
    except ImportError:
        logging.warning("Could not import specific transformers classes")

# Use dependency checker to import and potentially install sklearn
sklearn = import_optional_dependency("sklearn")
sklearn_available = sklearn is not None
train_test_split = None

if sklearn_available:
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        logging.warning("Could not import train_test_split from scikit-learn")

from training.dataset_handler import DatasetHandler
from training.config import TrainingConfig
from models.ai_model import register_model, unregister_model, get_model

logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Base class for model trainers
    """
    
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """
        Initialize trainer
        
        Args:
            model_id: Unique identifier for the model
            model_config: Configuration for the model
        """
        self.model_id = model_id
        self.model_config = model_config
        self.model_type = model_config.get("model_type", "t5")
        self.model_path = model_config.get("model_path", "")
        self.training_config = {}
        self.learning_type = "supervised"  # Default learning type
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def prepare_training_config(self, custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare training configuration by merging with defaults
        
        Args:
            custom_config: Custom configuration for training
            
        Returns:
            Merged training configuration
        """
        self.learning_type = custom_config.get("learning_type", self.learning_type)
        self.training_config = TrainingConfig.merge_with_defaults(custom_config, self.learning_type)
        return self.training_config
    
    def validate_training_config(self) -> List[str]:
        """
        Validate training configuration
        
        Returns:
            List of validation error messages (empty if validation successful)
        """
        return TrainingConfig.validate_config(self.training_config, self.learning_type)
    
    def load_model(self) -> Any:
        """
        Load model for training
        
        Returns:
            Loaded model
        """
        raise NotImplementedError("Subclasses must implement load_model method")
    
    def prepare_dataset(self, dataset: Any, input_field: str, output_field: Optional[str] = None) -> Any:
        """
        Prepare dataset for training
        
        Args:
            dataset: Dataset to prepare
            input_field: Field containing input data
            output_field: Field containing output data (for supervised learning)
            
        Returns:
            Prepared dataset
        """
        raise NotImplementedError("Subclasses must implement prepare_dataset method")
    
    def train(self, dataset: Any, input_field: str, output_field: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model
        
        Args:
            dataset: Dataset for training
            input_field: Field containing input data
            output_field: Field containing output data (for supervised learning)
            
        Returns:
            Dictionary with training results
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            save_path: Path to save the model (if None, use default path)
            
        Returns:
            Path where model was saved
        """
        raise NotImplementedError("Subclasses must implement save_model method")


class SupervisedTrainer(BaseTrainer):
    """
    Trainer for supervised learning
    """
    
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """
        Initialize supervised trainer
        
        Args:
            model_id: Unique identifier for the model
            model_config: Configuration for the model
        """
        super().__init__(model_id, model_config)
        self.learning_type = "supervised"
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> Any:
        """
        Load model for supervised training
        
        Returns:
            Loaded model
        """
        try:
            # Check if model path is from Hugging Face or local
            if "/" in self.model_path and not os.path.exists(self.model_path):
                # Assume it's a Hugging Face model ID
                logger.info(f"Loading model from Hugging Face: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            elif os.path.exists(self.model_path):
                # Local model
                logger.info(f"Loading model from local path: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            else:
                raise ValueError(f"Invalid model path: {self.model_path}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_dataset(self, dataset: DatasetHandler, input_field: str, output_field: str) -> Any:
        """
        Prepare dataset for supervised training
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Field containing output data
            
        Returns:
            Prepared dataset for training
        """
        try:
            # Convert dataset to training format
            training_data = dataset.convert_to_training_format(
                input_field=input_field,
                output_field=output_field,
                learning_type="supervised"
            )
            
            if not training_data or "inputs" not in training_data or "outputs" not in training_data:
                raise ValueError("Invalid training data format")
            
            # Split into train and eval
            train_texts, eval_texts, train_labels, eval_labels = train_test_split(
                training_data["inputs"],
                training_data["outputs"],
                test_size=0.2,
                random_state=42
            )
            
            # Tokenize inputs
            train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
            eval_encodings = self.tokenizer(eval_texts, truncation=True, padding=True)
            
            # Create torch datasets
            class SupervisedDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels

                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item['labels'] = torch.tensor(self.labels[idx])
                    return item

                def __len__(self):
                    return len(self.labels)
            
            # Convert labels to integers if they are strings
            if isinstance(train_labels[0], str):
                label_map = {label: i for i, label in enumerate(set(train_labels + eval_labels))}
                train_labels = [label_map[label] for label in train_labels]
                eval_labels = [label_map[label] for label in eval_labels]
                
                # Save label map
                os.makedirs("models/mappings", exist_ok=True)
                with open(f"models/mappings/{self.model_id}_label_map.json", "w") as f:
                    json.dump(label_map, f)
            
            train_dataset = SupervisedDataset(train_encodings, train_labels)
            eval_dataset = SupervisedDataset(eval_encodings, eval_labels)
            
            return {
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset
            }
        
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise
    
    def train(self, dataset: DatasetHandler, input_field: str, output_field: str) -> Dict[str, Any]:
        """
        Train model with supervised learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Field containing output data
            
        Returns:
            Dictionary with training results
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Prepare dataset
            prepared_dataset = self.prepare_dataset(dataset, input_field, output_field)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=f"models/outputs/{self.model_id}",
                num_train_epochs=self.training_config.get("num_train_epochs", 3),
                per_device_train_batch_size=self.training_config.get("batch_size", 8),
                per_device_eval_batch_size=self.training_config.get("batch_size", 8),
                warmup_steps=self.training_config.get("warmup_steps", 500),
                weight_decay=self.training_config.get("weight_decay", 0.01),
                logging_dir=f"models/logs/{self.model_id}",
                logging_steps=100,
                evaluation_strategy=self.training_config.get("evaluation_strategy", "epoch"),
                save_strategy=self.training_config.get("evaluation_strategy", "epoch"),
                load_best_model_at_end=True,
                learning_rate=self.training_config.get("learning_rate", 2e-5),
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=prepared_dataset["train_dataset"],
                eval_dataset=prepared_dataset["eval_dataset"],
            )
            
            # Train model
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Evaluate model
            eval_result = trainer.evaluate()
            
            # Save model
            save_path = self.save_model()
            
            # Register model
            register_model(self.model_id, self.model, self.model_type)
            
            return {
                "status": "success",
                "model_id": self.model_id,
                "training_time": training_time,
                "train_loss": train_result.metrics.get("train_loss"),
                "eval_loss": eval_result.get("eval_loss"),
                "save_path": save_path
            }
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {
                "status": "error",
                "model_id": self.model_id,
                "error": str(e)
            }
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            save_path: Path to save the model (if None, use default path)
            
        Returns:
            Path where model was saved
        """
        try:
            if save_path is None:
                save_path = f"models/trained/{self.model_id}"
            
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save training config
            with open(os.path.join(save_path, "training_config.json"), "w") as f:
                json.dump(self.training_config, f, indent=2)
            
            return save_path
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


class UnsupervisedTrainer(BaseTrainer):
    """
    Trainer for unsupervised learning
    """
    
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """
        Initialize unsupervised trainer
        
        Args:
            model_id: Unique identifier for the model
            model_config: Configuration for the model
        """
        super().__init__(model_id, model_config)
        self.learning_type = "unsupervised"
        self.model = None
        self.tokenizer = None
    
    def load_model(self) -> Any:
        """
        Load model for unsupervised training
        
        Returns:
            Loaded model
        """
        try:
            # For unsupervised learning, typically use autoencoder or language model
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            
            # Check if model path is from Hugging Face or local
            if "/" in self.model_path and not os.path.exists(self.model_path):
                # Assume it's a Hugging Face model ID
                logger.info(f"Loading model from Hugging Face: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            elif os.path.exists(self.model_path):
                # Local model
                logger.info(f"Loading model from local path: {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            else:
                raise ValueError(f"Invalid model path: {self.model_path}")
            
            return self.model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_dataset(self, dataset: DatasetHandler, input_field: str, output_field: Optional[str] = None) -> Any:
        """
        Prepare dataset for unsupervised training
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Not used for unsupervised learning
            
        Returns:
            Prepared dataset for training
        """
        try:
            # Convert dataset to training format
            training_data = dataset.convert_to_training_format(
                input_field=input_field,
                learning_type="unsupervised"
            )
            
            if not training_data or "inputs" not in training_data:
                raise ValueError("Invalid training data format")
            
            # Split into train and eval
            train_texts, eval_texts = train_test_split(
                training_data["inputs"],
                test_size=0.2,
                random_state=42
            )
            
            # Tokenize inputs
            train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
            eval_encodings = self.tokenizer(eval_texts, truncation=True, padding=True)
            
            # Create torch datasets
            class UnsupervisedDataset(torch.utils.data.Dataset):
                def __init__(self, encodings):
                    self.encodings = encodings

                def __getitem__(self, idx):
                    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

                def __len__(self):
                    return len(self.encodings.input_ids)
            
            train_dataset = UnsupervisedDataset(train_encodings)
            eval_dataset = UnsupervisedDataset(eval_encodings)
            
            return {
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset
            }
        
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise
    
    def train(self, dataset: DatasetHandler, input_field: str, output_field: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model with unsupervised learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Not used for unsupervised learning
            
        Returns:
            Dictionary with training results
        """
        try:
            # Load model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Prepare dataset
            prepared_dataset = self.prepare_dataset(dataset, input_field)
            
            # Configure training arguments
            training_args = TrainingArguments(
                output_dir=f"models/outputs/{self.model_id}",
                num_train_epochs=self.training_config.get("num_train_epochs", 5),
                per_device_train_batch_size=self.training_config.get("batch_size", 16),
                per_device_eval_batch_size=self.training_config.get("batch_size", 16),
                warmup_steps=self.training_config.get("warmup_steps", 1000),
                weight_decay=self.training_config.get("weight_decay", 0.1),
                logging_dir=f"models/logs/{self.model_id}",
                logging_steps=100,
                save_steps=self.training_config.get("save_steps", 10000),
                learning_rate=self.training_config.get("learning_rate", 1e-4),
            )
            
            # DataCollator for language modeling
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=True, 
                mlm_probability=0.15
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=prepared_dataset["train_dataset"],
                eval_dataset=prepared_dataset["eval_dataset"],
            )
            
            # Train model
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Evaluate model
            eval_result = trainer.evaluate()
            
            # Save model
            save_path = self.save_model()
            
            # Register model
            register_model(self.model_id, self.model, self.model_type)
            
            return {
                "status": "success",
                "model_id": self.model_id,
                "training_time": training_time,
                "train_loss": train_result.metrics.get("train_loss"),
                "eval_loss": eval_result.get("eval_loss"),
                "save_path": save_path
            }
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {
                "status": "error",
                "model_id": self.model_id,
                "error": str(e)
            }
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            save_path: Path to save the model (if None, use default path)
            
        Returns:
            Path where model was saved
        """
        try:
            if save_path is None:
                save_path = f"models/trained/{self.model_id}"
            
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save training config
            with open(os.path.join(save_path, "training_config.json"), "w") as f:
                json.dump(self.training_config, f, indent=2)
            
            return save_path
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


# Factory for creating trainers
def create_trainer(learning_type: str, model_id: str, model_config: Dict[str, Any]) -> BaseTrainer:
    """
    Create trainer for specified learning type
    
    Args:
        learning_type: Type of learning (supervised, unsupervised, etc.)
        model_id: Unique identifier for the model
        model_config: Configuration for the model
        
    Returns:
        Trainer instance for the specified learning type
    """
    if learning_type == "supervised":
        return SupervisedTrainer(model_id, model_config)
    elif learning_type == "unsupervised":
        return UnsupervisedTrainer(model_id, model_config)
    elif learning_type == "reinforcement":
        # For future implementation
        raise NotImplementedError("Reinforcement learning trainer not implemented yet")
    elif learning_type == "semi_supervised":
        # For future implementation
        raise NotImplementedError("Semi-supervised learning trainer not implemented yet")
    elif learning_type == "self_supervised":
        # For future implementation
        raise NotImplementedError("Self-supervised learning trainer not implemented yet")
    elif learning_type == "online":
        # For future implementation
        raise NotImplementedError("Online learning trainer not implemented yet")
    elif learning_type == "federated":
        # For future implementation
        raise NotImplementedError("Federated learning trainer not implemented yet")
    else:
        raise ValueError(f"Unsupported learning type: {learning_type}")