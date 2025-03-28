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
        self.job_id = None  # Will be set by the training process
        
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
            
            # Create a custom callback for monitoring progress and checking for stop requests
            class MonitorCallback(transformers.TrainerCallback):
                def __init__(self, trainer_instance):
                    self.trainer_instance = trainer_instance
                
                def on_epoch_begin(self, args, state, control, **kwargs):
                    # Update current epoch
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import update_training_status, add_log_message
                        update_training_status(
                            self.trainer_instance.job_id,
                            current_epoch=state.epoch,
                            progress=min(100, int((state.epoch / state.max_steps) * 100))
                        )
                        add_log_message(self.trainer_instance.job_id, f"Starting epoch {state.epoch}")
                
                def on_step_end(self, args, state, control, **kwargs):
                    # Check for stop requests
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import check_stop_requested, update_training_status
                        
                        # Update progress
                        progress = min(100, int((state.global_step / state.max_steps) * 100))
                        update_training_status(
                            self.trainer_instance.job_id,
                            current_step=state.global_step,
                            progress=progress
                        )
                        
                        # Check if stop was requested
                        if check_stop_requested(self.trainer_instance.job_id):
                            update_training_status(
                                self.trainer_instance.job_id,
                                status="stopped",
                                current_step=state.global_step
                            )
                            control.should_training_stop = True
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    # Update metrics
                    if logs and hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import update_training_status
                        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                        update_training_status(
                            self.trainer_instance.job_id,
                            metrics=metrics
                        )
            
            # Create trainer with callback
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=prepared_dataset["train_dataset"],
                eval_dataset=prepared_dataset["eval_dataset"],
                callbacks=[MonitorCallback(self)]
            )
            
            # Report progress through monitoring system if job_id is set
            if hasattr(self, 'job_id') and self.job_id:
                from api.monitor import update_training_status, add_log_message
                update_training_status(
                    self.job_id,
                    status="training",
                    progress=5,
                    current_epoch=0
                )
                add_log_message(self.job_id, "Starting training...")
            
            # Train model
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Check if training was stopped
            was_stopped = False
            if hasattr(self, 'job_id') and self.job_id:
                from api.monitor import get_job_status
                job_status = get_job_status(self.job_id)
                was_stopped = job_status and job_status.get("status") == "stopped"
            
            if was_stopped:
                # If training was stopped by user, don't continue with evaluation
                if hasattr(self, 'job_id') and self.job_id:
                    from api.monitor import add_log_message
                    add_log_message(self.job_id, "Training was stopped by user request")
                    
                return {
                    "status": "stopped",
                    "model_id": self.model_id,
                    "training_time": training_time,
                    "message": "Training was stopped by user request"
                }
            
            # Evaluate model
            eval_result = trainer.evaluate()
            
            # Save model
            save_path = self.save_model()
            
            # Register model
            register_model(self.model_id, self.model, self.model_type)
            
            # Update final status if we have a job_id
            if hasattr(self, 'job_id') and self.job_id:
                from api.monitor import update_training_status, add_log_message
                update_training_status(
                    self.job_id,
                    status="completed",
                    progress=100,
                    metrics={
                        "train_loss": train_result.metrics.get("train_loss"),
                        "eval_loss": eval_result.get("eval_loss")
                    }
                )
                add_log_message(self.job_id, f"Training completed in {training_time:.2f} seconds")
            
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
            
            # Create a custom callback for monitoring progress and checking for stop requests
            class MonitorCallback(transformers.TrainerCallback):
                def __init__(self, trainer_instance):
                    self.trainer_instance = trainer_instance
                
                def on_epoch_begin(self, args, state, control, **kwargs):
                    # Update current epoch
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import update_training_status, add_log_message
                        update_training_status(
                            self.trainer_instance.job_id,
                            current_epoch=state.epoch,
                            progress=min(100, int((state.epoch / state.max_steps) * 100))
                        )
                        add_log_message(self.trainer_instance.job_id, f"Starting epoch {state.epoch}")
                
                def on_step_end(self, args, state, control, **kwargs):
                    # Check for stop requests
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import check_stop_requested, update_training_status
                        
                        # Update progress
                        progress = min(100, int((state.global_step / state.max_steps) * 100))
                        update_training_status(
                            self.trainer_instance.job_id,
                            current_step=state.global_step,
                            progress=progress
                        )
                        
                        # Check if stop was requested
                        if check_stop_requested(self.trainer_instance.job_id):
                            update_training_status(
                                self.trainer_instance.job_id,
                                status="stopped",
                                current_step=state.global_step
                            )
                            control.should_training_stop = True
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    # Update metrics
                    if logs and hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import update_training_status
                        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                        update_training_status(
                            self.trainer_instance.job_id,
                            metrics=metrics
                        )
            
            # DataCollator for language modeling
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=True, 
                mlm_probability=0.15
            )
            
            # Create trainer with callback
            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=prepared_dataset["train_dataset"],
                eval_dataset=prepared_dataset["eval_dataset"],
                callbacks=[MonitorCallback(self)]
            )
            
            # Report progress through monitoring system if job_id is set
            if hasattr(self, 'job_id') and self.job_id:
                from api.monitor import update_training_status, add_log_message
                update_training_status(
                    self.job_id,
                    status="training",
                    progress=5,
                    current_epoch=0
                )
                add_log_message(self.job_id, "Starting training...")
            
            # Train model
            start_time = time.time()
            train_result = trainer.train()
            training_time = time.time() - start_time
            
            # Check if training was stopped
            was_stopped = False
            if hasattr(self, 'job_id') and self.job_id:
                from api.monitor import get_job_status
                job_status = get_job_status(self.job_id)
                was_stopped = job_status and job_status.get("status") == "stopped"
            
            if was_stopped:
                # If training was stopped by user, don't continue with evaluation
                if hasattr(self, 'job_id') and self.job_id:
                    from api.monitor import add_log_message
                    add_log_message(self.job_id, "Training was stopped by user request")
                    
                return {
                    "status": "stopped",
                    "model_id": self.model_id,
                    "training_time": training_time,
                    "message": "Training was stopped by user request"
                }
            
            # Evaluate model
            eval_result = trainer.evaluate()
            
            # Save model
            save_path = self.save_model()
            
            # Register model
            register_model(self.model_id, self.model, self.model_type)
            
            # Update final status if we have a job_id
            if hasattr(self, 'job_id') and self.job_id:
                from api.monitor import update_training_status, add_log_message
                update_training_status(
                    self.job_id,
                    status="completed",
                    progress=100,
                    metrics={
                        "train_loss": train_result.metrics.get("train_loss"),
                        "eval_loss": eval_result.get("eval_loss")
                    }
                )
                add_log_message(self.job_id, f"Training completed in {training_time:.2f} seconds")
            
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
class ReinforcementTrainer(BaseTrainer):
    """
    Trainer for reinforcement learning
    """
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """
        Initialize reinforcement trainer
        
        Args:
            model_id: Unique identifier for the model
            model_config: Configuration for the model
        """
        super().__init__(model_id, model_config)
        self.job_id = None
        
    def load_model(self) -> Any:
        """
        Load model for reinforcement training
        
        Returns:
            Loaded model
        """
        import torch
        import transformers
        
        model_path = self.model_config.get('model_path')
        model_type = self.model_config.get('model_type', 'auto')
        
        # For reinforcement learning, we need a policy network and often a value network
        if model_type == 't5':
            # Use T5 as the base for our policy network
            tokenizer = transformers.T5Tokenizer.from_pretrained(model_path)
            config = transformers.T5Config.from_pretrained(model_path)
            policy_model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)
            
            # Create a simple value network (for state-value estimation)
            class ValueNetwork(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.dense = torch.nn.Linear(config.d_model, 512)
                    self.activation = torch.nn.ReLU()
                    self.output = torch.nn.Linear(512, 1)
                
                def forward(self, hidden_states):
                    x = self.dense(hidden_states)
                    x = self.activation(x)
                    return self.output(x)
            
            value_model = ValueNetwork(config)
            
            return {
                'policy_model': policy_model,
                'value_model': value_model,
                'tokenizer': tokenizer,
                'config': config
            }
        else:
            # Generic approach for other model types
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            config = transformers.AutoConfig.from_pretrained(model_path)
            policy_model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
            
            # For value network, create a simpler model
            class ValueNetwork(torch.nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.dense = torch.nn.Linear(config.hidden_size, 512)
                    self.activation = torch.nn.ReLU()
                    self.output = torch.nn.Linear(512, 1)
                
                def forward(self, hidden_states):
                    x = self.dense(hidden_states)
                    x = self.activation(x)
                    return self.output(x)
            
            value_model = ValueNetwork(config)
            
            return {
                'policy_model': policy_model,
                'value_model': value_model,
                'tokenizer': tokenizer,
                'config': config
            }
    
    def prepare_dataset(self, dataset: DatasetHandler, input_field: str, output_field: Optional[str] = None) -> Any:
        """
        Prepare dataset for reinforcement learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing state representations
            output_field: Field containing action-reward pairs (if available)
            
        Returns:
            Prepared dataset for training
        """
        import torch
        import transformers
        
        # For RL, we need a dataset that contains state representations and possibly action-reward pairs
        train_data, val_data = dataset.get_train_val_split()
        
        # Get components from load_model
        model_components = self.load_model()
        tokenizer = model_components['tokenizer']
        
        # For RL, we prepare states as inputs
        train_states = train_data[input_field]
        val_states = val_data[input_field] if val_data is not None else None
        
        # Tokenize states
        train_encodings = tokenizer(train_states, truncation=True, 
                                   padding="max_length", 
                                   max_length=self.training_config.get('max_seq_length', 128))
        
        val_encodings = None
        if val_states is not None:
            val_encodings = tokenizer(val_states, truncation=True, 
                                     padding="max_length", 
                                     max_length=self.training_config.get('max_seq_length', 128))
        
        # For RL, we may also have action-reward pairs in the dataset
        train_actions = None
        train_rewards = None
        if output_field and output_field in train_data:
            if isinstance(train_data[output_field], dict):
                train_actions = train_data[output_field].get('action')
                train_rewards = train_data[output_field].get('reward')
            elif hasattr(train_data, 'action') and hasattr(train_data, 'reward'):
                train_actions = train_data.action
                train_rewards = train_data.reward
        
        # Create RL dataset class
        class RLDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, actions=None, rewards=None):
                self.encodings = encodings
                self.actions = actions
                self.rewards = rewards
                
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                if self.actions is not None and idx < len(self.actions):
                    item['action'] = self.actions[idx]
                if self.rewards is not None and idx < len(self.rewards):
                    item['reward'] = self.rewards[idx]
                return item
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        # Create training and validation datasets
        train_dataset = RLDataset(train_encodings, train_actions, train_rewards)
        val_dataset = RLDataset(val_encodings) if val_encodings is not None else None
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'model_components': model_components
        }
    
    def train(self, dataset: DatasetHandler, input_field: str, output_field: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model with reinforcement learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing state representations
            output_field: Field containing action-reward pairs (if available)
            
        Returns:
            Dictionary with training results
        """
        import os
        import torch
        import numpy as np
        import transformers
        from api.monitor import register_training_job, update_training_status, add_log_message
        
        # Get prepared datasets and models
        prepared_data = self.prepare_dataset(dataset, input_field, output_field)
        train_dataset = prepared_data['train_dataset']
        val_dataset = prepared_data['val_dataset']
        model_components = prepared_data['model_components']
        
        policy_model = model_components['policy_model']
        value_model = model_components['value_model']
        tokenizer = model_components['tokenizer']
        
        # Create output directory
        output_dir = self.training_config.get('output_dir', 'models/trained')
        os.makedirs(output_dir, exist_ok=True)
        
        # RL parameters
        discount_factor = self.training_config.get('discount_factor', 0.99)
        target_update_interval = self.training_config.get('target_update_interval', 1000)
        replay_buffer_size = self.training_config.get('replay_buffer_size', 10000)
        exploration_rate = self.training_config.get('exploration_rate', 0.1)
        max_steps = self.training_config.get('max_steps', 100000)
        batch_size = self.training_config.get('batch_size', 64)
        learning_rate = self.training_config.get('learning_rate', 5e-5)
        
        # Register training job for monitoring
        self.job_id = dataset.name + '_' + self.model_id + '_' + str(int(time.time()))
        register_training_job(
            self.job_id, 
            self.model_id, 
            dataset.name, 
            self.training_config
        )
        
        try:
            # Update status to training
            update_training_status(self.job_id, status="training", progress=0)
            add_log_message(self.job_id, "Starting reinforcement learning training")
            
            # Create optimizers
            policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
            value_optimizer = torch.optim.Adam(value_model.parameters(), lr=learning_rate)
            
            # Create target networks (for stability)
            target_policy_model = type(policy_model)(model_components['config'])
            target_policy_model.load_state_dict(policy_model.state_dict())
            
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            # Setup replay buffer for RL
            class ReplayBuffer:
                def __init__(self, capacity):
                    self.capacity = capacity
                    self.buffer = []
                    self.position = 0
                
                def push(self, state, action, reward, next_state, done):
                    if len(self.buffer) < self.capacity:
                        self.buffer.append(None)
                    self.buffer[self.position] = (state, action, reward, next_state, done)
                    self.position = (self.position + 1) % self.capacity
                
                def sample(self, batch_size):
                    batch = random.sample(self.buffer, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    return states, actions, rewards, next_states, dones
                
                def __len__(self):
                    return len(self.buffer)
            
            # Create replay buffer
            replay_buffer = ReplayBuffer(replay_buffer_size)
            
            # Main training loop
            global_step = 0
            epoch = 0
            best_reward = -float('inf')
            
            # In a real environment, we would interact with it
            # Here we'll simulate RL training using the provided dataset
            while global_step < max_steps:
                epoch += 1
                epoch_rewards = []
                
                for batch_idx, batch in enumerate(dataloader):
                    # Check for stop request
                    from api.monitor import check_stop_requested
                    if check_stop_requested(self.job_id):
                        update_training_status(
                            self.job_id,
                            status="stopped",
                            current_step=global_step
                        )
                        add_log_message(self.job_id, "Training stopped by user request")
                        
                        # Save the current model
                        policy_model.save_pretrained(os.path.join(output_dir, "policy_model"))
                        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
                        torch.save(value_model.state_dict(), os.path.join(output_dir, "value_model.pt"))
                        
                        return {
                            "success": True,
                            "model_path": output_dir,
                            "status": "stopped"
                        }
                    
                    # Update progress
                    global_step += 1
                    progress = min(100, int((global_step / max_steps) * 100))
                    update_training_status(
                        self.job_id,
                        status="training",
                        current_step=global_step,
                        current_epoch=epoch,
                        progress=progress
                    )
                    
                    # Get batch data
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                    
                    # Forward pass through policy network
                    with torch.no_grad():
                        policy_outputs = policy_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True
                        )
                        
                        # Get action logits - for text generation, this is next token prediction
                        logits = policy_outputs.logits
                        
                        # Apply epsilon-greedy exploration
                        if random.random() < exploration_rate:
                            # Random action
                            action = torch.randint(0, logits.size(-1), (logits.size(0), 1))
                        else:
                            # Greedy action
                            action = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    
                    # Simulate environment interaction (in a real scenario, we'd get this from the environment)
                    # Here we'll just use random rewards for demonstration
                    reward = torch.rand(input_ids.size(0), 1)
                    done = torch.zeros(input_ids.size(0), 1)
                    epoch_rewards.extend(reward.numpy())
                    
                    # Create next state (in a real scenario, this would come from environment)
                    next_input_ids = torch.cat([input_ids[:, 1:], action], dim=1)
                    
                    # Store in replay buffer
                    for i in range(input_ids.size(0)):
                        replay_buffer.push(
                            (input_ids[i], attention_mask[i]), 
                            action[i], 
                            reward[i], 
                            (next_input_ids[i], attention_mask[i]), 
                            done[i]
                        )
                    
                    # Sample from replay buffer and update networks (if buffer is large enough)
                    if len(replay_buffer) >= batch_size:
                        # Sample transitions
                        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                        
                        # Prepare batch data
                        state_input_ids = torch.stack([s[0] for s in states])
                        state_attention_mask = torch.stack([s[1] for s in states])
                        action_batch = torch.stack(actions)
                        reward_batch = torch.stack(rewards)
                        next_state_input_ids = torch.stack([s[0] for s in next_states])
                        next_state_attention_mask = torch.stack([s[1] for s in next_states])
                        done_batch = torch.stack(dones)
                        
                        # Compute target Q-values
                        with torch.no_grad():
                            # Get next action from target policy
                            next_policy_outputs = target_policy_model(
                                input_ids=next_state_input_ids,
                                attention_mask=next_state_attention_mask,
                                return_dict=True
                            )
                            next_logits = next_policy_outputs.logits
                            next_action = torch.argmax(next_logits[:, -1, :], dim=-1, keepdim=True)
                            
                            # Get value of next state-action
                            next_hidden_states = next_policy_outputs.encoder_last_hidden_state
                            if next_hidden_states is None:
                                next_hidden_states = next_policy_outputs.last_hidden_state[:, 0, :]
                            next_values = value_model(next_hidden_states)
                            
                            # Compute target values (Bellman equation)
                            target_values = reward_batch + discount_factor * next_values * (1 - done_batch)
                        
                        # Get current Q-values from policy and value networks
                        policy_outputs = policy_model(
                            input_ids=state_input_ids,
                            attention_mask=state_attention_mask,
                            return_dict=True
                        )
                        logits = policy_outputs.logits
                        
                        # Gather the logits corresponding to the actions taken
                        action_logits = torch.gather(
                            logits[:, -1, :], 1, action_batch
                        )
                        
                        # Get current state value
                        hidden_states = policy_outputs.encoder_last_hidden_state
                        if hidden_states is None:
                            hidden_states = policy_outputs.last_hidden_state[:, 0, :]
                        current_values = value_model(hidden_states)
                        
                        # Compute actor loss (policy gradient)
                        advantage = (target_values - current_values).detach()
                        actor_loss = -torch.log_softmax(logits[:, -1, :], dim=-1).gather(1, action_batch) * advantage
                        actor_loss = actor_loss.mean()
                        
                        # Compute critic loss (value function)
                        critic_loss = torch.nn.functional.mse_loss(current_values, target_values)
                        
                        # Update policy network
                        policy_optimizer.zero_grad()
                        actor_loss.backward()
                        policy_optimizer.step()
                        
                        # Update value network
                        value_optimizer.zero_grad()
                        critic_loss.backward()
                        value_optimizer.step()
                        
                        # Periodically update target networks
                        if global_step % target_update_interval == 0:
                            target_policy_model.load_state_dict(policy_model.state_dict())
                            
                            # Log the update
                            add_log_message(self.job_id, f"Updated target networks at step {global_step}")
                        
                        # Log metrics
                        if global_step % 100 == 0:
                            metrics = {
                                'actor_loss': actor_loss.item(),
                                'critic_loss': critic_loss.item(),
                                'avg_reward': reward_batch.mean().item(),
                                'exploration_rate': exploration_rate
                            }
                            
                            update_training_status(
                                self.job_id,
                                metrics=metrics
                            )
                            
                            add_log_message(
                                self.job_id, 
                                f"Step {global_step}: Actor Loss: {actor_loss.item():.4f}, "
                                f"Critic Loss: {critic_loss.item():.4f}, "
                                f"Avg Reward: {reward_batch.mean().item():.4f}"
                            )
                
                # End of epoch
                avg_epoch_reward = np.mean(epoch_rewards) if epoch_rewards else 0
                add_log_message(self.job_id, f"Epoch {epoch} completed with average reward: {avg_epoch_reward:.4f}")
                
                # Track best reward and save model if improved
                if avg_epoch_reward > best_reward:
                    best_reward = avg_epoch_reward
                    policy_model.save_pretrained(os.path.join(output_dir, "policy_model_best"))
                    add_log_message(self.job_id, f"New best model saved with reward: {best_reward:.4f}")
                
                # Decay exploration rate
                exploration_rate = max(0.05, exploration_rate * 0.95)
            
            # Training complete
            policy_model.save_pretrained(os.path.join(output_dir, "policy_model"))
            tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
            torch.save(value_model.state_dict(), os.path.join(output_dir, "value_model.pt"))
            
            update_training_status(self.job_id, status="completed", progress=100)
            add_log_message(self.job_id, f"Training completed successfully with final reward: {best_reward:.4f}")
            
            return {
                "success": True,
                "model_path": output_dir,
                "best_reward": best_reward,
                "status": "completed"
            }
            
        except Exception as e:
            update_training_status(self.job_id, status="error", error=str(e))
            add_log_message(self.job_id, f"Error during reinforcement learning: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            save_path: Path to save the model (if None, use default path)
            
        Returns:
            Path where model was saved
        """
        import os
        import torch
        
        # Default path if none provided
        if save_path is None:
            save_path = f"models/trained/{self.model_id}"
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Get model components
        try:
            model_components = self.load_model()
            policy_model = model_components['policy_model']
            value_model = model_components['value_model']
            tokenizer = model_components['tokenizer']
            
            # Save model components
            policy_model.save_pretrained(os.path.join(save_path, "policy_model"))
            tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
            torch.save(value_model.state_dict(), os.path.join(save_path, "value_model.pt"))
            
            return save_path
        except Exception as e:
            raise ValueError(f"Failed to save model: {str(e)}")


class SemiSupervisedTrainer(BaseTrainer):
    """
    Trainer for semi-supervised learning
    """
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """
        Initialize semi-supervised trainer
        
        Args:
            model_id: Unique identifier for the model
            model_config: Configuration for the model
        """
        super().__init__(model_id, model_config)
        self.job_id = None
    
    def load_model(self) -> Any:
        """
        Load model for semi-supervised training
        
        Returns:
            Loaded model
        """
        import torch
        import transformers
        
        model_path = self.model_config.get('model_path')
        model_type = self.model_config.get('model_type', 'auto')
        
        if model_type == 't5':
            # Load T5 model for semi-supervised learning
            tokenizer = transformers.T5Tokenizer.from_pretrained(model_path)
            config = transformers.T5Config.from_pretrained(model_path)
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
        else:
            # Default to auto model
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            config = transformers.AutoConfig.from_pretrained(model_path)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
    
    def prepare_dataset(self, dataset: DatasetHandler, input_field: str, output_field: str) -> Any:
        """
        Prepare dataset for semi-supervised training
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Field containing output data (for labeled examples)
            
        Returns:
            Prepared dataset for training
        """
        import torch
        import transformers
        
        # Get dataset split
        train_data, val_data = dataset.get_train_val_split()
        
        # Get model components
        model_components = self.load_model()
        tokenizer = model_components['tokenizer']
        
        # Process texts
        train_texts = train_data[input_field]
        train_labels = train_data[output_field] if output_field in train_data else None
        
        # Identify labeled and unlabeled data
        if train_labels is not None:
            # Find indices of labeled data (non-null entries)
            labeled_indices = [i for i, lbl in enumerate(train_labels) if lbl is not None]
            unlabeled_indices = [i for i, lbl in enumerate(train_labels) if lbl is None]
            
            labeled_texts = [train_texts[i] for i in labeled_indices]
            labeled_labels = [train_labels[i] for i in labeled_indices]
            unlabeled_texts = [train_texts[i] for i in unlabeled_indices]
        else:
            # If no label field is provided, assume first 20% is labeled (for demonstration)
            split_idx = int(len(train_texts) * 0.2)
            labeled_texts = train_texts[:split_idx]
            unlabeled_texts = train_texts[split_idx:]
            labeled_labels = None
        
        # Tokenize labeled inputs
        labeled_encodings = tokenizer(
            labeled_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=self.training_config.get('max_seq_length', 128)
        )
        
        # Tokenize unlabeled inputs
        unlabeled_encodings = tokenizer(
            unlabeled_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=self.training_config.get('max_seq_length', 128)
        )
        
        # Tokenize validation inputs if available
        val_encodings = None
        val_labels = None
        if val_data is not None:
            val_texts = val_data[input_field]
            val_labels = val_data[output_field] if output_field in val_data else None
            val_encodings = tokenizer(
                val_texts, 
                truncation=True, 
                padding="max_length", 
                max_length=self.training_config.get('max_seq_length', 128)
            )
        
        # Create dataset classes
        class LabeledDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        class UnlabeledDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        # Create datasets
        if labeled_labels:
            labeled_dataset = LabeledDataset(labeled_encodings, labeled_labels)
        else:
            # If no labels, use unlabeled dataset type
            labeled_dataset = UnlabeledDataset(labeled_encodings)
        
        unlabeled_dataset = UnlabeledDataset(unlabeled_encodings)
        
        val_dataset = None
        if val_encodings is not None:
            if val_labels is not None:
                val_dataset = LabeledDataset(val_encodings, val_labels)
            else:
                val_dataset = UnlabeledDataset(val_encodings)
        
        return {
            'labeled_dataset': labeled_dataset,
            'unlabeled_dataset': unlabeled_dataset,
            'val_dataset': val_dataset,
            'model_components': model_components
        }
    
    def train(self, dataset: DatasetHandler, input_field: str, output_field: str) -> Dict[str, Any]:
        """
        Train model with semi-supervised learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Field containing output data (for labeled examples)
            
        Returns:
            Dictionary with training results
        """
        import os
        import torch
        import transformers
        from api.monitor import register_training_job, update_training_status, add_log_message
        
        # Get prepared datasets
        prepared_data = self.prepare_dataset(dataset, input_field, output_field)
        labeled_dataset = prepared_data['labeled_dataset']
        unlabeled_dataset = prepared_data['unlabeled_dataset']
        val_dataset = prepared_data['val_dataset']
        model_components = prepared_data['model_components']
        
        # Setup training parameters
        model = model_components['model']
        tokenizer = model_components['tokenizer']
        
        # Create output directory
        output_dir = self.training_config.get('output_dir', 'models/trained')
        os.makedirs(output_dir, exist_ok=True)
        
        # Register training job
        self.job_id = dataset.name + '_' + self.model_id + '_' + str(int(time.time()))
        register_training_job(
            self.job_id, 
            self.model_id, 
            dataset.name, 
            self.training_config
        )
        
        try:
            # Update status to training
            update_training_status(self.job_id, status="training", progress=0)
            add_log_message(self.job_id, "Starting semi-supervised training")
            
            # Define training arguments
            training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.training_config.get('num_train_epochs', 3),
                per_device_train_batch_size=self.training_config.get('batch_size', 8),
                per_device_eval_batch_size=self.training_config.get('batch_size', 8),
                warmup_steps=self.training_config.get('warmup_steps', 500),
                weight_decay=self.training_config.get('weight_decay', 0.01),
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=10,
                save_steps=self.training_config.get('save_steps', 10000),
                learning_rate=self.training_config.get('learning_rate', 5e-5),
                do_eval=val_dataset is not None,
                seed=42
            )
            
            # Create trainer with monitoring callback
            class MonitorCallback(transformers.TrainerCallback):
                def __init__(self, trainer_instance):
                    self.trainer_instance = trainer_instance
                
                def on_epoch_begin(self, args, state, control, **kwargs):
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import add_log_message, update_training_status
                        
                        update_training_status(
                            self.trainer_instance.job_id,
                            status="training",
                            current_epoch=state.epoch
                        )
                        
                        add_log_message(self.trainer_instance.job_id, f"Starting epoch {state.epoch}")
                
                def on_step_end(self, args, state, control, **kwargs):
                    # Check for stop requests
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import check_stop_requested, update_training_status
                        
                        # Update progress
                        progress = min(100, int((state.global_step / state.max_steps) * 100))
                        update_training_status(
                            self.trainer_instance.job_id,
                            current_step=state.global_step,
                            progress=progress
                        )
                        
                        # Check if stop was requested
                        if check_stop_requested(self.trainer_instance.job_id):
                            update_training_status(
                                self.trainer_instance.job_id,
                                status="stopped",
                                current_step=state.global_step
                            )
                            
                            # Set control.should_save and control.should_training_stop
                            control.should_save = True
                            control.should_training_stop = True
                            
                            add_log_message(self.trainer_instance.job_id, "Training stopped by user request")
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import update_training_status
                        
                        # Process logs to extract metrics
                        if logs:
                            metrics = {}
                            for key, value in logs.items():
                                if isinstance(value, (int, float)):
                                    metrics[key] = value
                            
                            if metrics:
                                update_training_status(
                                    self.trainer_instance.job_id,
                                    metrics=metrics
                                )
            
            # Implement pseudo-labeling approach for semi-supervised learning
            unlabeled_weight = self.training_config.get('unlabeled_weight', 0.5)
            num_epochs = self.training_config.get('num_train_epochs', 3)
            
            # First, train model on labeled data only
            add_log_message(self.job_id, "Phase 1: Training on labeled data only")
            
            trainer = transformers.Trainer(
                model=model,
                args=training_args,
                train_dataset=labeled_dataset,
                eval_dataset=val_dataset,
                callbacks=[MonitorCallback(self)]
            )
            
            # First phase: supervised training
            trainer.train()
            
            # Second phase: generate pseudo-labels and train on combined data
            add_log_message(self.job_id, "Phase 2: Generating pseudo-labels for unlabeled data")
            
            # Create dataloader for unlabeled data
            unlabeled_dataloader = torch.utils.data.DataLoader(
                unlabeled_dataset,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=transformers.default_data_collator
            )
            
            # Generate pseudo-labels
            model.eval()
            pseudo_labels = []
            
            with torch.no_grad():
                for batch in unlabeled_dataloader:
                    # Move batch to device
                    batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
                    
                    # Get predictions
                    outputs = model(**batch)
                    logits = outputs.logits
                    
                    # Get predicted labels
                    if logits.shape[-1] > 1:  # Multi-class classification
                        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                    else:  # Regression or binary classification
                        predictions = (logits > 0).long().cpu().numpy()
                    
                    pseudo_labels.extend(predictions.tolist())
            
            # Create a new dataset with pseudo-labeled data
            class PseudoLabeledDataset(torch.utils.data.Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels = labels
                
                def __getitem__(self, idx):
                    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                    item['labels'] = torch.tensor(self.labels[idx])
                    return item
                
                def __len__(self):
                    return len(self.encodings['input_ids'])
            
            pseudo_labeled_dataset = PseudoLabeledDataset(
                unlabeled_dataset.encodings,
                pseudo_labels
            )
            
            # Combine datasets (with proper weighting)
            class CombinedDataset(torch.utils.data.Dataset):
                def __init__(self, labeled_dataset, pseudo_labeled_dataset, unlabeled_weight):
                    self.labeled_dataset = labeled_dataset
                    self.pseudo_labeled_dataset = pseudo_labeled_dataset
                    self.unlabeled_weight = unlabeled_weight
                    
                    # Compute effective size
                    self.labeled_size = len(labeled_dataset)
                    self.pseudo_size = len(pseudo_labeled_dataset)
                    
                    # How many times to repeat the labeled data
                    self.repeat_factor = max(1, int((self.pseudo_size * unlabeled_weight) / 
                                                  (self.labeled_size * (1 - unlabeled_weight))))
                
                def __getitem__(self, idx):
                    if idx < self.labeled_size * self.repeat_factor:
                        # Access labeled data (with repetition)
                        return self.labeled_dataset[idx % self.labeled_size]
                    else:
                        # Access pseudo-labeled data
                        pseudo_idx = idx - (self.labeled_size * self.repeat_factor)
                        item = self.pseudo_labeled_dataset[pseudo_idx]
                        return item
                
                def __len__(self):
                    return self.labeled_size * self.repeat_factor + self.pseudo_size
            
            combined_dataset = CombinedDataset(
                labeled_dataset,
                pseudo_labeled_dataset,
                unlabeled_weight
            )
            
            # Third phase: train on combined data
            add_log_message(self.job_id, "Phase 3: Training on combined labeled and pseudo-labeled data")
            
            # Update training arguments for phase 3
            training_args = transformers.TrainingArguments(
                output_dir=os.path.join(output_dir, "combined"),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=self.training_config.get('batch_size', 8),
                per_device_eval_batch_size=self.training_config.get('batch_size', 8),
                warmup_steps=self.training_config.get('warmup_steps', 500),
                weight_decay=self.training_config.get('weight_decay', 0.01),
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=10,
                save_steps=self.training_config.get('save_steps', 10000),
                learning_rate=self.training_config.get('learning_rate', 5e-5) * 0.5,  # Lower learning rate
                do_eval=val_dataset is not None,
                seed=42
            )
            
            # Create trainer for phase 3
            trainer = transformers.Trainer(
                model=model,
                args=training_args,
                train_dataset=combined_dataset,
                eval_dataset=val_dataset,
                callbacks=[MonitorCallback(self)]
            )
            
            # Train on combined data
            train_result = trainer.train()
            
            # Save final model
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Update status to completed
            update_training_status(self.job_id, status="completed", progress=100)
            add_log_message(self.job_id, "Semi-supervised training completed successfully")
            
            return {
                "success": True,
                "model_path": output_dir,
                "training_stats": train_result.metrics,
                "status": "completed"
            }
            
        except Exception as e:
            # Update status to error
            update_training_status(self.job_id, status="error", error=str(e))
            add_log_message(self.job_id, f"Error during semi-supervised training: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            save_path: Path to save the model (if None, use default path)
            
        Returns:
            Path where model was saved
        """
        import os
        
        # Default path if none provided
        if save_path is None:
            save_path = f"models/trained/{self.model_id}"
        
        # Get model components
        try:
            model_components = self.load_model()
            model = model_components['model']
            tokenizer = model_components['tokenizer']
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            return save_path
        except Exception as e:
            raise ValueError(f"Failed to save model: {str(e)}")


class SelfSupervisedTrainer(BaseTrainer):
    """
    Trainer for self-supervised learning
    """
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """
        Initialize self-supervised trainer
        
        Args:
            model_id: Unique identifier for the model
            model_config: Configuration for the model
        """
        super().__init__(model_id, model_config)
        self.job_id = None
    
    def load_model(self) -> Any:
        """
        Load model for self-supervised training
        
        Returns:
            Loaded model
        """
        import torch
        import transformers
        
        model_path = self.model_config.get('model_path')
        model_type = self.model_config.get('model_type', 'auto')
        
        if model_type == 't5':
            # Load T5 model for self-supervised learning
            tokenizer = transformers.T5Tokenizer.from_pretrained(model_path)
            config = transformers.T5Config.from_pretrained(model_path)
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
        else:
            # Default to auto model
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            config = transformers.AutoConfig.from_pretrained(model_path)
            
            # For self-supervised learning, we might use different model types
            # Masked Language Modeling
            model = transformers.AutoModelForMaskedLM.from_pretrained(model_path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
    
    def prepare_dataset(self, dataset: DatasetHandler, input_field: str, output_field: Optional[str] = None) -> Any:
        """
        Prepare dataset for self-supervised training
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Not used for self-supervised learning
            
        Returns:
            Prepared dataset for training
        """
        import torch
        import transformers
        import random
        
        # Get dataset split
        train_data, val_data = dataset.get_train_val_split()
        
        # Get model components
        model_components = self.load_model()
        tokenizer = model_components['tokenizer']
        
        # Process texts
        train_texts = train_data[input_field]
        val_texts = val_data[input_field] if val_data is not None else None
        
        # Tokenize inputs
        train_encodings = tokenizer(
            train_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=self.training_config.get('max_seq_length', 128)
        )
        
        val_encodings = None
        if val_texts is not None:
            val_encodings = tokenizer(
                val_texts, 
                truncation=True, 
                padding="max_length", 
                max_length=self.training_config.get('max_seq_length', 128)
            )
        
        # For self-supervised learning, we can use different approaches
        # Here, we'll use a masked language modeling approach
        
        # Create dataset class
        class SelfSupervisedDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __getitem__(self, idx):
                return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        train_dataset = SelfSupervisedDataset(train_encodings)
        val_dataset = SelfSupervisedDataset(val_encodings) if val_encodings is not None else None
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'model_components': model_components
        }
    
    def train(self, dataset: DatasetHandler, input_field: str, output_field: Optional[str] = None) -> Dict[str, Any]:
        """
        Train model with self-supervised learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Not used for self-supervised learning
            
        Returns:
            Dictionary with training results
        """
        import os
        import torch
        import transformers
        from api.monitor import register_training_job, update_training_status, add_log_message
        
        # Get prepared datasets
        prepared_data = self.prepare_dataset(dataset, input_field, output_field)
        train_dataset = prepared_data['train_dataset']
        val_dataset = prepared_data['val_dataset']
        model_components = prepared_data['model_components']
        
        # Setup training parameters
        model = model_components['model']
        tokenizer = model_components['tokenizer']
        
        # Create output directory
        output_dir = self.training_config.get('output_dir', 'models/trained')
        os.makedirs(output_dir, exist_ok=True)
        
        # Register training job
        self.job_id = dataset.name + '_' + self.model_id + '_' + str(int(time.time()))
        register_training_job(
            self.job_id, 
            self.model_id, 
            dataset.name, 
            self.training_config
        )
        
        try:
            # Update status to training
            update_training_status(self.job_id, status="training", progress=0)
            add_log_message(self.job_id, "Starting self-supervised training")
            
            # Define training arguments
            training_args = transformers.TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.training_config.get('num_train_epochs', 3),
                per_device_train_batch_size=self.training_config.get('batch_size', 8),
                per_device_eval_batch_size=self.training_config.get('batch_size', 8),
                warmup_steps=self.training_config.get('warmup_steps', 500),
                weight_decay=self.training_config.get('weight_decay', 0.01),
                logging_dir=os.path.join(output_dir, 'logs'),
                logging_steps=10,
                save_steps=self.training_config.get('save_steps', 10000),
                learning_rate=self.training_config.get('learning_rate', 5e-5),
                do_eval=val_dataset is not None,
                seed=42
            )
            
            # Create trainer with monitoring callback
            class MonitorCallback(transformers.TrainerCallback):
                def __init__(self, trainer_instance):
                    self.trainer_instance = trainer_instance
                
                def on_epoch_begin(self, args, state, control, **kwargs):
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import add_log_message, update_training_status
                        
                        update_training_status(
                            self.trainer_instance.job_id,
                            status="training",
                            current_epoch=state.epoch
                        )
                        
                        add_log_message(self.trainer_instance.job_id, f"Starting epoch {state.epoch}")
                
                def on_step_end(self, args, state, control, **kwargs):
                    # Check for stop requests
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import check_stop_requested, update_training_status
                        
                        # Update progress
                        progress = min(100, int((state.global_step / state.max_steps) * 100))
                        update_training_status(
                            self.trainer_instance.job_id,
                            current_step=state.global_step,
                            progress=progress
                        )
                        
                        # Check if stop was requested
                        if check_stop_requested(self.trainer_instance.job_id):
                            update_training_status(
                                self.trainer_instance.job_id,
                                status="stopped",
                                current_step=state.global_step
                            )
                            
                            # Set control.should_save and control.should_training_stop
                            control.should_save = True
                            control.should_training_stop = True
                            
                            add_log_message(self.trainer_instance.job_id, "Training stopped by user request")
                
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if hasattr(self.trainer_instance, 'job_id') and self.trainer_instance.job_id:
                        from api.monitor import update_training_status
                        
                        # Process logs to extract metrics
                        if logs:
                            metrics = {}
                            for key, value in logs.items():
                                if isinstance(value, (int, float)):
                                    metrics[key] = value
                            
                            if metrics:
                                update_training_status(
                                    self.trainer_instance.job_id,
                                    metrics=metrics
                                )
            
            # Create data collator for self-supervised learning
            # For masked language modeling
            data_collator = transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer, 
                mlm=True, 
                mlm_probability=0.15
            )
            
            # Create trainer
            trainer = transformers.Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
                callbacks=[MonitorCallback(self)]
            )
            
            # Train model
            train_result = trainer.train()
            
            # Save model
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Update status to completed
            update_training_status(self.job_id, status="completed", progress=100)
            add_log_message(self.job_id, "Self-supervised training completed successfully")
            
            return {
                "success": True,
                "model_path": output_dir,
                "training_stats": train_result.metrics,
                "status": "completed"
            }
            
        except Exception as e:
            # Update status to error
            update_training_status(self.job_id, status="error", error=str(e))
            add_log_message(self.job_id, f"Error during self-supervised training: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            save_path: Path to save the model (if None, use default path)
            
        Returns:
            Path where model was saved
        """
        import os
        
        # Default path if none provided
        if save_path is None:
            save_path = f"models/trained/{self.model_id}"
        
        # Get model components
        try:
            model_components = self.load_model()
            model = model_components['model']
            tokenizer = model_components['tokenizer']
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            return save_path
        except Exception as e:
            raise ValueError(f"Failed to save model: {str(e)}")


class OnlineLearningTrainer(BaseTrainer):
    """
    Trainer for online learning
    """
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """
        Initialize online learning trainer
        
        Args:
            model_id: Unique identifier for the model
            model_config: Configuration for the model
        """
        super().__init__(model_id, model_config)
        self.job_id = None
    
    def load_model(self) -> Any:
        """
        Load model for online learning
        
        Returns:
            Loaded model
        """
        import torch
        import transformers
        
        model_path = self.model_config.get('model_path')
        model_type = self.model_config.get('model_type', 'auto')
        
        if model_type == 't5':
            # Load T5 model for online learning
            tokenizer = transformers.T5Tokenizer.from_pretrained(model_path)
            config = transformers.T5Config.from_pretrained(model_path)
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
        else:
            # Default to auto model
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            config = transformers.AutoConfig.from_pretrained(model_path)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
    
    def prepare_dataset(self, dataset: DatasetHandler, input_field: str, output_field: str) -> Any:
        """
        Prepare dataset for online learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Field containing output data
            
        Returns:
            Prepared dataset for training
        """
        import torch
        import transformers
        
        # For online learning, we need to simulate a stream of data
        # We'll use the entire dataset and process it in small batches
        
        # Get dataset
        train_data, _ = dataset.get_train_val_split(0.0)  # No validation split for online learning
        
        # Get model components
        model_components = self.load_model()
        tokenizer = model_components['tokenizer']
        
        # Process texts
        train_texts = train_data[input_field]
        train_labels = train_data[output_field] if output_field in train_data else None
        
        # Tokenize all inputs
        encodings = tokenizer(
            train_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=self.training_config.get('max_seq_length', 128)
        )
        
        # Create dataset class
        class OnlineLearningDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels=None):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                if self.labels is not None:
                    item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        dataset = OnlineLearningDataset(encodings, train_labels)
        
        return {
            'dataset': dataset,
            'model_components': model_components
        }
    
    def train(self, dataset: DatasetHandler, input_field: str, output_field: str) -> Dict[str, Any]:
        """
        Train model with online learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Field containing output data
            
        Returns:
            Dictionary with training results
        """
        import os
        import torch
        import numpy as np
        import transformers
        from api.monitor import register_training_job, update_training_status, add_log_message
        from collections import deque
        
        # Get prepared dataset
        prepared_data = self.prepare_dataset(dataset, input_field, output_field)
        full_dataset = prepared_data['dataset']
        model_components = prepared_data['model_components']
        
        # Setup training parameters
        model = model_components['model']
        tokenizer = model_components['tokenizer']
        
        # Create output directory
        output_dir = self.training_config.get('output_dir', 'models/trained')
        os.makedirs(output_dir, exist_ok=True)
        
        # Online learning parameters
        learning_rate = self.training_config.get('learning_rate', 1e-3)
        window_size = self.training_config.get('window_size', 1000)
        forget_factor = self.training_config.get('forget_factor', 0.1)
        update_interval = self.training_config.get('update_interval', 100)
        max_samples = self.training_config.get('max_samples', 100000)
        
        # Register training job
        self.job_id = dataset.name + '_' + self.model_id + '_' + str(int(time.time()))
        register_training_job(
            self.job_id, 
            self.model_id, 
            dataset.name, 
            self.training_config
        )
        
        try:
            # Update status to training
            update_training_status(self.job_id, status="training", progress=0)
            add_log_message(self.job_id, "Starting online learning training")
            
            # Create optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # For online learning, we use a sliding window of recent examples
            recent_examples = deque(maxlen=window_size)
            
            # Create dataloader for full dataset
            dataloader = torch.utils.data.DataLoader(
                full_dataset, 
                batch_size=1,  # Process one example at a time for online learning
                shuffle=True  # Shuffle to simulate random arrival of data
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Metrics tracking
            running_loss = 0.0
            correctly_predicted = 0
            total_predicted = 0
            
            # Main online learning loop
            examples_processed = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Check if we've processed enough samples
                if examples_processed >= max_samples:
                    break
                
                # Check for stop request
                from api.monitor import check_stop_requested
                if check_stop_requested(self.job_id):
                    update_training_status(
                        self.job_id,
                        status="stopped",
                        current_step=examples_processed
                    )
                    add_log_message(self.job_id, "Training stopped by user request")
                    
                    # Save current model
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    
                    return {
                        "success": True,
                        "model_path": output_dir,
                        "examples_processed": examples_processed,
                        "status": "stopped"
                    }
                
                # Update progress
                examples_processed += 1
                progress = min(100, int((examples_processed / max_samples) * 100))
                
                # Log progress periodically
                if examples_processed % update_interval == 0 or examples_processed == 1:
                    update_training_status(
                        self.job_id,
                        status="training",
                        current_step=examples_processed,
                        progress=progress,
                        metrics={
                            "running_loss": running_loss / max(1, examples_processed % (update_interval * 10)),
                            "accuracy": correctly_predicted / max(1, total_predicted) if total_predicted > 0 else 0,
                            "examples_seen": examples_processed
                        }
                    )
                    
                    add_log_message(
                        self.job_id,
                        f"Examples processed: {examples_processed}, "
                        f"Loss: {running_loss / max(1, examples_processed % (update_interval * 10)):.4f}, "
                        f"Accuracy: {correctly_predicted / max(1, total_predicted) if total_predicted > 0 else 0:.4f}"
                    )
                    
                    # Reset metrics periodically
                    if examples_processed % (update_interval * 10) == 0:
                        running_loss = 0.0
                        correctly_predicted = 0
                        total_predicted = 0
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Add example to recent examples queue
                recent_examples.append({k: v.clone() for k, v in batch.items()})
                
                # Only update model every update_interval examples
                if examples_processed % update_interval == 0:
                    # Update model using recent examples
                    model.train()
                    
                    # Create mini-batch from recent examples
                    mini_batch_size = min(32, len(recent_examples))
                    indices = np.random.choice(len(recent_examples), mini_batch_size, replace=False)
                    
                    mini_batch = {}
                    for idx in indices:
                        example = recent_examples[idx]
                        for k, v in example.items():
                            if k not in mini_batch:
                                mini_batch[k] = []
                            mini_batch[k].append(v)
                    
                    # Stack tensors
                    mini_batch = {k: torch.stack(v) for k, v in mini_batch.items()}
                    
                    # Forward pass
                    outputs = model(**mini_batch)
                    loss = outputs.loss
                    
                    # Apply forget factor to gradually forget old examples
                    loss = loss * (1 - forget_factor)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Track metrics
                    running_loss += loss.item()
                    
                    # Calculate accuracy if this is a classification task
                    if 'labels' in mini_batch and hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        correctly_predicted += (predictions == mini_batch['labels']).sum().item()
                        total_predicted += mini_batch['labels'].size(0)
                
                # Periodically save the model
                if examples_processed % (update_interval * 10) == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{examples_processed}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    
                    add_log_message(self.job_id, f"Saved checkpoint at {examples_processed} examples")
            
            # Training complete
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            update_training_status(self.job_id, status="completed", progress=100)
            add_log_message(self.job_id, "Online learning completed successfully")
            
            return {
                "success": True,
                "model_path": output_dir,
                "examples_processed": examples_processed,
                "status": "completed"
            }
            
        except Exception as e:
            # Update status to error
            update_training_status(self.job_id, status="error", error=str(e))
            add_log_message(self.job_id, f"Error during online learning: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            save_path: Path to save the model (if None, use default path)
            
        Returns:
            Path where model was saved
        """
        import os
        
        # Default path if none provided
        if save_path is None:
            save_path = f"models/trained/{self.model_id}"
        
        # Get model components
        try:
            model_components = self.load_model()
            model = model_components['model']
            tokenizer = model_components['tokenizer']
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            return save_path
        except Exception as e:
            raise ValueError(f"Failed to save model: {str(e)}")


class FederatedLearningTrainer(BaseTrainer):
    """
    Trainer for federated learning
    """
    def __init__(self, model_id: str, model_config: Dict[str, Any]):
        """
        Initialize federated learning trainer
        
        Args:
            model_id: Unique identifier for the model
            model_config: Configuration for the model
        """
        super().__init__(model_id, model_config)
        self.job_id = None
    
    def load_model(self) -> Any:
        """
        Load model for federated learning
        
        Returns:
            Loaded model
        """
        import torch
        import transformers
        
        model_path = self.model_config.get('model_path')
        model_type = self.model_config.get('model_type', 'auto')
        
        if model_type == 't5':
            # Load T5 model
            tokenizer = transformers.T5Tokenizer.from_pretrained(model_path)
            config = transformers.T5Config.from_pretrained(model_path)
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
        else:
            # Default to auto model
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
            config = transformers.AutoConfig.from_pretrained(model_path)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'config': config
            }
    
    def prepare_dataset(self, dataset: DatasetHandler, input_field: str, output_field: str) -> Any:
        """
        Prepare dataset for federated learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Field containing output data
            
        Returns:
            Prepared dataset for training
        """
        import torch
        import transformers
        import numpy as np
        
        # Get dataset
        train_data, val_data = dataset.get_train_val_split()
        
        # Get model components
        model_components = self.load_model()
        tokenizer = model_components['tokenizer']
        
        # Process texts
        train_texts = train_data[input_field]
        train_labels = train_data[output_field] if output_field in train_data else None
        
        # For federated learning, we need to simulate multiple clients
        # We'll split the dataset among simulated clients
        num_clients = self.training_config.get('num_clients', 10)
        
        # Tokenize all inputs
        all_encodings = tokenizer(
            train_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=self.training_config.get('max_seq_length', 128)
        )
        
        # Create dataset class
        class FederatedDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels=None):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                if self.labels is not None:
                    item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.encodings['input_ids'])
        
        # Split data among clients (simulate heterogeneous data distribution)
        client_datasets = []
        
        # Get indices for each client
        indices = np.arange(len(train_texts))
        np.random.shuffle(indices)
        
        # Split indices among clients
        client_indices = np.array_split(indices, num_clients)
        
        # Create datasets for each client
        for client_idx in client_indices:
            # Get subset of encodings for this client
            client_encodings = {key: [val[i] for i in client_idx] for key, val in all_encodings.items()}
            
            # Get labels for this client if available
            client_labels = None
            if train_labels is not None:
                client_labels = [train_labels[i] for i in client_idx]
            
            # Create dataset for this client
            client_dataset = FederatedDataset(client_encodings, client_labels)
            client_datasets.append(client_dataset)
        
        # Process validation data if available
        val_dataset = None
        if val_data is not None:
            val_texts = val_data[input_field]
            val_labels = val_data[output_field] if output_field in val_data else None
            
            val_encodings = tokenizer(
                val_texts, 
                truncation=True, 
                padding="max_length", 
                max_length=self.training_config.get('max_seq_length', 128)
            )
            
            val_dataset = FederatedDataset(val_encodings, val_labels)
        
        return {
            'client_datasets': client_datasets,
            'val_dataset': val_dataset,
            'model_components': model_components
        }
    
    def train(self, dataset: DatasetHandler, input_field: str, output_field: str) -> Dict[str, Any]:
        """
        Train model with federated learning
        
        Args:
            dataset: Dataset handler
            input_field: Field containing input data
            output_field: Field containing output data
            
        Returns:
            Dictionary with training results
        """
        import os
        import torch
        import copy
        import transformers
        from api.monitor import register_training_job, update_training_status, add_log_message
        
        # Get prepared datasets
        prepared_data = self.prepare_dataset(dataset, input_field, output_field)
        client_datasets = prepared_data['client_datasets']
        val_dataset = prepared_data['val_dataset']
        model_components = prepared_data['model_components']
        
        # Setup training parameters
        global_model = model_components['model']
        tokenizer = model_components['tokenizer']
        
        # Create output directory
        output_dir = self.training_config.get('output_dir', 'models/trained')
        os.makedirs(output_dir, exist_ok=True)
        
        # Federated learning parameters
        num_rounds = self.training_config.get('num_rounds', 100)
        client_fraction = self.training_config.get('client_fraction', 0.2)
        local_epochs = self.training_config.get('local_epochs', 2)
        batch_size = self.training_config.get('batch_size', 16)
        learning_rate = self.training_config.get('learning_rate', 1e-4)
        aggregation = self.training_config.get('aggregation', 'fedavg')
        
        # Register training job
        self.job_id = dataset.name + '_' + self.model_id + '_' + str(int(time.time()))
        register_training_job(
            self.job_id, 
            self.model_id, 
            dataset.name, 
            self.training_config
        )
        
        try:
            # Update status to training
            update_training_status(self.job_id, status="training", progress=0)
            add_log_message(self.job_id, "Starting federated learning training")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            global_model.to(device)
            
            # For evaluating the global model
            eval_global_model = lambda: self._evaluate_model(global_model, val_dataset, device, batch_size)
            
            # Initial evaluation
            if val_dataset is not None:
                initial_metrics = eval_global_model()
                add_log_message(self.job_id, f"Initial global model metrics: {initial_metrics}")
            
            # Main federated learning loop
            for round_num in range(num_rounds):
                # Check for stop request
                from api.monitor import check_stop_requested
                if check_stop_requested(self.job_id):
                    update_training_status(
                        self.job_id,
                        status="stopped",
                        current_step=round_num
                    )
                    add_log_message(self.job_id, "Training stopped by user request")
                    
                    # Save current global model
                    global_model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    
                    return {
                        "success": True,
                        "model_path": output_dir,
                        "status": "stopped"
                    }
                
                # Update progress
                progress = min(100, int((round_num / num_rounds) * 100))
                update_training_status(
                    self.job_id,
                    status="training",
                    current_step=round_num,
                    progress=progress
                )
                
                add_log_message(self.job_id, f"Starting round {round_num + 1}/{num_rounds}")
                
                # Select random subset of clients
                num_selected_clients = max(1, int(client_fraction * len(client_datasets)))
                selected_clients = np.random.choice(
                    len(client_datasets), 
                    size=num_selected_clients, 
                    replace=False
                )
                
                # Train selected clients locally
                client_models = []
                client_sizes = []
                
                for client_idx in selected_clients:
                    # Initialize client model with global model
                    client_model = copy.deepcopy(global_model)
                    client_dataset = client_datasets[client_idx]
                    
                    # Create dataloader for this client
                    client_dataloader = torch.utils.data.DataLoader(
                        client_dataset, 
                        batch_size=batch_size, 
                        shuffle=True
                    )
                    
                    # Create optimizer for client model
                    client_optimizer = torch.optim.Adam(client_model.parameters(), lr=learning_rate)
                    
                    # Train client model
                    client_model.train()
                    for epoch in range(local_epochs):
                        for batch in client_dataloader:
                            # Move batch to device
                            batch = {k: v.to(device) for k, v in batch.items()}
                            
                            # Forward pass
                            outputs = client_model(**batch)
                            loss = outputs.loss
                            
                            # Backward pass and optimize
                            client_optimizer.zero_grad()
                            loss.backward()
                            client_optimizer.step()
                    
                    # Add client model to collection
                    client_models.append(client_model)
                    client_sizes.append(len(client_dataset))
                    
                    add_log_message(self.job_id, f"Client {client_idx} completed local training")
                
                # Aggregate client models
                self._aggregate_models(global_model, client_models, client_sizes, aggregation)
                
                add_log_message(self.job_id, f"Aggregated {len(client_models)} client models")
                
                # Evaluate global model
                if val_dataset is not None and (round_num + 1) % 5 == 0:
                    metrics = eval_global_model()
                    add_log_message(self.job_id, f"Round {round_num + 1} global model metrics: {metrics}")
                    
                    update_training_status(
                        self.job_id,
                        metrics=metrics
                    )
                
                # Save checkpoint
                if (round_num + 1) % 10 == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{round_num + 1}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    global_model.save_pretrained(checkpoint_dir)
                    
                    add_log_message(self.job_id, f"Saved checkpoint after round {round_num + 1}")
            
            # Training complete
            global_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Final evaluation
            final_metrics = {}
            if val_dataset is not None:
                final_metrics = eval_global_model()
                add_log_message(self.job_id, f"Final global model metrics: {final_metrics}")
            
            update_training_status(self.job_id, status="completed", progress=100)
            add_log_message(self.job_id, "Federated learning completed successfully")
            
            return {
                "success": True,
                "model_path": output_dir,
                "final_metrics": final_metrics,
                "status": "completed"
            }
            
        except Exception as e:
            # Update status to error
            update_training_status(self.job_id, status="error", error=str(e))
            add_log_message(self.job_id, f"Error during federated learning: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def _evaluate_model(self, model, dataset, device, batch_size):
        """
        Evaluate model on dataset
        
        Args:
            model: Model to evaluate
            dataset: Dataset to evaluate on
            device: Device to use
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if dataset is None:
            return {}
        
        import torch
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
                
                # Calculate accuracy if this is a classification task
                if 'labels' in batch and hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == batch['labels']).sum().item()
                    total += batch['labels'].size(0)
        
        metrics = {"loss": total_loss / len(dataloader)}
        if total > 0:
            metrics["accuracy"] = correct / total
        
        return metrics
    
    def _aggregate_models(self, global_model, client_models, client_sizes, aggregation='fedavg'):
        """
        Aggregate client models into global model
        
        Args:
            global_model: Global model to update
            client_models: List of client models
            client_sizes: List of client dataset sizes
            aggregation: Aggregation method to use ('fedavg', 'fedsgd', etc.)
        """
        import torch
        
        # FedAvg - weighted average based on dataset size
        if aggregation.lower() == 'fedavg':
            # Calculate total size
            total_size = sum(client_sizes)
            
            # Get state dictionary of global model
            global_state_dict = global_model.state_dict()
            
            # Initialize aggregated parameters with zeros
            for key in global_state_dict:
                global_state_dict[key] = torch.zeros_like(global_state_dict[key])
            
            # Weighted average of client models
            for i, client_model in enumerate(client_models):
                client_state_dict = client_model.state_dict()
                weight = client_sizes[i] / total_size
                
                for key in global_state_dict:
                    global_state_dict[key] += client_state_dict[key] * weight
            
            # Load aggregated parameters into global model
            global_model.load_state_dict(global_state_dict)
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save trained model
        
        Args:
            save_path: Path to save the model (if None, use default path)
            
        Returns:
            Path where model was saved
        """
        import os
        
        # Default path if none provided
        if save_path is None:
            save_path = f"models/trained/{self.model_id}"
        
        # Get model components
        try:
            model_components = self.load_model()
            model = model_components['model']
            tokenizer = model_components['tokenizer']
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            return save_path
        except Exception as e:
            raise ValueError(f"Failed to save model: {str(e)}")


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
        return ReinforcementTrainer(model_id, model_config)
    elif learning_type == "semi_supervised":
        return SemiSupervisedTrainer(model_id, model_config)
    elif learning_type == "self_supervised":
        return SelfSupervisedTrainer(model_id, model_config)
    elif learning_type == "online":
        return OnlineLearningTrainer(model_id, model_config)
    elif learning_type == "federated":
        return FederatedLearningTrainer(model_id, model_config)
    else:
        raise ValueError(f"Unsupported learning type: {learning_type}")