#!/usr/bin/env python
"""
Test script for API endpoints
"""
import os
import sys
import unittest
import json
import tempfile
import requests
from io import BytesIO

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import app
from app import app

class APIEndpointTests(unittest.TestCase):
    """Test case for API endpoints"""
    
    def setUp(self):
        """Set up test client"""
        app.testing = True
        self.client = app.test_client()
        
    def test_api_models_endpoint(self):
        """Test /api/models endpoint"""
        response = self.client.get('/api/models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        
    def test_api_model_config_endpoint(self):
        """Test /api/model-config endpoint"""
        response = self.client.get('/api/model-config')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('models', data)
        
    def test_openai_compatible_models_endpoint(self):
        """Test OpenAI-compatible /v1/models endpoint"""
        response = self.client.get('/api/v1/models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        
    def test_training_config_endpoint(self):
        """Test /api/training/config endpoint"""
        response = self.client.get('/api/training/config')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('learning_types', data)
        
    def test_datasets_endpoint(self):
        """Test /api/training/datasets endpoint"""
        response = self.client.get('/api/training/datasets')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('datasets', data)

    def test_upload_dataset(self):
        """Test dataset upload endpoint"""
        # Create a temporary CSV file for testing
        csv_content = "input,output\nHello,World\nTest,Data"
        data = {'file': (BytesIO(csv_content.encode()), 'test_dataset.csv')}
        
        response = self.client.post(
            '/api/training/upload-dataset',
            data=data,
            content_type='multipart/form-data'
        )
        
        # Check if upload was successful
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'success')
        
        # Clean up - delete the dataset
        if 'filename' in data:
            self.client.delete(f"/api/training/datasets/{data['filename']}")
    
    def test_huggingface_tasks_endpoint(self):
        """Test /api/training/huggingface/tasks endpoint"""
        response = self.client.get('/api/training/huggingface/tasks')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('tasks', data)
        
    def test_monitor_jobs_endpoint(self):
        """Test /api/monitor/jobs endpoint"""
        response = self.client.get('/api/monitor/jobs')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('jobs', data)
        
    def test_browse_directories_endpoint(self):
        """Test /api/training/browse-dirs endpoint"""
        # Use the correct API endpoint path
        response = self.client.get('/api/training/browse-dirs')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('contents', data)
        self.assertIn('parent', data)
        self.assertIn('path', data)
        
    def test_openapi_spec_endpoint(self):
        """Test /openapi.json endpoint"""
        response = self.client.get('/openapi.json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('openapi', data)
        self.assertIn('paths', data)

class TrainingTypeTests(unittest.TestCase):
    """Test case for training types"""
    
    def setUp(self):
        """Import training module"""
        from training.trainer import create_trainer
        self.create_trainer = create_trainer
        
    def test_trainer_creation(self):
        """Test creation of different trainer types"""
        model_id = "test_model"
        model_config = {"model_path": "t5-small", "model_type": "t5"}
        
        # Test all trainer types
        learning_types = [
            "supervised", 
            "unsupervised", 
            "reinforcement", 
            "semi_supervised", 
            "self_supervised", 
            "online", 
            "federated"
        ]
        
        for learning_type in learning_types:
            trainer = self.create_trainer(learning_type, model_id, model_config)
            self.assertIsNotNone(trainer)
            # Check if trainer has the required methods
            self.assertTrue(hasattr(trainer, 'load_model'))
            self.assertTrue(hasattr(trainer, 'prepare_dataset'))
            self.assertTrue(hasattr(trainer, 'train'))
            self.assertTrue(hasattr(trainer, 'save_model'))
            
    def test_invalid_trainer_type(self):
        """Test invalid trainer type"""
        model_id = "test_model"
        model_config = {"model_path": "t5-small", "model_type": "t5"}
        
        with self.assertRaises(ValueError):
            self.create_trainer("invalid_type", model_id, model_config)

if __name__ == '__main__':
    unittest.main()