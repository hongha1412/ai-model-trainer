// Frontend store test script
// Run with: npm test

import { describe, it, beforeEach, expect, vi } from 'vitest';
import { setActivePinia, createPinia } from 'pinia';
import { useModelsStore } from '../frontend/src/store/models';
import { useTrainingStore } from '../frontend/src/store/training';
import axios from 'axios';

// Mock axios
vi.mock('axios');

describe('Models Store', () => {
  beforeEach(() => {
    // Create a fresh pinia instance for each test
    setActivePinia(createPinia());
    
    // Reset axios mocks
    vi.resetAllMocks();
  });
  
  it('fetches models correctly', async () => {
    // Setup mock response
    const mockModels = {
      models: [
        { id: 'model1', name: 'Model 1', loaded: true },
        { id: 'model2', name: 'Model 2', loaded: false }
      ]
    };
    
    axios.get.mockResolvedValue({ data: mockModels });
    
    // Get store and call action
    const store = useModelsStore();
    await store.fetchModels();
    
    // Assertions
    expect(axios.get).toHaveBeenCalledWith('/api/models');
    expect(store.models).toEqual(mockModels.models);
    expect(store.error).toBeNull();
  });
  
  it('handles fetch models error', async () => {
    // Setup mock error
    const error = new Error('API Error');
    axios.get.mockRejectedValue(error);
    
    // Get store and call action
    const store = useModelsStore();
    await store.fetchModels();
    
    // Assertions
    expect(axios.get).toHaveBeenCalledWith('/api/models');
    expect(store.error).toBe('Failed to fetch models: API Error');
  });
  
  it('loads a model correctly', async () => {
    // Setup mock response
    const mockResponse = { success: true, message: 'Model loaded' };
    axios.post.mockResolvedValue({ data: mockResponse });
    
    // Get store and call action
    const store = useModelsStore();
    const result = await store.loadModel('test-model');
    
    // Assertions
    expect(axios.post).toHaveBeenCalledWith('/api/load-model/test-model');
    expect(result.success).toBe(true);
    expect(store.error).toBeNull();
  });
  
  it('handles load model error', async () => {
    // Setup mock error
    const error = new Error('API Error');
    axios.post.mockRejectedValue(error);
    
    // Get store and call action
    const store = useModelsStore();
    const result = await store.loadModel('test-model');
    
    // Assertions
    expect(axios.post).toHaveBeenCalledWith('/api/load-model/test-model');
    expect(result.success).toBe(false);
    expect(store.error).toBe('Failed to load model: API Error');
  });
});

describe('Training Store', () => {
  beforeEach(() => {
    // Create a fresh pinia instance for each test
    setActivePinia(createPinia());
    
    // Reset axios mocks
    vi.resetAllMocks();
  });
  
  it('fetches default configs correctly', async () => {
    // Setup mock response
    const mockConfigs = {
      learning_types: {
        supervised: {
          batch_size: 8,
          learning_rate: 2e-5
        },
        unsupervised: {
          batch_size: 16,
          learning_rate: 1e-4
        }
      }
    };
    
    axios.get.mockResolvedValue({ data: mockConfigs });
    
    // Get store and call action
    const store = useTrainingStore();
    await store.fetchDefaultConfigs();
    
    // Assertions
    expect(axios.get).toHaveBeenCalledWith('/api/training/config');
    expect(store.defaultConfigs).toEqual(mockConfigs.learning_types);
    expect(store.error).toBeNull();
  });
  
  it('handles fetch default configs error', async () => {
    // Setup mock error
    const error = new Error('API Error');
    axios.get.mockRejectedValue(error);
    
    // Get store and call action
    const store = useTrainingStore();
    await store.fetchDefaultConfigs();
    
    // Assertions
    expect(axios.get).toHaveBeenCalledWith('/api/training/config');
    expect(store.error).toBe('Failed to fetch training configs: API Error');
  });
  
  it('fetches datasets correctly', async () => {
    // Setup mock response
    const mockDatasets = {
      datasets: [
        { name: 'dataset1.csv', format: 'csv', size: 1024 },
        { name: 'dataset2.json', format: 'json', size: 2048 }
      ]
    };
    
    axios.get.mockResolvedValue({ data: mockDatasets });
    
    // Get store and call action
    const store = useTrainingStore();
    await store.fetchDatasets();
    
    // Assertions
    expect(axios.get).toHaveBeenCalledWith('/api/datasets');
    expect(store.datasets).toEqual(mockDatasets.datasets);
    expect(store.error).toBeNull();
  });
  
  it('handles fetch datasets error', async () => {
    // Setup mock error
    const error = new Error('API Error');
    axios.get.mockRejectedValue(error);
    
    // Get store and call action
    const store = useTrainingStore();
    await store.fetchDatasets();
    
    // Assertions
    expect(axios.get).toHaveBeenCalledWith('/api/datasets');
    expect(store.error).toBe('Failed to fetch datasets: API Error');
  });
  
  it('trains a model correctly', async () => {
    // Setup mock response
    const mockJobId = 'job123';
    const mockResponse = { 
      success: true, 
      job_id: mockJobId, 
      message: 'Training started' 
    };
    
    axios.post.mockResolvedValue({ data: mockResponse });
    
    // Training parameters
    const trainingParams = {
      model_source: 'huggingface',
      model_id: 't5-small',
      dataset_name: 'test_dataset.csv',
      input_field: 'input',
      output_field: 'output',
      learning_type: 'supervised',
      config: {
        batch_size: 8,
        learning_rate: 2e-5
      }
    };
    
    // Get store and call action
    const store = useTrainingStore();
    const result = await store.trainModel(trainingParams);
    
    // Assertions
    expect(axios.post).toHaveBeenCalledWith('/api/train', trainingParams);
    expect(result.success).toBe(true);
    expect(store.trainingStatus.jobId).toBe(mockJobId);
    expect(store.trainingStatus.isTraining).toBe(true);
    expect(store.error).toBeNull();
  });
  
  it('handles train model error', async () => {
    // Setup mock error
    const error = new Error('API Error');
    axios.post.mockRejectedValue(error);
    
    // Training parameters
    const trainingParams = {
      model_source: 'huggingface',
      model_id: 't5-small',
      dataset_name: 'test_dataset.csv',
      learning_type: 'supervised',
      config: {}
    };
    
    // Get store and call action
    const store = useTrainingStore();
    const result = await store.trainModel(trainingParams);
    
    // Assertions
    expect(axios.post).toHaveBeenCalledWith('/api/train', trainingParams);
    expect(result.success).toBe(false);
    expect(store.error).toBe('Failed to start training: API Error');
  });
  
  it('gets training status correctly', async () => {
    // Setup mock response
    const mockStatus = { 
      progress: 50, 
      status: 'training', 
      current_epoch: 2, 
      current_step: 100,
      metrics: { loss: 0.5, accuracy: 0.8 },
      log: ['Starting epoch 2', 'Step 100 completed']
    };
    
    axios.get.mockResolvedValue({ data: mockStatus });
    
    // Get store and call action
    const store = useTrainingStore();
    await store.getTrainingStatus('job123');
    
    // Assertions
    expect(axios.get).toHaveBeenCalledWith('/api/monitor/jobs/job123');
    expect(store.trainingStatus.progress).toBe(50);
    expect(store.trainingStatus.status).toBe('training');
    expect(store.trainingStatus.currentEpoch).toBe(2);
    expect(store.trainingStatus.currentStep).toBe(100);
    expect(store.trainingStatus.metrics).toEqual({ loss: 0.5, accuracy: 0.8 });
    expect(store.trainingStatus.log).toEqual(['Starting epoch 2', 'Step 100 completed']);
    expect(store.error).toBeNull();
  });
  
  it('handles get training status error', async () => {
    // Setup mock error
    const error = new Error('API Error');
    axios.get.mockRejectedValue(error);
    
    // Get store and call action
    const store = useTrainingStore();
    await store.getTrainingStatus('job123');
    
    // Assertions
    expect(axios.get).toHaveBeenCalledWith('/api/monitor/jobs/job123');
    expect(store.error).toBe('Failed to get training status: API Error');
  });
  
  it('stops training job correctly', async () => {
    // Setup mock response
    const mockResponse = { 
      message: 'Request to stop training job job123 has been sent' 
    };
    
    axios.post.mockResolvedValue({ data: mockResponse });
    
    // Get store and call action
    const store = useTrainingStore();
    const result = await store.stopTrainingJob('job123');
    
    // Assertions
    expect(axios.post).toHaveBeenCalledWith('/api/monitor/jobs/job123/stop');
    expect(result.success).toBe(true);
    expect(store.trainingStatus.status).toBe('stopping');
    expect(result.message).toBe('Request to stop training job job123 has been sent');
  });
  
  it('handles stop training job error', async () => {
    // Setup mock error
    const error = new Error('API Error');
    axios.post.mockRejectedValue(error);
    
    // Get store and call action
    const store = useTrainingStore();
    const result = await store.stopTrainingJob('job123');
    
    // Assertions
    expect(axios.post).toHaveBeenCalledWith('/api/monitor/jobs/job123/stop');
    expect(result.success).toBe(false);
    expect(result.message).toBe('Failed to stop training job');
  });
});