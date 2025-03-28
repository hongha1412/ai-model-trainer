import { defineStore } from 'pinia'
import axios from 'axios'

export interface TrainingConfig {
  learning_type: string
  batch_size: number
  learning_rate: number
  num_train_epochs: number
  weight_decay: number
  warmup_steps: number
  max_seq_length: number
  save_steps?: number
  evaluation_strategy?: string
  discount_factor?: number
  target_update_interval?: number
  replay_buffer_size?: number
  exploration_rate?: number
  max_steps?: number
  unlabeled_weight?: number
  window_size?: number
  forget_factor?: number
  update_interval?: number
  max_samples?: number
  num_clients?: number
  client_fraction?: number
  local_epochs?: number
  num_rounds?: number
  aggregation?: string
  [key: string]: any
}

export interface DatasetInfo {
  name: string
  format: string
  size: number
  preview?: string
  columns?: string[]
  created: string
}

export interface ModelInfo {
  id: string
  name: string
  type: string
  task?: string
  size?: number
  description?: string
  source: 'huggingface' | 'local'
  path?: string
}

export interface HuggingFaceTask {
  id: string
  name: string
  description: string
}

export interface TrainingState {
  defaultConfigs: Record<string, TrainingConfig>
  currentConfig: TrainingConfig | null
  datasets: DatasetInfo[]
  browsedDirectories: {
    path: string
    directories: string[]
    files: string[]
  }
  huggingfaceModels: ModelInfo[]
  huggingfaceTasks: HuggingFaceTask[]
  selectedModel: ModelInfo | null
  selectedDataset: DatasetInfo | null
  trainingStatus: {
    isTraining: boolean
    progress: number
    log: string[]
    error: string | null
  }
  loading: boolean
  error: string | null
}

export const useTrainingStore = defineStore('training', {
  state: (): TrainingState => ({
    defaultConfigs: {},
    currentConfig: null,
    datasets: [],
    browsedDirectories: {
      path: '',
      directories: [],
      files: []
    },
    huggingfaceModels: [],
    huggingfaceTasks: [],
    selectedModel: null,
    selectedDataset: null,
    trainingStatus: {
      isTraining: false,
      progress: 0,
      log: [],
      error: null
    },
    loading: false,
    error: null
  }),

  actions: {
    async fetchDefaultConfigs() {
      this.loading = true
      try {
        const response = await axios.get('/api/training/config')
        if (response.data && response.data.default_configs) {
          this.defaultConfigs = response.data.default_configs
        }
        this.error = null
      } catch (error: any) {
        console.error('Error fetching default configs:', error)
        this.error = error.message || 'Failed to fetch default training configurations'
      } finally {
        this.loading = false
      }
    },

    async saveTrainingConfig(config: TrainingConfig) {
      this.loading = true
      try {
        const response = await axios.post('/api/training/config', { config })
        this.currentConfig = response.data.config
        return { success: true, message: 'Training configuration saved successfully' }
      } catch (error: any) {
        console.error('Error saving training config:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to save training configuration'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async fetchDatasets() {
      this.loading = true
      try {
        const response = await axios.get('/api/training/datasets')
        if (response.data && response.data.datasets) {
          this.datasets = response.data.datasets
        }
        this.error = null
      } catch (error: any) {
        console.error('Error fetching datasets:', error)
        this.error = error.message || 'Failed to fetch datasets'
      } finally {
        this.loading = false
      }
    },

    async uploadDataset(file: File) {
      this.loading = true
      try {
        const formData = new FormData()
        formData.append('file', file)
        
        const response = await axios.post('/api/training/datasets/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        })
        
        // Refresh datasets after upload
        await this.fetchDatasets()
        
        return { success: true, message: response.data.message || 'Dataset uploaded successfully' }
      } catch (error: any) {
        console.error('Error uploading dataset:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to upload dataset'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async deleteDataset(filename: string) {
      this.loading = true
      try {
        await axios.delete(`/api/training/datasets/${filename}`)
        
        // Refresh datasets after deletion
        await this.fetchDatasets()
        
        return { success: true, message: 'Dataset deleted successfully' }
      } catch (error: any) {
        console.error('Error deleting dataset:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to delete dataset'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async getDatasetInfo(filename: string) {
      this.loading = true
      try {
        const response = await axios.get(`/api/training/datasets/${filename}`)
        return { success: true, data: response.data }
      } catch (error: any) {
        console.error('Error getting dataset info:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to get dataset information'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async searchHuggingFaceModels(query: string, task?: string) {
      this.loading = true
      try {
        const params: Record<string, string> = { query }
        if (task) {
          params.task = task
        }
        
        const response = await axios.get('/api/training/huggingface/search', { params })
        if (response.data && response.data.models) {
          this.huggingfaceModels = response.data.models
        }
        this.error = null
        return { success: true, models: this.huggingfaceModels }
      } catch (error: any) {
        console.error('Error searching Hugging Face models:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to search Hugging Face models'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async getHuggingFaceTasks() {
      this.loading = true
      try {
        const response = await axios.get('/api/training/huggingface/tasks')
        if (response.data && response.data.tasks) {
          this.huggingfaceTasks = response.data.tasks
        }
        this.error = null
        return { success: true, tasks: this.huggingfaceTasks }
      } catch (error: any) {
        console.error('Error fetching Hugging Face tasks:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to fetch Hugging Face tasks'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async getHuggingFaceModelInfo(modelId: string) {
      this.loading = true
      try {
        const response = await axios.get(`/api/training/huggingface/models/${modelId}`)
        return { success: true, data: response.data }
      } catch (error: any) {
        console.error('Error getting Hugging Face model info:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to get Hugging Face model information'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async downloadHuggingFaceModel(modelId: string, modelClass: string) {
      this.loading = true
      try {
        const response = await axios.post('/api/training/huggingface/download', {
          model_id: modelId,
          model_class: modelClass
        })
        
        return { success: true, message: response.data.message || 'Model downloaded successfully', path: response.data.path }
      } catch (error: any) {
        console.error('Error downloading Hugging Face model:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to download Hugging Face model'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async browseDirectories(path?: string) {
      this.loading = true
      try {
        const params: Record<string, string> = {}
        if (path) {
          params.path = path
        }
        
        const response = await axios.get('/api/training/browse', { params })
        if (response.data) {
          this.browsedDirectories = {
            path: response.data.current_path,
            directories: response.data.directories,
            files: response.data.files
          }
        }
        this.error = null
        return { success: true, data: this.browsedDirectories }
      } catch (error: any) {
        console.error('Error browsing directories:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to browse directories'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },

    async trainModel(params: {
      model_source: 'huggingface' | 'local',
      model_id?: string,
      model_path?: string,
      dataset_name: string,
      input_field: string,
      output_field?: string,
      learning_type: string,
      config: TrainingConfig,
      output_dir: string
    }) {
      this.trainingStatus = {
        isTraining: true,
        progress: 0,
        log: ['Starting training...'],
        error: null
      }
      
      try {
        const response = await axios.post('/api/training/train', params)
        
        // Simulate updating training logs every second
        // In a real application, you would use WebSockets or Server-Sent Events
        const trainingInterval = setInterval(() => {
          this.trainingStatus.progress += 5
          this.trainingStatus.log.push(`Training progress: ${this.trainingStatus.progress}%`)
          
          if (this.trainingStatus.progress >= 100) {
            clearInterval(trainingInterval)
            this.trainingStatus.isTraining = false
            this.trainingStatus.log.push('Training completed successfully!')
          }
        }, 1000)
        
        return { success: true, message: response.data.message || 'Training started successfully' }
      } catch (error: any) {
        console.error('Error starting training:', error)
        this.trainingStatus.isTraining = false
        this.trainingStatus.error = error.response?.data?.error || error.message || 'Failed to start training'
        this.trainingStatus.log.push(`Error: ${this.trainingStatus.error}`)
        return { success: false, message: this.trainingStatus.error }
      }
    },

    setSelectedModel(model: ModelInfo | null) {
      this.selectedModel = model
    },

    setSelectedDataset(dataset: DatasetInfo | null) {
      this.selectedDataset = dataset
    },

    updateCurrentConfig(config: TrainingConfig) {
      this.currentConfig = { ...config }
    },

    resetTrainingStatus() {
      this.trainingStatus = {
        isTraining: false,
        progress: 0,
        log: [],
        error: null
      }
    }
  }
})