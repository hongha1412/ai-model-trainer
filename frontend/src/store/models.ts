import { defineStore } from 'pinia'
import axios from 'axios'

interface ModelInfo {
  id: string
  owner: string
  ready: boolean
  type: string
  created: number
}

interface ModelConfig {
  models: {
    [key: string]: {
      path: string
      type: string
    }
  }
}

interface State {
  models: ModelInfo[]
  modelConfig: ModelConfig | null
  loading: boolean
  error: string | null
}

export const useModelStore = defineStore('models', {
  state: (): State => ({
    models: [],
    modelConfig: null,
    loading: false,
    error: null
  }),
  
  actions: {
    async fetchModels() {
      this.loading = true
      try {
        const response = await axios.get('/api/models')
        if (response.data && response.data.data) {
          this.models = response.data.data
        }
        this.error = null
      } catch (error: any) {
        console.error('Error fetching models:', error)
        this.error = error.message || 'Failed to fetch models'
      } finally {
        this.loading = false
      }
    },
    
    async fetchModelConfig() {
      this.loading = true
      try {
        const response = await axios.get('/api/model-config')
        this.modelConfig = response.data
        this.error = null
      } catch (error: any) {
        console.error('Error fetching model config:', error)
        this.error = error.message || 'Failed to fetch model configuration'
      } finally {
        this.loading = false
      }
    },
    
    async saveModelConfig(modelId: string, modelPath: string, modelType: string) {
      this.loading = true
      try {
        const response = await axios.post('/api/model-config', {
          model_id: modelId,
          model_path: modelPath,
          model_type: modelType
        })
        
        // Refresh model config after saving
        await this.fetchModelConfig()
        
        return { success: true, message: response.data.message }
      } catch (error: any) {
        console.error('Error saving model config:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to save model configuration'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },
    
    async removeModelConfig(modelId: string) {
      this.loading = true
      try {
        await axios.delete(`/api/model-config/${modelId}`)
        
        // Refresh model config after removing
        await this.fetchModelConfig()
        
        return { success: true }
      } catch (error: any) {
        console.error('Error removing model config:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to remove model configuration'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },
    
    async loadModel(modelId: string) {
      this.loading = true
      try {
        const response = await axios.post(`/api/models/${modelId}/load`)
        
        // Refresh model list after loading
        await this.fetchModels()
        
        return { success: true, message: response.data.message }
      } catch (error: any) {
        console.error('Error loading model:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to load model'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    },
    
    async unloadModel(modelId: string) {
      this.loading = true
      try {
        const response = await axios.post(`/api/models/${modelId}/unload`)
        
        // Refresh model list after unloading
        await this.fetchModels()
        
        return { success: true, message: response.data.message }
      } catch (error: any) {
        console.error('Error unloading model:', error)
        this.error = error.response?.data?.error || error.message || 'Failed to unload model'
        return { success: false, message: this.error }
      } finally {
        this.loading = false
      }
    }
  }
})