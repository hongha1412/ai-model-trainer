<template>
  <div class="model-config">
    <div class="config-header">
      <h2 class="section-title">Model Configuration</h2>
      <button @click="refreshConfig" class="btn btn-secondary">
        <span v-if="modelStore.loading" class="loading-spinner"></span>
        <span v-else>Refresh</span>
      </button>
    </div>
    
    <div v-if="modelStore.error" class="alert alert-error">
      {{ modelStore.error }}
    </div>
    
    <div class="card">
      <h3 class="card-title">Add New Model</h3>
      <form @submit.prevent="saveModel">
        <div class="form-group">
          <label for="model-id" class="form-label">Model ID</label>
          <input 
            type="text" 
            id="model-id" 
            v-model="newModel.id"
            class="form-control"
            placeholder="e.g., t5-small" 
            required
          />
          <small class="form-text">A unique identifier for your model</small>
        </div>
        
        <div class="form-group">
          <label for="model-path" class="form-label">Model Path</label>
          <input 
            type="text" 
            id="model-path" 
            v-model="newModel.path"
            class="form-control"
            placeholder="e.g., /path/to/model" 
            required
          />
          <small class="form-text">Full path to the model files on disk</small>
        </div>
        
        <div class="form-group">
          <label for="model-type" class="form-label">Model Type</label>
          <select id="model-type" v-model="newModel.type" class="form-control" required>
            <option value="t5">T5</option>
            <option value="gpt2">GPT-2</option>
            <option value="custom">Custom</option>
          </select>
          <small class="form-text">The architecture of your model</small>
        </div>
        
        <div class="form-actions">
          <button type="submit" class="btn btn-primary" :disabled="modelStore.loading">
            <span v-if="modelStore.loading" class="loading-spinner"></span>
            <span v-else>Add Model</span>
          </button>
        </div>
      </form>
    </div>
    
    <div class="card" v-if="modelStore.modelConfig && Object.keys(modelStore.modelConfig.models).length > 0">
      <h3 class="card-title">Configured Models</h3>
      <table class="table">
        <thead>
          <tr>
            <th>Model ID</th>
            <th>Path</th>
            <th>Type</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(config, id) in modelStore.modelConfig.models" :key="id">
            <td>{{ id }}</td>
            <td>{{ config.path }}</td>
            <td>{{ config.type }}</td>
            <td>
              <div class="btn-group">
                <button @click="loadModel(id)" class="btn btn-primary btn-sm" :disabled="modelStore.loading">
                  Load
                </button>
                <button @click="removeModel(id)" class="btn btn-danger btn-sm" :disabled="modelStore.loading">
                  Remove
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    
    <div class="card" v-else-if="modelStore.modelConfig">
      <h3 class="card-title">Configured Models</h3>
      <p>No models configured yet. Add a model using the form above.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useModelStore } from '../store/models'

const modelStore = useModelStore()

const newModel = ref({
  id: '',
  path: '',
  type: 't5'
})

onMounted(() => {
  refreshConfig()
})

async function refreshConfig() {
  await modelStore.fetchModelConfig()
}

async function saveModel() {
  const result = await modelStore.saveModelConfig(
    newModel.value.id,
    newModel.value.path,
    newModel.value.type
  )
  
  if (result.success) {
    alert(result.message || 'Model configuration saved successfully')
    newModel.value = { id: '', path: '', type: 't5' }
  } else {
    alert(result.message || 'Failed to save model configuration')
  }
}

async function loadModel(modelId: string) {
  const result = await modelStore.loadModel(modelId)
  if (result.success) {
    alert(result.message || 'Model loaded successfully')
  } else {
    alert(result.message || 'Failed to load model')
  }
}

async function removeModel(modelId: string) {
  if (confirm(`Are you sure you want to remove model "${modelId}" from configuration?`)) {
    const result = await modelStore.removeModelConfig(modelId)
    if (result.success) {
      alert('Model configuration removed successfully')
    } else {
      alert(result.message || 'Failed to remove model configuration')
    }
  }
}
</script>

<style scoped>
.config-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.section-title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
}

.form-text {
  display: block;
  margin-top: 4px;
  font-size: 0.875rem;
  color: var(--color-text-light);
}

.form-actions {
  margin-top: var(--spacing-lg);
}

.btn-sm {
  padding: 4px 8px;
  font-size: 0.875rem;
}

.btn-group {
  display: flex;
  gap: 8px;
}
</style>