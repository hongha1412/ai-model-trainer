<template>
  <div class="dashboard">
    <div class="dashboard-header">
      <h2 class="section-title">Models Dashboard</h2>
      <button @click="refreshModels" class="btn btn-secondary">
        <span v-if="modelStore.loading" class="loading-spinner"></span>
        <span v-else>Refresh</span>
      </button>
    </div>
    
    <div v-if="modelStore.error" class="alert alert-error">
      {{ modelStore.error }}
    </div>
    
    <div class="card">
      <h3 class="card-title">Available Models</h3>
      <p v-if="modelStore.models.length === 0 && !modelStore.loading">
        No models are currently loaded. Add a model in the <router-link to="/model-config">Model Config</router-link> section.
      </p>
      
      <div v-else>
        <table class="table">
          <thead>
            <tr>
              <th>Model ID</th>
              <th>Type</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="model in modelStore.models" :key="model.id">
              <td>{{ model.id }}</td>
              <td>{{ model.type }}</td>
              <td>
                <span v-if="model.ready" class="badge badge-success">Ready</span>
                <span v-else class="badge badge-warning">Loading</span>
              </td>
              <td>
                <div class="btn-group">
                  <router-link :to="'/model-test/' + model.id" class="btn btn-secondary btn-sm">Test</router-link>
                  <button @click="unloadModel(model.id)" class="btn btn-danger btn-sm" :disabled="modelStore.loading">
                    Unload
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    
    <div class="card">
      <h3 class="card-title">Available APIs</h3>
      <div class="api-list">
        <div class="api-item">
          <div class="api-info">
            <h4 class="api-name">Chat Completions</h4>
            <p class="api-description">
              Generate chat completions from models.
            </p>
          </div>
          <div class="api-example">
            <pre class="code-block">
POST /v1/chat/completions
{
  "model": "model_id",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
}</pre>
          </div>
        </div>
        
        <div class="api-item">
          <div class="api-info">
            <h4 class="api-name">Completions</h4>
            <p class="api-description">
              Generate text completions from models.
            </p>
          </div>
          <div class="api-example">
            <pre class="code-block">
POST /v1/completions
{
  "model": "model_id",
  "prompt": "Once upon a time",
  "max_tokens": 50
}</pre>
          </div>
        </div>
        
        <div class="api-item">
          <div class="api-info">
            <h4 class="api-name">Models</h4>
            <p class="api-description">
              List and retrieve model information.
            </p>
          </div>
          <div class="api-example">
            <pre class="code-block">
GET /v1/models
GET /v1/models/{model_id}</pre>
          </div>
        </div>
      </div>
      
      <div class="mt-4">
        <router-link to="/openapi" class="btn btn-primary">View Full API Documentation</router-link>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useModelStore } from '../store/models'

const modelStore = useModelStore()

onMounted(() => {
  refreshModels()
})

async function refreshModels() {
  await modelStore.fetchModels()
}

async function unloadModel(modelId: string) {
  if (confirm(`Are you sure you want to unload model "${modelId}"?`)) {
    const result = await modelStore.unloadModel(modelId)
    if (result.success) {
      alert(result.message || 'Model unloaded successfully')
    } else {
      alert(result.message || 'Failed to unload model')
    }
  }
}
</script>

<style scoped>
.dashboard-header {
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

.btn-sm {
  padding: 4px 8px;
  font-size: 0.875rem;
}

.btn-group {
  display: flex;
  gap: 8px;
}

.api-list {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-lg);
}

.api-item {
  display: flex;
  gap: var(--spacing-lg);
  flex-wrap: wrap;
}

.api-info {
  flex: 1;
  min-width: 200px;
}

.api-example {
  flex: 2;
  min-width: 300px;
}

.api-name {
  margin-top: 0;
  margin-bottom: var(--spacing-xs);
  font-size: 1.125rem;
  font-weight: 600;
}

.api-description {
  margin-top: 0;
  color: var(--color-text-light);
}

.mt-4 {
  margin-top: 24px;
}

@media (max-width: 768px) {
  .api-item {
    flex-direction: column;
  }
  
  .api-example {
    width: 100%;
  }
}
</style>