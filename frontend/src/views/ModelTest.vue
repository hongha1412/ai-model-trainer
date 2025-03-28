<template>
  <div class="model-test">
    <div class="test-header">
      <h2 class="section-title">Test Model: {{ id }}</h2>
      <div class="header-actions">
        <router-link to="/" class="btn btn-secondary">Back to Dashboard</router-link>
      </div>
    </div>
    
    <div v-if="error" class="alert alert-error">
      {{ error }}
    </div>
    
    <div class="card">
      <h3 class="card-title">Model Information</h3>
      <div v-if="model" class="model-info">
        <div class="info-item">
          <span class="info-label">ID:</span>
          <span class="info-value">{{ model.id }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Type:</span>
          <span class="info-value">{{ model.type }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Status:</span>
          <span class="info-value">
            <span v-if="model.ready" class="badge badge-success">Ready</span>
            <span v-else class="badge badge-warning">Loading</span>
          </span>
        </div>
        <div class="info-item">
          <span class="info-label">Created:</span>
          <span class="info-value">{{ formatDate(model.created) }}</span>
        </div>
      </div>
      <div v-else-if="loading" class="loading-container">
        <div class="loading-spinner"></div>
        <p>Loading model information...</p>
      </div>
      <div v-else class="error-container">
        <p>Model information not available. Please ensure the model is loaded.</p>
        <button @click="loadModel" class="btn btn-primary">
          <span v-if="loading" class="loading-spinner"></span>
          <span v-else>Load Model</span>
        </button>
      </div>
    </div>
    
    <div class="card">
      <h3 class="card-title">Test Completion</h3>
      <div class="form-group">
        <label for="prompt" class="form-label">Prompt</label>
        <textarea 
          id="prompt" 
          v-model="prompt"
          class="form-control"
          rows="4"
          placeholder="Enter your prompt here..."
        ></textarea>
      </div>
      
      <div class="form-row">
        <div class="form-group flex-1">
          <label for="max-tokens" class="form-label">Max Tokens</label>
          <input 
            type="number" 
            id="max-tokens" 
            v-model="maxTokens"
            class="form-control"
            min="1" 
            max="2048"
          />
        </div>
        
        <div class="form-group flex-1">
          <label for="temperature" class="form-label">Temperature</label>
          <input 
            type="range" 
            id="temperature" 
            v-model="temperature"
            class="form-control"
            min="0" 
            max="2" 
            step="0.1"
          />
          <div class="range-value">{{ temperature }}</div>
        </div>
        
        <div class="form-group flex-1">
          <label for="top-p" class="form-label">Top P</label>
          <input 
            type="range" 
            id="top-p" 
            v-model="topP"
            class="form-control"
            min="0" 
            max="1" 
            step="0.05"
          />
          <div class="range-value">{{ topP }}</div>
        </div>
      </div>
      
      <div class="form-actions">
        <button @click="runCompletion" class="btn btn-primary" :disabled="loading || !model">
          <span v-if="loading" class="loading-spinner"></span>
          <span v-else>Generate Completion</span>
        </button>
      </div>
      
      <div v-if="result" class="completion-result">
        <h4 class="result-title">Completion Result</h4>
        <pre class="code-block">{{ result }}</pre>
        
        <div class="form-actions">
          <button @click="copyResult" class="btn btn-secondary">
            Copy to Clipboard
          </button>
        </div>
      </div>
    </div>
    
    <div class="card">
      <h3 class="card-title">API Example</h3>
      <div class="api-example">
        <h4 class="example-title">cURL Command</h4>
        <pre class="code-block">curl -X POST http://localhost:5000/api/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "{{ id }}",
  "prompt": "{{ prompt || 'Enter your prompt here' }}",
  "max_tokens": {{ maxTokens }},
  "temperature": {{ temperature }},
  "top_p": {{ topP }}
}'</pre>
        
        <h4 class="example-title">Python</h4>
        <pre class="code-block">import requests

response = requests.post(
    "http://localhost:5000/api/v1/completions",
    json={
        "model": "{{ id }}",
        "prompt": "{{ prompt || 'Enter your prompt here' }}",
        "max_tokens": {{ maxTokens }},
        "temperature": {{ temperature }},
        "top_p": {{ topP }}
    }
)

result = response.json()
print(result["choices"][0]["text"])</pre>
        
        <h4 class="example-title">JavaScript</h4>
        <pre class="code-block">fetch('http://localhost:5000/api/v1/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    model: '{{ id }}',
    prompt: '{{ prompt || 'Enter your prompt here' }}',
    max_tokens: {{ maxTokens }},
    temperature: {{ temperature }},
    top_p: {{ topP }}
  })
})
.then(response => response.json())
.then(data => console.log(data.choices[0].text));</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useModelStore } from '../store/models'

const props = defineProps<{
  id: string
}>()

const modelStore = useModelStore()
const model = ref(null)
const loading = ref(false)
const error = ref('')
const prompt = ref('')
const maxTokens = ref(50)
const temperature = ref(0.7)
const topP = ref(1.0)
const result = ref('')

onMounted(async () => {
  await fetchModelInfo()
})

async function fetchModelInfo() {
  loading.value = true
  error.value = ''
  
  try {
    await modelStore.fetchModels()
    model.value = modelStore.models.find(m => m.id === props.id) || null
    
    if (!model.value) {
      error.value = `Model "${props.id}" not found or not loaded`
    }
  } catch (err: any) {
    error.value = err.message || 'Error fetching model information'
    console.error('Error fetching model:', err)
  } finally {
    loading.value = false
  }
}

async function loadModel() {
  loading.value = true
  error.value = ''
  
  try {
    const result = await modelStore.loadModel(props.id)
    if (result.success) {
      await fetchModelInfo()
    } else {
      error.value = result.message || 'Failed to load model'
    }
  } catch (err: any) {
    error.value = err.message || 'Error loading model'
    console.error('Error loading model:', err)
  } finally {
    loading.value = false
  }
}

async function runCompletion() {
  if (!prompt.value.trim()) {
    alert('Please enter a prompt')
    return
  }
  
  loading.value = true
  error.value = ''
  result.value = ''
  
  try {
    const response = await axios.post('/api/v1/completions', {
      model: props.id,
      prompt: prompt.value,
      max_tokens: parseInt(maxTokens.value.toString()),
      temperature: parseFloat(temperature.value.toString()),
      top_p: parseFloat(topP.value.toString())
    })
    
    if (response.data && response.data.choices && response.data.choices.length > 0) {
      result.value = response.data.choices[0].text
    } else {
      error.value = 'No completion generated'
    }
  } catch (err: any) {
    error.value = err.response?.data?.error || err.message || 'Error generating completion'
    console.error('Error generating completion:', err)
  } finally {
    loading.value = false
  }
}

function copyResult() {
  if (result.value) {
    navigator.clipboard.writeText(result.value)
      .then(() => alert('Result copied to clipboard'))
      .catch(err => console.error('Failed to copy result:', err))
  }
}

function formatDate(timestamp: number) {
  return new Date(timestamp * 1000).toLocaleString()
}
</script>

<style scoped>
.test-header {
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

.model-info {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-md);
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
}

.info-label {
  font-weight: 500;
  color: var(--color-text-light);
  font-size: 0.875rem;
}

.info-value {
  font-size: 1rem;
}

.form-row {
  display: flex;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-md);
}

.flex-1 {
  flex: 1;
}

.range-value {
  text-align: center;
  margin-top: 4px;
  font-size: 0.875rem;
  color: var(--color-text-light);
}

.form-actions {
  margin-top: var(--spacing-md);
}

.completion-result {
  margin-top: var(--spacing-lg);
  padding-top: var(--spacing-lg);
  border-top: 1px solid var(--color-border);
}

.result-title {
  margin-top: 0;
  margin-bottom: var(--spacing-sm);
  font-size: 1.125rem;
  font-weight: 600;
}

.loading-container, .error-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-lg) 0;
}

.api-example {
  margin-top: var(--spacing-md);
}

.example-title {
  margin-top: var(--spacing-lg);
  margin-bottom: var(--spacing-sm);
  font-size: 1rem;
  font-weight: 600;
}

.example-title:first-child {
  margin-top: 0;
}
</style>