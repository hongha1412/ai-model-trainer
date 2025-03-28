<template>
  <div class="openapi">
    <div class="openapi-header">
      <h2 class="section-title">API Documentation</h2>
    </div>
    
    <div v-if="loading" class="loading-container">
      <div class="loading-spinner"></div>
      <p>Loading API documentation...</p>
    </div>
    
    <div v-else-if="error" class="alert alert-error">
      {{ error }}
    </div>
    
    <div v-else-if="spec" class="openapi-content">
      <div class="card">
        <h3 class="card-title">{{ spec.info.title }} <span class="api-version">v{{ spec.info.version }}</span></h3>
        <p class="api-description">{{ spec.info.description }}</p>
        
        <div class="server-info">
          <h4 class="section-subtitle">Server</h4>
          <div v-for="(server, index) in spec.servers" :key="index" class="server-url">
            {{ server.url }} <span v-if="server.description" class="server-description">- {{ server.description }}</span>
          </div>
        </div>
      </div>
      
      <div v-for="(pathObj, path) in spec.paths" :key="path" class="card">
        <h3 class="card-title endpoint-path">{{ path }}</h3>
        
        <div v-for="(operation, method) in pathObj" :key="`${path}-${method}`" class="endpoint">
          <div class="endpoint-header">
            <div class="http-method" :class="`method-${method}`">{{ method.toUpperCase() }}</div>
            <h4 class="endpoint-title">{{ operation.summary }}</h4>
          </div>
          
          <div class="endpoint-description" v-if="operation.description">
            {{ operation.description }}
          </div>
          
          <div class="parameters" v-if="operation.parameters && operation.parameters.length > 0">
            <h5 class="parameters-title">Parameters</h5>
            <table class="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>In</th>
                  <th>Type</th>
                  <th>Required</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="param in operation.parameters" :key="param.name">
                  <td><code>{{ param.name }}</code></td>
                  <td>{{ param.in }}</td>
                  <td>{{ getParameterType(param) }}</td>
                  <td>{{ param.required ? 'Yes' : 'No' }}</td>
                  <td>{{ param.description }}</td>
                </tr>
              </tbody>
            </table>
          </div>
          
          <div class="request-body" v-if="operation.requestBody">
            <h5 class="request-body-title">Request Body</h5>
            <div v-if="operation.requestBody.description" class="request-body-description">
              {{ operation.requestBody.description }}
            </div>
            
            <div v-if="operation.requestBody.content" class="content-types">
              <div v-for="(content, contentType) in operation.requestBody.content" :key="contentType" class="content-type">
                <div class="content-type-header">
                  <h6 class="content-type-title">{{ contentType }}</h6>
                  <div class="required-badge" v-if="operation.requestBody.required">Required</div>
                </div>
                
                <div v-if="content.schema" class="schema">
                  <pre class="code-block">{{ formatSchema(content.schema) }}</pre>
                </div>
              </div>
            </div>
          </div>
          
          <div class="responses">
            <h5 class="responses-title">Responses</h5>
            <div v-for="(response, status) in operation.responses" :key="status" class="response">
              <div class="response-header">
                <div class="status-code">{{ status }}</div>
                <div class="response-description">{{ response.description }}</div>
              </div>
              
              <div v-if="response.content" class="content-types">
                <div v-for="(content, contentType) in response.content" :key="contentType" class="content-type">
                  <h6 class="content-type-title">{{ contentType }}</h6>
                  
                  <div v-if="content.schema" class="schema">
                    <pre class="code-block">{{ formatSchema(content.schema) }}</pre>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div class="example" v-if="getExampleForEndpoint(path, method)">
            <h5 class="example-title">Example</h5>
            <pre class="code-block">{{ getExampleForEndpoint(path, method) }}</pre>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

const spec = ref(null)
const loading = ref(false)
const error = ref('')

onMounted(async () => {
  await fetchOpenApiSpec()
})

async function fetchOpenApiSpec() {
  loading.value = true
  error.value = ''
  
  try {
    const response = await axios.get('/api/openapi/openapi.json')
    spec.value = response.data
  } catch (err: any) {
    error.value = err.message || 'Error fetching OpenAPI specification'
    console.error('Error fetching OpenAPI spec:', err)
  } finally {
    loading.value = false
  }
}

function getParameterType(param: any): string {
  if (!param.schema) return 'unknown'
  
  if (param.schema.type === 'array' && param.schema.items) {
    return `array of ${param.schema.items.type || 'unknown'}`
  }
  
  return param.schema.type || 'unknown'
}

function formatSchema(schema: any): string {
  return JSON.stringify(schema, null, 2)
}

function getExampleForEndpoint(path: string, method: string): string | null {
  const examples: Record<string, Record<string, string>> = {
    '/v1/models': {
      get: `curl -X GET http://localhost:5000/v1/models`
    },
    '/v1/models/{model}': {
      get: `curl -X GET http://localhost:5000/v1/models/t5-small`
    },
    '/v1/completions': {
      post: `curl -X POST http://localhost:5000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "t5-small",
  "prompt": "Translate to French: Hello, how are you?",
  "max_tokens": 50,
  "temperature": 0.7
}'`
    },
    '/v1/chat/completions': {
      post: `curl -X POST http://localhost:5000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
  "model": "t5-small",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "max_tokens": 50,
  "temperature": 0.7
}'`
    }
  }
  
  return examples[path]?.[method] || null
}
</script>

<style scoped>
.openapi-header {
  margin-bottom: var(--spacing-lg);
}

.section-title {
  margin: 0;
  font-size: 1.5rem;
  font-weight: 600;
}

.api-version {
  font-size: 1rem;
  font-weight: normal;
  color: var(--color-text-light);
}

.api-description {
  margin-top: var(--spacing-sm);
  margin-bottom: var(--spacing-lg);
}

.section-subtitle {
  font-size: 1.125rem;
  font-weight: 600;
  margin-top: var(--spacing-lg);
  margin-bottom: var(--spacing-sm);
}

.server-url {
  font-family: 'Roboto Mono', monospace;
  font-size: 0.875rem;
  padding: var(--spacing-sm);
  background-color: rgba(0, 0, 0, 0.03);
  border-radius: var(--border-radius);
  margin-bottom: var(--spacing-sm);
}

.server-description {
  font-family: 'Inter', sans-serif;
  color: var(--color-text-light);
}

.endpoint-path {
  font-family: 'Roboto Mono', monospace;
  word-break: break-word;
}

.endpoint {
  padding: var(--spacing-md) 0;
  border-bottom: 1px solid var(--color-border);
}

.endpoint:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.endpoint-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-md);
}

.http-method {
  font-size: 0.75rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
  min-width: 60px;
  text-align: center;
}

.method-get {
  background-color: rgba(97, 175, 254, 0.1);
  color: #61affe;
}

.method-post {
  background-color: rgba(73, 204, 144, 0.1);
  color: #49cc90;
}

.method-put {
  background-color: rgba(252, 161, 48, 0.1);
  color: #fca130;
}

.method-delete {
  background-color: rgba(249, 62, 62, 0.1);
  color: #f93e3e;
}

.endpoint-title {
  margin: 0;
  font-size: 1.125rem;
  font-weight: 500;
}

.endpoint-description {
  margin-bottom: var(--spacing-md);
}

.parameters-title, .request-body-title, .responses-title, .example-title {
  font-size: 1rem;
  font-weight: 600;
  margin-top: var(--spacing-lg);
  margin-bottom: var(--spacing-md);
}

.content-type-title {
  font-size: 0.875rem;
  font-weight: 500;
  margin-top: var(--spacing-md);
  margin-bottom: var(--spacing-sm);
}

.content-type-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
}

.required-badge {
  font-size: 0.75rem;
  background-color: rgba(249, 62, 62, 0.1);
  color: #f93e3e;
  padding: 2px 6px;
  border-radius: 4px;
}

.response-header {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-sm);
}

.status-code {
  font-weight: 600;
  font-family: 'Roboto Mono', monospace;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--spacing-md);
  padding: var(--spacing-xl) 0;
}

code {
  font-family: 'Roboto Mono', monospace;
  font-size: 0.875rem;
  background-color: rgba(0, 0, 0, 0.03);
  padding: 2px 4px;
  border-radius: 4px;
}
</style>