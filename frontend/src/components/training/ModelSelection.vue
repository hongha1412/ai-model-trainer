<template>
  <div class="model-selection">
    <h2>Select Base Model</h2>
    
    <div class="model-source-tabs">
      <button 
        :class="['tab-btn', { active: activeSource === 'huggingface' }]" 
        @click="activeSource = 'huggingface'"
      >
        Hugging Face Models
      </button>
      <button 
        :class="['tab-btn', { active: activeSource === 'local' }]" 
        @click="activeSource = 'local'"
      >
        Local Models
      </button>
    </div>
    
    <div class="tab-content">
      <!-- Hugging Face Models Tab -->
      <div v-if="activeSource === 'huggingface'" class="huggingface-models">
        <div class="search-filters card">
          <div class="form-group">
            <label class="form-label">Search Models</label>
            <div class="search-input-wrapper">
              <input 
                type="text" 
                v-model="searchQuery" 
                class="form-control" 
                placeholder="Search by name, task, or description..."
                @keyup.enter="searchModels"
              >
              <button class="search-btn" @click="searchModels">
                <span v-if="searching" class="loading-spinner"></span>
                <span v-else>Search</span>
              </button>
            </div>
          </div>
          
          <div class="form-group">
            <label class="form-label">Filter by Task</label>
            <select v-model="selectedTask" class="form-control" @change="searchModels">
              <option value="">All Tasks</option>
              <option 
                v-for="task in huggingfaceTasks" 
                :key="task.id" 
                :value="task.id"
              >
                {{ task.name }}
              </option>
            </select>
          </div>
        </div>
        
        <div v-if="loading && !searching" class="loading-container">
          <div class="loading-spinner"></div>
          <p>Loading available tasks...</p>
        </div>
        
        <div v-else-if="error" class="alert alert-error">
          {{ error }}
        </div>
        
        <div v-else-if="huggingfaceModels.length === 0 && searching" class="empty-state">
          <p>No models found matching your search criteria.</p>
        </div>
        
        <div v-else-if="huggingfaceModels.length === 0" class="empty-state">
          <p>Search for models to begin. You can search by name, task, or description.</p>
        </div>
        
        <div v-else class="models-grid">
          <div 
            v-for="model in huggingfaceModels" 
            :key="model.id"
            :class="['model-card', { 'selected': selectedModel?.id === model.id && selectedModel?.source === 'huggingface' }]"
            @click="selectHuggingFaceModel(model)"
          >
            <div class="model-card-header">
              <span class="model-type">{{ model.type || 'Unknown' }}</span>
              <span v-if="model.task" class="model-task">{{ model.task }}</span>
            </div>
            
            <h3 class="model-name">{{ model.name }}</h3>
            
            <p v-if="model.description" class="model-description">
              {{ truncateDescription(model.description) }}
            </p>
            
            <div class="model-actions">
              <button class="btn btn-primary btn-sm" @click.stop="downloadModel(model)">
                <span v-if="downloadingModel === model.id" class="loading-spinner"></span>
                <span v-else>Download</span>
              </button>
              <button class="btn btn-secondary btn-sm" @click.stop="viewModelDetails(model)">
                Details
              </button>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Local Models Tab -->
      <div v-if="activeSource === 'local'" class="local-models">
        <div class="directory-browser card">
          <div class="current-path">
            <span class="path-label">Current Directory:</span>
            <div class="breadcrumb">
              <button 
                class="breadcrumb-item"
                @click="browseDirectory('/models')"
              >
                /models
              </button>
              <template v-for="(part, index) in pathParts" :key="index">
                <span class="separator">/</span>
                <button 
                  class="breadcrumb-item"
                  @click="browseDirectory(buildPath(index))"
                >
                  {{ part }}
                </button>
              </template>
            </div>
          </div>
          
          <div v-if="browsing" class="loading-container">
            <div class="loading-spinner"></div>
            <p>Loading directory contents...</p>
          </div>
          
          <div v-else-if="browseError" class="alert alert-error">
            {{ browseError }}
          </div>
          
          <div v-else class="directory-contents">
            <div class="directories">
              <h4 v-if="browsedDirectories.directories.length > 0">Directories</h4>
              <div 
                v-for="dir in browsedDirectories.directories" 
                :key="dir"
                class="directory-item"
                @click="browseDirectory(`${browsedDirectories.path}/${dir}`)"
              >
                <span class="directory-icon">📁</span>
                <span class="directory-name">{{ dir }}</span>
              </div>
              
              <div v-if="browsedDirectories.directories.length === 0" class="empty-directories">
                <p>No subdirectories found.</p>
              </div>
            </div>
            
            <div class="files">
              <h4 v-if="browsedDirectories.files.length > 0">Model Files</h4>
              <div 
                v-for="file in modelFiles" 
                :key="file"
                class="file-item"
                @click="selectLocalModelFile(file)"
              >
                <span class="file-icon">📄</span>
                <span class="file-name">{{ file }}</span>
              </div>
              
              <div v-if="browsedDirectories.files.length === 0" class="empty-files">
                <p>No model files found in this directory.</p>
              </div>
            </div>
          </div>
          
          <div class="model-selection-info">
            <h4>Selected Model Directory</h4>
            <div v-if="selectedModel && selectedModel.source === 'local'" class="selected-path">
              <span class="selected-path-label">Path:</span>
              <span class="selected-path-value">{{ selectedModel.path }}</span>
            </div>
            <div v-else class="selected-path-placeholder">
              <p>No model directory selected. Browse and select a directory containing model files.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Model Details Modal -->
    <div v-if="showModelModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Model Details: {{ modelDetails?.name }}</h3>
          <button class="close-modal" @click="closeModelModal">×</button>
        </div>
        
        <div class="modal-body">
          <div v-if="loadingModelDetails" class="loading-container">
            <div class="loading-spinner"></div>
            <p>Loading model details...</p>
          </div>
          
          <div v-else-if="modelDetailsError" class="alert alert-error">
            {{ modelDetailsError }}
          </div>
          
          <div v-else-if="modelDetails">
            <div class="model-metadata">
              <div class="metadata-item">
                <strong>ID:</strong> {{ modelDetails.id }}
              </div>
              <div class="metadata-item" v-if="modelDetails.type">
                <strong>Type:</strong> {{ modelDetails.type }}
              </div>
              <div class="metadata-item" v-if="modelDetails.task">
                <strong>Task:</strong> {{ modelDetails.task }}
              </div>
              <div class="metadata-item" v-if="modelDetails.size">
                <strong>Size:</strong> {{ formatSize(modelDetails.size) }}
              </div>
            </div>
            
            <h4>Description</h4>
            <p class="model-full-description">{{ modelDetails.description || 'No description available.' }}</p>
          </div>
        </div>
        
        <div class="modal-footer">
          <button 
            class="btn btn-primary" 
            @click="downloadModel(modelDetails); closeModelModal()"
          >
            Download Model
          </button>
          <button class="btn btn-secondary" @click="closeModelModal">
            Close
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted, watch } from 'vue'
import { useTrainingStore } from '../../store/training'
import type { ModelInfo, HuggingFaceTask } from '../../store/training'

export default defineComponent({
  name: 'ModelSelection',
  
  emits: ['model-selected'],
  
  setup(_, { emit }) {
    const trainingStore = useTrainingStore()
    
    // Tab state
    const activeSource = ref('huggingface')
    
    // Hugging Face search state
    const searchQuery = ref('')
    const selectedTask = ref('')
    const searching = ref(false)
    const downloadingModel = ref<string | null>(null)
    
    // Model selection state
    const selectedModel = ref<ModelInfo | null>(null)
    
    // Modal state
    const showModelModal = ref(false)
    const modelDetails = ref<ModelInfo | null>(null)
    const loadingModelDetails = ref(false)
    const modelDetailsError = ref<string | null>(null)
    
    // Directory browsing state
    const browsing = ref(false)
    const browseError = ref<string | null>(null)
    
    // Computed properties
    const loading = computed(() => trainingStore.loading)
    const error = computed(() => trainingStore.error)
    const huggingfaceModels = computed(() => trainingStore.huggingfaceModels)
    const huggingfaceTasks = computed(() => trainingStore.huggingfaceTasks)
    const browsedDirectories = computed(() => trainingStore.browsedDirectories)
    
    const pathParts = computed(() => {
      const path = browsedDirectories.value.path;
      if (!path || path === '/models') return [];
      
      return path.replace('/models/', '').split('/').filter(Boolean);
    });
    
    const modelFiles = computed(() => {
      return browsedDirectories.value.files.filter(file => 
        file.endsWith('.bin') || 
        file.endsWith('.pt') || 
        file.endsWith('.pth') || 
        file.endsWith('.onnx') || 
        file.endsWith('.safetensors') ||
        file.endsWith('.h5') ||
        file.endsWith('.ckpt') ||
        file.endsWith('.model')
      );
    });
    
    // Methods
    const searchModels = async () => {
      if (!searchQuery.value && !selectedTask.value) return;
      
      searching.value = true;
      try {
        await trainingStore.searchHuggingFaceModels(
          searchQuery.value, 
          selectedTask.value || undefined
        );
      } finally {
        searching.value = false;
      }
    };
    
    const truncateDescription = (description: string, maxLength = 100) => {
      if (description.length <= maxLength) return description;
      return description.substring(0, maxLength) + '...';
    };
    
    const selectHuggingFaceModel = (model: ModelInfo) => {
      const modelInfo: ModelInfo = {
        ...model,
        source: 'huggingface'
      };
      
      selectedModel.value = modelInfo;
      emit('model-selected', modelInfo);
    };
    
    const selectLocalModelFile = (fileName: string) => {
      // Select the parent directory as the model path
      const modelPath = browsedDirectories.value.path;
      
      const modelInfo: ModelInfo = {
        id: fileName,
        name: fileName,
        type: 'local',
        source: 'local',
        path: modelPath
      };
      
      selectedModel.value = modelInfo;
      emit('model-selected', modelInfo);
    };
    
    const viewModelDetails = async (model: ModelInfo) => {
      loadingModelDetails.value = true;
      modelDetailsError.value = null;
      showModelModal.value = true;
      
      try {
        const result = await trainingStore.getHuggingFaceModelInfo(model.id);
        if (result.success && result.data) {
          modelDetails.value = {
            ...result.data,
            source: 'huggingface'
          };
        } else {
          modelDetailsError.value = result.message || 'Failed to load model details';
        }
      } catch (error: any) {
        modelDetailsError.value = error.message || 'An error occurred while loading model details';
      } finally {
        loadingModelDetails.value = false;
      }
    };
    
    const closeModelModal = () => {
      showModelModal.value = false;
      modelDetails.value = null;
      modelDetailsError.value = null;
    };
    
    const downloadModel = async (model: ModelInfo) => {
      if (!model?.id) return;
      
      downloadingModel.value = model.id;
      
      try {
        // Use a default model class for now
        const modelClass = 'sequence-classification';
        const result = await trainingStore.downloadHuggingFaceModel(model.id, modelClass);
        
        if (result.success && result.path) {
          // After downloading, update the model with the local path
          const updatedModel: ModelInfo = {
            ...model,
            source: 'huggingface', // Keep the source as HuggingFace
            path: result.path // But add the local path
          };
          
          selectedModel.value = updatedModel;
          emit('model-selected', updatedModel);
        }
      } finally {
        downloadingModel.value = null;
      }
    };
    
    const browseDirectory = async (path: string) => {
      browsing.value = true;
      browseError.value = null;
      
      try {
        await trainingStore.browseDirectories(path);
      } catch (error: any) {
        browseError.value = error.message || 'Failed to browse directory';
      } finally {
        browsing.value = false;
      }
    };
    
    const buildPath = (upToIndex: number) => {
      const basePath = '/models';
      if (upToIndex < 0) return basePath;
      
      const parts = pathParts.value.slice(0, upToIndex + 1);
      return `${basePath}/${parts.join('/')}`;
    };
    
    const formatSize = (bytes?: number) => {
      if (!bytes) return 'Unknown';
      if (bytes === 0) return '0 Bytes';
      
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    // Watch for source changes to initialize data
    watch(activeSource, (newSource) => {
      if (newSource === 'huggingface') {
        trainingStore.getHuggingFaceTasks();
      } else if (newSource === 'local') {
        browseDirectory('/models');
      }
    });
    
    // Initialize data on component mount
    onMounted(() => {
      // Load Hugging Face tasks
      trainingStore.getHuggingFaceTasks();
    });
    
    return {
      activeSource,
      searchQuery,
      selectedTask,
      searching,
      downloadingModel,
      selectedModel,
      showModelModal,
      modelDetails,
      loadingModelDetails,
      modelDetailsError,
      browsing,
      browseError,
      
      loading,
      error,
      huggingfaceModels,
      huggingfaceTasks,
      browsedDirectories,
      pathParts,
      modelFiles,
      
      searchModels,
      truncateDescription,
      selectHuggingFaceModel,
      selectLocalModelFile,
      viewModelDetails,
      closeModelModal,
      downloadModel,
      browseDirectory,
      buildPath,
      formatSize
    };
  }
});
</script>

<style scoped>
.model-selection h2 {
  margin-top: 0;
  margin-bottom: var(--spacing-lg);
}

.model-source-tabs {
  display: flex;
  margin-bottom: var(--spacing-lg);
  border-bottom: 1px solid var(--color-border);
}

.tab-btn {
  background: none;
  border: none;
  padding: var(--spacing-sm) var(--spacing-md);
  font-size: 1rem;
  font-weight: 500;
  color: var(--color-text-light);
  cursor: pointer;
  position: relative;
  transition: all 0.2s ease;
}

.tab-btn.active {
  color: var(--color-primary);
}

.tab-btn.active::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  right: 0;
  height: 2px;
  background-color: var(--color-primary);
}

.tab-content {
  min-height: 400px;
}

/* Search and Filters */
.search-filters {
  margin-bottom: var(--spacing-lg);
}

.search-input-wrapper {
  display: flex;
  gap: var(--spacing-xs);
}

.search-btn {
  padding: var(--spacing-sm) var(--spacing-md);
  background-color: var(--color-primary);
  color: white;
  border: none;
  border-radius: var(--border-radius);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 80px;
}

.search-btn:hover {
  background-color: var(--color-primary-dark);
}

/* Models Grid */
.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: var(--spacing-md);
}

.model-card {
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  cursor: pointer;
  transition: all 0.2s ease;
}

.model-card:hover {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
  transform: translateY(-2px);
}

.model-card.selected {
  border-color: var(--color-primary);
  background-color: rgba(16, 163, 127, 0.05);
}

.model-card-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--spacing-xs);
}

.model-type {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-primary);
  background-color: rgba(16, 163, 127, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
}

.model-task {
  font-size: 0.75rem;
  color: var(--color-text-light);
  background-color: #f5f5f5;
  padding: 2px 6px;
  border-radius: 4px;
}

.model-name {
  margin: var(--spacing-xs) 0;
  font-size: 1rem;
  word-break: break-word;
}

.model-description {
  margin-bottom: var(--spacing-sm);
  font-size: 0.875rem;
  color: var(--color-text-light);
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.model-actions {
  display: flex;
  justify-content: space-between;
  margin-top: var(--spacing-sm);
}

.btn-sm {
  font-size: 0.75rem;
  padding: 4px 8px;
}

/* Directory Browser */
.directory-browser {
  margin-bottom: var(--spacing-lg);
}

.current-path {
  margin-bottom: var(--spacing-md);
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: var(--spacing-xs);
}

.path-label {
  font-weight: 500;
  color: var(--color-text-light);
}

.breadcrumb {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.breadcrumb-item {
  background: none;
  border: none;
  color: var(--color-primary);
  cursor: pointer;
  padding: 2px 4px;
  font-size: 0.875rem;
}

.breadcrumb-item:hover {
  text-decoration: underline;
}

.separator {
  color: var(--color-text-light);
  margin: 0 2px;
}

.directory-contents {
  margin-bottom: var(--spacing-md);
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-md);
}

.directories, .files {
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  min-height: 200px;
}

.directories h4, .files h4 {
  margin-top: 0;
  margin-bottom: var(--spacing-sm);
  color: var(--color-text-light);
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.directory-item, .file-item {
  display: flex;
  align-items: center;
  padding: var(--spacing-xs) var(--spacing-sm);
  margin-bottom: var(--spacing-xs);
  border-radius: var(--border-radius);
  cursor: pointer;
  transition: all 0.2s ease;
}

.directory-item:hover, .file-item:hover {
  background-color: rgba(0, 0, 0, 0.05);
}

.directory-icon, .file-icon {
  margin-right: var(--spacing-sm);
  font-size: 1.25rem;
}

.directory-name, .file-name {
  font-size: 0.875rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.empty-directories, .empty-files {
  color: var(--color-text-light);
  font-size: 0.875rem;
  text-align: center;
  padding: var(--spacing-lg);
}

.model-selection-info {
  margin-top: var(--spacing-md);
}

.model-selection-info h4 {
  margin-top: 0;
  margin-bottom: var(--spacing-sm);
}

.selected-path {
  background-color: #f5f5f5;
  border-radius: var(--border-radius);
  padding: var(--spacing-sm);
  display: flex;
  align-items: center;
}

.selected-path-label {
  font-weight: 500;
  margin-right: var(--spacing-sm);
}

.selected-path-value {
  font-family: 'Roboto Mono', monospace;
  font-size: 0.875rem;
  word-break: break-all;
}

.selected-path-placeholder {
  color: var(--color-text-light);
  font-size: 0.875rem;
  padding: var(--spacing-sm);
  border: 1px dashed var(--color-border);
  border-radius: var(--border-radius);
  text-align: center;
}

/* Modal */
.modal {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background-color: white;
  border-radius: var(--border-radius);
  max-width: 800px;
  width: 90%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
}

.modal-header {
  padding: var(--spacing-md);
  border-bottom: 1px solid var(--color-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.modal-header h3 {
  margin: 0;
}

.close-modal {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: var(--color-text-light);
}

.modal-body {
  padding: var(--spacing-md);
  overflow-y: auto;
  flex-grow: 1;
}

.modal-footer {
  padding: var(--spacing-md);
  border-top: 1px solid var(--color-border);
  display: flex;
  justify-content: flex-end;
  gap: var(--spacing-sm);
}

.model-metadata {
  margin-bottom: var(--spacing-md);
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-sm);
}

.metadata-item {
  padding: var(--spacing-xs);
}

.model-full-description {
  white-space: pre-line;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-lg);
}

.empty-state {
  text-align: center;
  padding: var(--spacing-xl);
  color: var(--color-text-light);
}

@media (max-width: 768px) {
  .models-grid {
    grid-template-columns: 1fr;
  }
  
  .directory-contents {
    grid-template-columns: 1fr;
  }
  
  .model-metadata {
    grid-template-columns: 1fr;
  }
}
</style>