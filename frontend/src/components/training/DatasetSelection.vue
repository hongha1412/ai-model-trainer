<template>
  <div class="dataset-selection">
    <h2>Select Dataset</h2>
    
    <div class="dataset-tabs">
      <button 
        :class="['tab-btn', { active: activeTab === 'existing' }]" 
        @click="activeTab = 'existing'"
      >
        Existing Datasets
      </button>
      <button 
        :class="['tab-btn', { active: activeTab === 'upload' }]" 
        @click="activeTab = 'upload'"
      >
        Upload New Dataset
      </button>
    </div>
    
    <div class="tab-content">
      <!-- Existing Datasets Tab -->
      <div v-if="activeTab === 'existing'" class="existing-datasets">
        <div v-if="loading" class="loading-container">
          <div class="loading-spinner"></div>
          <p>Loading datasets...</p>
        </div>
        
        <div v-else-if="error" class="alert alert-error">
          {{ error }}
        </div>
        
        <div v-else-if="datasets.length === 0" class="empty-state">
          <p>No datasets available. Please upload a dataset first.</p>
        </div>
        
        <div v-else>
          <div class="dataset-grid">
            <div 
              v-for="dataset in datasets" 
              :key="dataset.name"
              :class="['dataset-card', { 'selected': selectedDataset?.name === dataset.name }]"
              @click="selectDataset(dataset)"
            >
              <div class="dataset-card-header">
                <span class="dataset-format">{{ dataset.format.toUpperCase() }}</span>
                <span class="dataset-size">{{ formatSize(dataset.size) }}</span>
              </div>
              
              <h3 class="dataset-name">{{ dataset.name }}</h3>
              
              <div class="dataset-info">
                <p class="dataset-created">Uploaded: {{ formatDate(dataset.created) }}</p>
                <p v-if="dataset.columns && dataset.columns.length > 0" class="dataset-columns">
                  Columns: {{ dataset.columns.join(', ') }}
                </p>
              </div>
              
              <div class="dataset-actions">
                <button class="btn btn-secondary btn-sm" @click.stop="viewDatasetDetails(dataset)">
                  View Details
                </button>
                <button class="btn btn-danger btn-sm" @click.stop="confirmDeleteDataset(dataset)">
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Upload New Dataset Tab -->
      <div v-if="activeTab === 'upload'" class="upload-dataset">
        <div class="card">
          <h3>Upload Dataset</h3>
          
          <div class="form-group">
            <label class="form-label">Dataset Format</label>
            <div class="format-options">
              <div 
                v-for="format in supportedFormats" 
                :key="format"
                :class="['format-option', { 'active': selectedFormat === format }]"
                @click="selectedFormat = format"
              >
                <span class="format-label">{{ format.toUpperCase() }}</span>
                <span class="format-description">{{ getFormatDescription(format) }}</span>
              </div>
            </div>
          </div>
          
          <div class="form-group">
            <label class="form-label">Dataset File</label>
            <div 
              class="file-drop-area"
              :class="{ 'drag-over': isDragging }"
              @dragover.prevent="isDragging = true"
              @dragleave.prevent="isDragging = false"
              @drop.prevent="onFileDrop"
              @click="triggerFileInput"
            >
              <input 
                type="file" 
                ref="fileInput" 
                style="display: none" 
                @change="onFileChange"
              >
              
              <div v-if="!selectedFile" class="drop-message">
                <i class="upload-icon">📤</i>
                <p>Drag your file here or click to browse</p>
                <p class="supported-formats">
                  Supported formats: {{ supportedFormats.map(f => f.toUpperCase()).join(', ') }}
                </p>
              </div>
              
              <div v-else class="file-selected">
                <i class="file-icon">📄</i>
                <p class="file-name">{{ selectedFile.name }}</p>
                <p class="file-size">{{ formatSize(selectedFile.size) }}</p>
                <button class="remove-file" @click.stop="removeSelectedFile">×</button>
              </div>
            </div>
          </div>
          
          <div class="form-actions">
            <button 
              class="btn btn-primary" 
              :disabled="!canUpload || uploading" 
              @click="uploadFile"
            >
              <div v-if="uploading" class="loading-spinner"></div>
              <span v-else>Upload Dataset</span>
            </button>
          </div>
        </div>
        
        <div v-if="uploadResult" :class="['alert', uploadResult.success ? 'alert-success' : 'alert-error']">
          {{ uploadResult.message }}
        </div>
      </div>
    </div>
    
    <!-- Dataset Preview Modal -->
    <div v-if="showPreviewModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Dataset Preview: {{ datasetPreview?.name }}</h3>
          <button class="close-modal" @click="closePreviewModal">×</button>
        </div>
        
        <div class="modal-body">
          <div v-if="loadingPreview" class="loading-container">
            <div class="loading-spinner"></div>
            <p>Loading dataset details...</p>
          </div>
          
          <div v-else-if="previewError" class="alert alert-error">
            {{ previewError }}
          </div>
          
          <div v-else-if="datasetPreview">
            <div class="dataset-metadata">
              <div class="metadata-item">
                <strong>Format:</strong> {{ datasetPreview.format.toUpperCase() }}
              </div>
              <div class="metadata-item">
                <strong>Size:</strong> {{ formatSize(datasetPreview.size) }}
              </div>
              <div class="metadata-item">
                <strong>Uploaded:</strong> {{ formatDate(datasetPreview.created) }}
              </div>
              <div v-if="datasetPreview.columns && datasetPreview.columns.length > 0" class="metadata-item">
                <strong>Columns:</strong> {{ datasetPreview.columns.join(', ') }}
              </div>
            </div>
            
            <h4>Content Preview</h4>
            <pre class="preview-content">{{ datasetPreview.preview }}</pre>
          </div>
        </div>
        
        <div class="modal-footer">
          <button 
            class="btn btn-primary" 
            @click="selectDataset(datasetPreview); closePreviewModal()"
          >
            Select This Dataset
          </button>
          <button class="btn btn-secondary" @click="closePreviewModal">
            Close
          </button>
        </div>
      </div>
    </div>
    
    <!-- Delete Confirmation Modal -->
    <div v-if="showDeleteModal" class="modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Confirm Deletion</h3>
          <button class="close-modal" @click="showDeleteModal = false">×</button>
        </div>
        
        <div class="modal-body">
          <p>Are you sure you want to delete the dataset <strong>{{ datasetToDelete?.name }}</strong>?</p>
          <p class="delete-warning">This action cannot be undone.</p>
        </div>
        
        <div class="modal-footer">
          <button 
            class="btn btn-danger" 
            :disabled="deleting"
            @click="deleteDataset"
          >
            <div v-if="deleting" class="loading-spinner"></div>
            <span v-else>Delete</span>
          </button>
          <button 
            class="btn btn-secondary" 
            :disabled="deleting"
            @click="showDeleteModal = false"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, onMounted } from 'vue'
import { useTrainingStore } from '../../store/training'
import type { DatasetInfo } from '../../store/training'

export default defineComponent({
  name: 'DatasetSelection',
  
  emits: ['dataset-selected'],
  
  setup(_, { emit }) {
    const trainingStore = useTrainingStore()
    const activeTab = ref('existing')
    const selectedDataset = ref<DatasetInfo | null>(null)
    const fileInput = ref<HTMLInputElement | null>(null)
    
    // Upload state
    const selectedFile = ref<File | null>(null)
    const selectedFormat = ref('auto')
    const isDragging = ref(false)
    const uploading = ref(false)
    const uploadResult = ref<{ success: boolean, message: string } | null>(null)
    
    // Preview modal state
    const showPreviewModal = ref(false)
    const datasetPreview = ref<DatasetInfo | null>(null)
    const loadingPreview = ref(false)
    const previewError = ref<string | null>(null)
    
    // Delete modal state
    const showDeleteModal = ref(false)
    const datasetToDelete = ref<DatasetInfo | null>(null)
    const deleting = ref(false)
    
    const supportedFormats = ref(['auto', 'text', 'json', 'csv'])
    
    // Computed properties
    const loading = computed(() => trainingStore.loading)
    const error = computed(() => trainingStore.error)
    const datasets = computed(() => trainingStore.datasets)
    
    const canUpload = computed(() => {
      return selectedFile.value !== null && selectedFormat.value !== ''
    })
    
    // Methods
    const fetchDatasets = async () => {
      await trainingStore.fetchDatasets()
    }
    
    const selectDataset = (dataset: DatasetInfo | null) => {
      selectedDataset.value = dataset
      if (dataset) {
        emit('dataset-selected', dataset)
      }
    }
    
    const viewDatasetDetails = async (dataset: DatasetInfo) => {
      loadingPreview.value = true
      previewError.value = null
      showPreviewModal.value = true
      
      try {
        const result = await trainingStore.getDatasetInfo(dataset.name)
        if (result.success && result.data) {
          datasetPreview.value = result.data
        } else {
          previewError.value = result.message || 'Failed to load dataset details'
        }
      } catch (error: any) {
        previewError.value = error.message || 'An error occurred while loading dataset details'
      } finally {
        loadingPreview.value = false
      }
    }
    
    const closePreviewModal = () => {
      showPreviewModal.value = false
      datasetPreview.value = null
      previewError.value = null
    }
    
    const confirmDeleteDataset = (dataset: DatasetInfo) => {
      datasetToDelete.value = dataset
      showDeleteModal.value = true
    }
    
    const deleteDataset = async () => {
      if (!datasetToDelete.value) return
      
      deleting.value = true
      
      try {
        const result = await trainingStore.deleteDataset(datasetToDelete.value.name)
        if (result.success) {
          // If the deleted dataset was selected, clear the selection
          if (selectedDataset.value?.name === datasetToDelete.value.name) {
            selectDataset(null)
          }
        }
      } finally {
        deleting.value = false
        showDeleteModal.value = false
        datasetToDelete.value = null
      }
    }
    
    const triggerFileInput = () => {
      if (fileInput.value) {
        fileInput.value.click()
      }
    }
    
    const onFileChange = (event: Event) => {
      const target = event.target as HTMLInputElement
      if (target.files && target.files.length > 0) {
        selectedFile.value = target.files[0]
        
        // Try to auto-detect format from file extension
        const fileName = selectedFile.value.name.toLowerCase()
        if (fileName.endsWith('.json')) {
          selectedFormat.value = 'json'
        } else if (fileName.endsWith('.csv')) {
          selectedFormat.value = 'csv'
        } else if (fileName.endsWith('.txt')) {
          selectedFormat.value = 'text'
        } else {
          selectedFormat.value = 'auto'
        }
      }
    }
    
    const onFileDrop = (event: DragEvent) => {
      isDragging.value = false
      if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
        selectedFile.value = event.dataTransfer.files[0]
        
        // Try to auto-detect format from file extension
        const fileName = selectedFile.value.name.toLowerCase()
        if (fileName.endsWith('.json')) {
          selectedFormat.value = 'json'
        } else if (fileName.endsWith('.csv')) {
          selectedFormat.value = 'csv'
        } else if (fileName.endsWith('.txt')) {
          selectedFormat.value = 'text'
        } else {
          selectedFormat.value = 'auto'
        }
      }
    }
    
    const removeSelectedFile = () => {
      selectedFile.value = null
      if (fileInput.value) {
        fileInput.value.value = ''
      }
    }
    
    const uploadFile = async () => {
      if (!selectedFile.value) return
      
      uploading.value = true
      uploadResult.value = null
      
      try {
        const result = await trainingStore.uploadDataset(selectedFile.value)
        uploadResult.value = result
        
        if (result.success) {
          // Clear selected file after successful upload
          removeSelectedFile()
          // Switch to existing datasets tab
          activeTab.value = 'existing'
        }
      } finally {
        uploading.value = false
      }
    }
    
    const getFormatDescription = (format: string) => {
      switch (format) {
        case 'auto':
          return 'Auto-detect format from file extension'
        case 'text':
          return 'Plain text data, one example per line'
        case 'json':
          return 'JSON structured data with defined fields'
        case 'csv':
          return 'CSV data with headers and columns'
        default:
          return ''
      }
    }
    
    const formatSize = (bytes: number) => {
      if (bytes === 0) return '0 Bytes'
      const k = 1024
      const sizes = ['Bytes', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    }
    
    const formatDate = (dateString: string) => {
      const date = new Date(dateString)
      return date.toLocaleDateString()
    }
    
    // Initialize data
    onMounted(() => {
      fetchDatasets()
    })
    
    return {
      activeTab,
      loading,
      error,
      datasets,
      selectedDataset,
      fileInput,
      selectedFile,
      selectedFormat,
      isDragging,
      uploading,
      uploadResult,
      supportedFormats,
      showPreviewModal,
      datasetPreview,
      loadingPreview,
      previewError,
      showDeleteModal,
      datasetToDelete,
      deleting,
      canUpload,
      
      selectDataset,
      viewDatasetDetails,
      closePreviewModal,
      confirmDeleteDataset,
      deleteDataset,
      triggerFileInput,
      onFileChange,
      onFileDrop,
      removeSelectedFile,
      uploadFile,
      getFormatDescription,
      formatSize,
      formatDate
    }
  }
})
</script>

<style scoped>
.dataset-selection h2 {
  margin-top: 0;
  margin-bottom: var(--spacing-lg);
}

.dataset-tabs {
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

/* Existing Datasets */
.dataset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: var(--spacing-md);
}

.dataset-card {
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  cursor: pointer;
  transition: all 0.2s ease;
}

.dataset-card:hover {
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
  transform: translateY(-2px);
}

.dataset-card.selected {
  border-color: var(--color-primary);
  background-color: rgba(16, 163, 127, 0.05);
}

.dataset-card-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: var(--spacing-xs);
}

.dataset-format {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-primary);
  background-color: rgba(16, 163, 127, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
}

.dataset-size {
  font-size: 0.75rem;
  color: var(--color-text-light);
}

.dataset-name {
  margin: var(--spacing-xs) 0;
  font-size: 1rem;
  word-break: break-word;
}

.dataset-info {
  margin-bottom: var(--spacing-sm);
  font-size: 0.875rem;
  color: var(--color-text-light);
}

.dataset-created, .dataset-columns {
  margin: var(--spacing-xs) 0;
}

.dataset-columns {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.dataset-actions {
  display: flex;
  justify-content: space-between;
  margin-top: var(--spacing-sm);
}

.btn-sm {
  font-size: 0.75rem;
  padding: 4px 8px;
}

/* Upload New Dataset */
.format-options {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md);
}

.format-option {
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  padding: var(--spacing-sm);
  cursor: pointer;
  transition: all 0.2s ease;
}

.format-option:hover {
  border-color: var(--color-primary);
}

.format-option.active {
  border-color: var(--color-primary);
  background-color: rgba(16, 163, 127, 0.05);
}

.format-label {
  display: block;
  font-weight: 600;
  margin-bottom: 4px;
}

.format-description {
  display: block;
  font-size: 0.75rem;
  color: var(--color-text-light);
}

.file-drop-area {
  border: 2px dashed var(--color-border);
  border-radius: var(--border-radius);
  padding: var(--spacing-xl);
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
}

.file-drop-area:hover, .file-drop-area.drag-over {
  border-color: var(--color-primary);
  background-color: rgba(16, 163, 127, 0.05);
}

.drop-message {
  color: var(--color-text-light);
}

.upload-icon {
  font-size: 2rem;
  margin-bottom: var(--spacing-sm);
  display: block;
}

.supported-formats {
  font-size: 0.75rem;
  margin-top: var(--spacing-sm);
}

.file-selected {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}

.file-icon {
  font-size: 2rem;
  margin-bottom: var(--spacing-sm);
}

.file-name {
  font-weight: 500;
  margin: var(--spacing-xs) 0;
  word-break: break-all;
}

.file-size {
  font-size: 0.875rem;
  color: var(--color-text-light);
}

.remove-file {
  position: absolute;
  top: 0;
  right: 0;
  background: none;
  border: none;
  font-size: 1.5rem;
  color: var(--color-text-light);
  cursor: pointer;
}

.remove-file:hover {
  color: var(--color-error);
}

.form-actions {
  margin-top: var(--spacing-lg);
  display: flex;
  justify-content: flex-end;
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

.dataset-metadata {
  margin-bottom: var(--spacing-md);
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-sm);
}

.metadata-item {
  padding: var(--spacing-xs);
}

.preview-content {
  background-color: #f5f5f5;
  padding: var(--spacing-md);
  border-radius: var(--border-radius);
  overflow-x: auto;
  font-family: 'Roboto Mono', monospace;
  font-size: 0.875rem;
  white-space: pre-wrap;
  max-height: 300px;
  overflow-y: auto;
}

.delete-warning {
  color: var(--color-error);
  font-weight: 500;
}

@media (max-width: 768px) {
  .dataset-grid {
    grid-template-columns: 1fr;
  }
  
  .format-options {
    grid-template-columns: 1fr;
  }
  
  .dataset-metadata {
    grid-template-columns: 1fr;
  }
}
</style>