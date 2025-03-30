<template>
  <div class="training-progress">
    <h2>Training Progress</h2>
    
    <div class="training-summary card">
      <div class="summary-grid">
        <div class="summary-item">
          <div class="summary-label">Dataset</div>
          <div class="summary-value">{{ dataset?.name || 'No dataset selected' }}</div>
        </div>
        
        <div class="summary-item">
          <div class="summary-label">Model</div>
          <div class="summary-value">{{ model?.name || 'No model selected' }}</div>
        </div>
        
        <div class="summary-item">
          <div class="summary-label">Learning Type</div>
          <div class="summary-value">{{ getLearningTypeLabel(config?.learning_type) }}</div>
        </div>
        
        <div class="summary-item">
          <div class="summary-label">Batch Size</div>
          <div class="summary-value">{{ config?.batch_size || 'N/A' }}</div>
        </div>
        
        <div class="summary-item">
          <div class="summary-label">Learning Rate</div>
          <div class="summary-value">{{ config?.learning_rate || 'N/A' }}</div>
        </div>
        
        <div class="summary-item" v-if="config?.num_train_epochs">
          <div class="summary-label">Epochs</div>
          <div class="summary-value">{{ config.num_train_epochs }}</div>
        </div>
        
        <div class="summary-item" v-if="config?.max_steps">
          <div class="summary-label">Max Steps</div>
          <div class="summary-value">{{ config.max_steps }}</div>
        </div>
      </div>
    </div>
    
    <div class="training-status card">
      <h3>Training Status</h3>
      
      <div class="status-indicator">
        <div class="status-label">Status:</div>
        <div :class="['status-value', isTraining ? 'status-running' : (error ? 'status-error' : 'status-ready')]">
          {{ statusText }}
        </div>
      </div>
      
      <div class="progress-bar-container">
        <div class="progress-bar" :style="{ width: `${progress}%` }"></div>
        <div class="progress-text">{{ progress }}%</div>
      </div>
      
      <div class="metrics-grid" v-if="metrics.length > 0">
        <div 
          v-for="metric in metrics" 
          :key="metric.name" 
          class="metric-item"
        >
          <div class="metric-name">{{ metric.name }}</div>
          <div class="metric-value">{{ formatMetricValue(metric.value) }}</div>
        </div>
      </div>
      
      <div v-if="error" class="alert alert-error">
        {{ error }}
      </div>
    </div>
    
    <div class="training-logs card">
      <h3>
        Training Logs
        <button 
          v-if="logs.length > 0" 
          class="btn-icon clear-logs" 
          @click="clearLogs"
          title="Clear logs"
        >
          ×
        </button>
      </h3>
      
      <div class="logs-container" ref="logsContainer">
        <div v-if="logs.length === 0" class="empty-logs">
          No logs available yet. Start training to see logs.
        </div>
        
        <div v-else class="logs-content">
          <div 
            v-for="(log, index) in logs" 
            :key="index" 
            class="log-entry"
          >
            <div class="log-timestamp">{{ formatTimestamp(log.timestamp) }}</div>
            <div class="log-message">{{ log.message }}</div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="actions-container">
      <button 
        class="btn btn-primary" 
        v-if="!isTraining" 
        @click="startTraining"
      >
        Start Training
      </button>
      
      <button 
        class="btn btn-danger" 
        v-if="isTraining" 
        @click="stopTraining"
      >
        Stop Training
      </button>
      
      <button 
        class="btn btn-secondary" 
        v-if="isComplete"
        @click="onTrainingComplete"
      >
        Finish
      </button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { useTrainingStore } from '../../store/training'
import type { DatasetInfo, ModelInfo, TrainingConfig } from '../../store/training'

interface LogEntry {
  timestamp: Date
  message: string
}

interface MetricItem {
  name: string
  value: number
}

export default defineComponent({
  name: 'TrainingProgress',
  
  props: {
    dataset: {
      type: Object as () => DatasetInfo | null,
      default: null
    },
    model: {
      type: Object as () => ModelInfo | null,
      default: null
    },
    config: {
      type: Object as () => TrainingConfig | null,
      default: null
    }
  },
  
  emits: ['training-complete'],
  
  setup(props, { emit }) {
    const trainingStore = useTrainingStore()
    const logsContainer = ref<HTMLElement | null>(null)
    
    // Training state
    const isTraining = computed(() => trainingStore.trainingStatus.isTraining)
    const progress = computed(() => trainingStore.trainingStatus.progress)
    const error = computed(() => trainingStore.trainingStatus.error)
    
    // Local state
    const logs = ref<LogEntry[]>([])
    const metrics = ref<MetricItem[]>([])
    const isComplete = ref(false)
    
    // Computed properties
    const statusText = computed(() => {
      if (error.value) return 'Error'
      
      const status = trainingStore.trainingStatus.status
      if (status) {
        // Map API status to user-friendly labels
        const statusMap: Record<string, string> = {
          'training': 'Training',
          'evaluating': 'Evaluating',
          'completed': 'Complete',
          'error': 'Error',
          'stopping': 'Stopping...',
          'stopped': 'Stopped'
        }
        return statusMap[status] || status
      }
      
      // Fallbacks if status is not provided
      if (isTraining.value) return 'Training'
      if (isComplete.value) return 'Complete'
      return 'Ready'
    })
    
    // Methods
    const startTraining = async () => {
      if (!props.dataset || !props.model || !props.config) return
      
      // Reset state
      logs.value = []
      metrics.value = []
      isComplete.value = false
      
      // Prepare training parameters
      const trainingParams = {
        model_source: props.model.source,
        dataset_filename: props.dataset.filename,
        input_field: props.config.input_field || 'input',
        output_field: props.config.output_field,
        learning_type: props.config.learning_type,
        config: props.config,
        output_dir: props.config.output_dir || 'models/trained'
      }
      
      // Add model information based on source
      if (props.model.source === 'huggingface') {
        trainingParams.model_id = props.model.id
      } else {
        trainingParams.model_path = props.model.path
      }
      
      // Start training
      await trainingStore.trainModel(trainingParams)
      
      // Add initial log
      addLog('Starting training...')
    }
    
    const stopTraining = async () => {
      if (trainingStore.trainingStatus.jobId) {
        await trainingStore.stopTrainingJob(trainingStore.trainingStatus.jobId)
        addLog('Training stop requested')
      }
    }
    
    const onTrainingComplete = () => {
      emit('training-complete')
    }
    
    const addLog = (message: string) => {
      logs.value.push({
        timestamp: new Date(),
        message
      })
      
      // Scroll to bottom of logs
      scrollLogsToBottom()
    }
    
    const clearLogs = () => {
      logs.value = []
    }
    
    const scrollLogsToBottom = async () => {
      await nextTick()
      if (logsContainer.value) {
        logsContainer.value.scrollTop = logsContainer.value.scrollHeight
      }
    }
    
    const formatTimestamp = (date: Date) => {
      return date.toLocaleTimeString()
    }
    
    const formatMetricValue = (value: number) => {
      // Format the metric value to 4 decimal places if it's small
      return value < 0.01 ? value.toExponential(2) : value.toFixed(4)
    }
    
    const getLearningTypeLabel = (type?: string) => {
      if (!type) return 'N/A'
      
      const typesMap: Record<string, string> = {
        'supervised': 'Supervised Learning',
        'unsupervised': 'Unsupervised Learning',
        'reinforcement': 'Reinforcement Learning',
        'semi_supervised': 'Semi-supervised Learning',
        'self_supervised': 'Self-supervised Learning',
        'online': 'Online Learning',
        'federated': 'Federated Learning'
      }
      
      return typesMap[type] || type
    }
    
    // Watch for training status updates
    watch(() => trainingStore.trainingStatus, (status) => {
      // Add new logs
      if (status.log && status.log.length > 0) {
        const lastLogIndex = logs.value.length
        const newLogs = status.log.slice(lastLogIndex)
        
        for (const logMessage of newLogs) {
          addLog(logMessage)
          
          // Check for metrics in log messages
          const lossMatch = logMessage.match(/loss: (\d+\.\d+)/)
          if (lossMatch && lossMatch[1]) {
            updateMetric('Loss', parseFloat(lossMatch[1]))
          }
          
          const accuracyMatch = logMessage.match(/accuracy: (\d+\.\d+)/)
          if (accuracyMatch && accuracyMatch[1]) {
            updateMetric('Accuracy', parseFloat(accuracyMatch[1]))
          }
          
          const f1Match = logMessage.match(/f1: (\d+\.\d+)/)
          if (f1Match && f1Match[1]) {
            updateMetric('F1 Score', parseFloat(f1Match[1]))
          }
        }
      }
      
      // Check if metrics are available from API response
      if (status.metrics) {
        for (const [name, value] of Object.entries(status.metrics)) {
          // Convert metric names to title case
          const formattedName = name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ')
          
          updateMetric(formattedName, value)
        }
      }
      
      // Update training step and epoch information
      if (status.currentEpoch !== undefined && status.currentStep !== undefined) {
        addLog(`Epoch ${status.currentEpoch}, Step ${status.currentStep}`)
      }
      
      // Update status text based on status field
      if (status.status) {
        const statusMap: Record<string, string> = {
          'training': 'Training in progress',
          'evaluating': 'Evaluating model',
          'completed': 'Training completed successfully',
          'error': 'Training failed with errors',
          'stopping': 'Stopping training...',
          'stopped': 'Training stopped by user'
        }
        
        const statusMessage = statusMap[status.status] || `Status: ${status.status}`
        
        // Don't add duplicates
        const lastLog = logs.value[logs.value.length - 1]
        if (!lastLog || lastLog.message !== statusMessage) {
          addLog(statusMessage)
        }
      }
      
      // Check if training is complete based on status field
      if (['completed', 'error', 'stopped'].includes(status.status || '') && !isComplete.value) {
        isComplete.value = true
        if (status.status === 'completed') {
          addLog('Training completed successfully!')
        }
      }
      
      // Check if training is complete based on progress (fallback)
      if (status.progress >= 100 && !status.isTraining && !isComplete.value) {
        isComplete.value = true
        addLog('Training completed successfully!')
      }
      
      // Check for errors
      if (status.error && status.error !== error.value) {
        addLog(`Error: ${status.error}`)
      }
    }, { deep: true })
    
    const updateMetric = (name: string, value: number) => {
      const existingMetric = metrics.value.find(m => m.name === name)
      if (existingMetric) {
        existingMetric.value = value
      } else {
        metrics.value.push({ name, value })
      }
    }
    
    // Set up polling for training status
    const pollingInterval = ref<number | null>(null)
    
    const startStatusPolling = () => {
      // Poll every 2 seconds
      pollingInterval.value = window.setInterval(() => {
        if (trainingStore.trainingStatus.jobId) {
          trainingStore.getTrainingStatus(trainingStore.trainingStatus.jobId)
        }
      }, 2000)
    }
    
    const stopStatusPolling = () => {
      if (pollingInterval.value) {
        clearInterval(pollingInterval.value)
        pollingInterval.value = null
      }
    }
    
    onMounted(() => {
      // Initialize empty metrics
      metrics.value = []
    })
    
    // Start polling when training starts, stop when it completes
    watch(() => trainingStore.trainingStatus.isTraining, (isTraining) => {
      if (isTraining) {
        startStatusPolling()
      } else {
        stopStatusPolling()
      }
    })
    
    // Clean up on component unmount
    onUnmounted(() => {
      stopStatusPolling()
    })
    
    return {
      isTraining,
      progress,
      error,
      logs,
      metrics,
      statusText,
      isComplete,
      logsContainer,
      
      startTraining,
      stopTraining,
      onTrainingComplete,
      clearLogs,
      formatTimestamp,
      formatMetricValue,
      getLearningTypeLabel
    }
  }
})
</script>

<style scoped>
.training-progress h2 {
  margin-top: 0;
  margin-bottom: var(--spacing-lg);
}

.training-summary {
  margin-bottom: var(--spacing-lg);
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: var(--spacing-md);
}

.summary-item {
  padding: var(--spacing-sm);
}

.summary-label {
  font-size: 0.75rem;
  text-transform: uppercase;
  color: var(--color-text-light);
  margin-bottom: 4px;
  font-weight: 500;
  letter-spacing: 0.5px;
}

.summary-value {
  font-weight: 500;
}

.training-status {
  margin-bottom: var(--spacing-lg);
}

.training-status h3 {
  margin-top: 0;
  margin-bottom: var(--spacing-md);
}

.status-indicator {
  display: flex;
  align-items: center;
  margin-bottom: var(--spacing-md);
}

.status-label {
  font-weight: 500;
  margin-right: var(--spacing-sm);
}

.status-value {
  font-weight: 600;
}

.status-ready {
  color: var(--color-text);
}

.status-running {
  color: var(--color-primary);
}

.status-error {
  color: var(--color-error);
}

.progress-bar-container {
  height: 24px;
  background-color: #f5f5f5;
  border-radius: var(--border-radius);
  margin-bottom: var(--spacing-md);
  position: relative;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background-color: var(--color-primary);
  transition: width 0.3s ease;
}

.progress-text {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text);
  font-weight: 500;
  font-size: 0.875rem;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: var(--spacing-md);
  margin-top: var(--spacing-md);
}

.metric-item {
  padding: var(--spacing-sm);
  background-color: #f9f9f9;
  border-radius: var(--border-radius);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.metric-name {
  font-size: 0.75rem;
  color: var(--color-text-light);
  margin-bottom: 4px;
  text-align: center;
}

.metric-value {
  font-weight: 600;
  font-size: 1.25rem;
  color: var(--color-primary);
}

.training-logs {
  margin-bottom: var(--spacing-lg);
}

.training-logs h3 {
  margin-top: 0;
  margin-bottom: var(--spacing-md);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logs-container {
  height: 300px;
  overflow-y: auto;
  background-color: #f5f5f5;
  border-radius: var(--border-radius);
  padding: var(--spacing-sm);
  font-family: 'Roboto Mono', monospace;
  font-size: 0.875rem;
}

.empty-logs {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-light);
}

.log-entry {
  display: flex;
  margin-bottom: 4px;
  line-height: 1.4;
}

.log-timestamp {
  color: var(--color-text-light);
  margin-right: var(--spacing-sm);
  font-size: 0.75rem;
  min-width: 80px;
}

.log-message {
  flex: 1;
  word-break: break-word;
}

.btn-icon {
  background: none;
  border: none;
  color: var(--color-text-light);
  font-size: 1.25rem;
  cursor: pointer;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
}

.btn-icon:hover {
  background-color: rgba(0, 0, 0, 0.05);
  color: var(--color-text);
}

.clear-logs {
  margin-left: var(--spacing-sm);
}

.actions-container {
  display: flex;
  gap: var(--spacing-md);
}

@media (max-width: 768px) {
  .summary-grid {
    grid-template-columns: 1fr 1fr;
  }
  
  .metrics-grid {
    grid-template-columns: 1fr 1fr;
  }
}
</style>