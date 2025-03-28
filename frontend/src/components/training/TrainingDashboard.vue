<template>
  <div class="training-dashboard">
    <div class="dashboard-header">
      <h2>Training Dashboard</h2>
      <div class="header-actions">
        <button class="btn btn-secondary btn-sm" @click="refreshJobs" title="Refresh jobs">
          <span v-if="loading" class="loading-spinner"></span>
          <span v-else>Refresh</span>
        </button>
      </div>
    </div>
    
    <div v-if="error" class="alert alert-error">
      {{ error }}
    </div>
    
    <div v-if="loading && jobs.length === 0" class="loading-container">
      <div class="loading-spinner"></div>
      <p>Loading training jobs...</p>
    </div>
    
    <div v-else-if="jobs.length === 0" class="empty-state">
      <div class="empty-icon">📊</div>
      <h3>No Training Jobs Found</h3>
      <p>Start a new training job to see metrics and monitor progress.</p>
      <router-link to="/training" class="btn btn-primary">Start Training</router-link>
    </div>
    
    <div v-else class="jobs-container">
      <div v-for="job in jobs" :key="job.job_id" class="job-card">
        <div class="job-header">
          <div class="job-title">
            <h3>{{ job.model_id }}</h3>
            <div class="job-status" :class="`status-${job.status}`">{{ getStatusLabel(job.status) }}</div>
          </div>
          <div class="job-actions">
            <button 
              v-if="['training', 'evaluating'].includes(job.status)" 
              class="btn btn-danger btn-sm" 
              @click="stopJob(job.job_id)"
              title="Stop training"
            >
              Stop
            </button>
            <button 
              class="btn btn-secondary btn-sm" 
              @click="showJobDetails(job.job_id)"
              title="View details"
            >
              Details
            </button>
          </div>
        </div>
        
        <div class="job-info">
          <div class="info-item">
            <div class="info-label">Dataset</div>
            <div class="info-value">{{ job.dataset }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Started</div>
            <div class="info-value">{{ formatTimestamp(job.start_time) }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">Duration</div>
            <div class="info-value">{{ calculateDuration(job) }}</div>
          </div>
        </div>
        
        <div class="progress-section">
          <div class="progress-header">
            <span>Progress</span>
            <span class="progress-percent">{{ job.progress.toFixed(1) }}%</span>
          </div>
          <div class="progress-bar-container">
            <div class="progress-bar" :style="{ width: `${job.progress}%` }"></div>
          </div>
          <div class="progress-details">
            <span v-if="job.total_epochs">Epoch {{ job.current_epoch || 0 }}/{{ job.total_epochs }}</span>
            <span v-if="job.current_step">Step {{ job.current_step }}</span>
          </div>
        </div>
        
        <div v-if="Object.keys(job.metrics).length > 0" class="metrics-section">
          <h4>Metrics</h4>
          <div class="metrics-grid">
            <div 
              v-for="(value, key) in job.metrics" 
              :key="`${job.job_id}-${key}`" 
              class="metric-item"
            >
              <div class="metric-name">{{ formatMetricName(key) }}</div>
              <div class="metric-value">{{ formatMetricValue(value) }}</div>
            </div>
          </div>
          
          <div class="metrics-chart">
            <!-- Placeholder for metrics chart - can be replaced with a charting library -->
            <div class="chart-placeholder">
              <div v-for="(value, key) in job.metrics" 
                :key="`chart-${job.job_id}-${key}`" 
                class="chart-bar"
                :style="{ height: `${calculateBarHeight(value)}px` }"
                :title="`${formatMetricName(key)}: ${value}`"
              >
                <div class="chart-bar-label">{{ key.substr(0, 3) }}</div>
              </div>
            </div>
          </div>
        </div>
        
        <div v-if="job.error" class="job-error">
          <h4>Error</h4>
          <div class="error-message">{{ job.error }}</div>
        </div>
        
        <div v-if="selectedJobId === job.job_id" class="job-details">
          <h4>Training Logs</h4>
          <div class="logs-container">
            <div v-if="job.logs && job.logs.length > 0" class="logs-content">
              <div 
                v-for="(log, index) in job.logs" 
                :key="`${job.job_id}-log-${index}`" 
                class="log-entry"
              >
                <div class="log-timestamp">{{ formatTimestamp(log.timestamp) }}</div>
                <div class="log-message">{{ log.message }}</div>
              </div>
            </div>
            <div v-else class="empty-logs">
              No logs available.
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, onMounted, onUnmounted } from 'vue'
import axios from 'axios'

interface TrainingLog {
  timestamp: number
  message: string
}

interface TrainingJob {
  job_id: string
  model_id: string
  dataset: string
  config: Record<string, any>
  start_time: number
  status: string
  progress: number
  current_epoch: number
  total_epochs: number
  current_step: number
  total_steps: number
  metrics: Record<string, number>
  logs: TrainingLog[]
  error: string | null
}

export default defineComponent({
  name: 'TrainingDashboard',
  
  setup() {
    const jobs = ref<TrainingJob[]>([])
    const loading = ref(false)
    const error = ref('')
    const selectedJobId = ref<string | null>(null)
    const pollingInterval = ref<number | null>(null)
    
    const refreshJobs = async () => {
      loading.value = true
      try {
        const response = await axios.get('/api/monitor/jobs')
        if (response.data && response.data.jobs) {
          jobs.value = response.data.jobs
        }
        error.value = ''
      } catch (err: any) {
        console.error('Error fetching training jobs:', err)
        error.value = err.message || 'Failed to fetch training jobs'
      } finally {
        loading.value = false
      }
    }
    
    const stopJob = async (jobId: string) => {
      try {
        await axios.post(`/api/monitor/jobs/${jobId}/stop`)
        // Update the job status locally
        const job = jobs.value.find(j => j.job_id === jobId)
        if (job) {
          job.status = 'stopping'
        }
      } catch (err: any) {
        console.error(`Error stopping job ${jobId}:`, err)
        error.value = err.message || `Failed to stop job ${jobId}`
      }
    }
    
    const showJobDetails = (jobId: string) => {
      if (selectedJobId.value === jobId) {
        selectedJobId.value = null
      } else {
        selectedJobId.value = jobId
      }
    }
    
    const formatTimestamp = (timestamp: number) => {
      const date = new Date(timestamp * 1000)
      return date.toLocaleString()
    }
    
    const calculateDuration = (job: TrainingJob) => {
      const startTime = job.start_time * 1000
      const endTime = ['completed', 'error', 'stopped'].includes(job.status) 
        ? Date.now() // Use current time for ongoing jobs
        : job.logs[job.logs.length - 1]?.timestamp * 1000 || Date.now()
      
      const durationMs = endTime - startTime
      
      // Format as hours:minutes:seconds
      const seconds = Math.floor((durationMs / 1000) % 60)
      const minutes = Math.floor((durationMs / (1000 * 60)) % 60)
      const hours = Math.floor(durationMs / (1000 * 60 * 60))
      
      if (hours > 0) {
        return `${hours}h ${minutes}m ${seconds}s`
      } else if (minutes > 0) {
        return `${minutes}m ${seconds}s`
      } else {
        return `${seconds}s`
      }
    }
    
    const formatMetricName = (key: string) => {
      // Convert snake_case to Title Case
      return key
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')
    }
    
    const formatMetricValue = (value: number) => {
      // Format metric values with appropriate precision
      if (Math.abs(value) < 0.001) {
        return value.toExponential(2)
      } else if (Math.abs(value) < 0.01) {
        return value.toFixed(4)
      } else {
        return value.toFixed(3)
      }
    }
    
    const calculateBarHeight = (value: number) => {
      // For the simple chart, scale values to a reasonable height
      // between 10 and 100 pixels
      const MIN_HEIGHT = 10
      const MAX_HEIGHT = 100
      
      // Handle different types of metrics
      if (value < 0) {
        // Negative values (like losses) - inverse scale
        return MIN_HEIGHT + (Math.min(Math.abs(value), 10) / 10) * (MAX_HEIGHT - MIN_HEIGHT)
      } else if (value <= 1) {
        // Values between 0 and 1 (like accuracy)
        return MIN_HEIGHT + (value * (MAX_HEIGHT - MIN_HEIGHT))
      } else {
        // Values greater than 1 (like perplexity)
        return MIN_HEIGHT + (Math.min(value, 10) / 10) * (MAX_HEIGHT - MIN_HEIGHT)
      }
    }
    
    const getStatusLabel = (status: string) => {
      const statusMap: Record<string, string> = {
        'initializing': 'Initializing',
        'training': 'Training',
        'evaluating': 'Evaluating',
        'completed': 'Completed',
        'error': 'Error',
        'stopping': 'Stopping',
        'stopped': 'Stopped'
      }
      return statusMap[status] || status
    }
    
    const startPolling = () => {
      // Poll every 10 seconds
      pollingInterval.value = window.setInterval(() => {
        refreshJobs()
      }, 10000)
    }
    
    const stopPolling = () => {
      if (pollingInterval.value) {
        clearInterval(pollingInterval.value)
        pollingInterval.value = null
      }
    }
    
    onMounted(() => {
      refreshJobs()
      startPolling()
    })
    
    onUnmounted(() => {
      stopPolling()
    })
    
    return {
      jobs,
      loading,
      error,
      selectedJobId,
      refreshJobs,
      stopJob,
      showJobDetails,
      formatTimestamp,
      calculateDuration,
      formatMetricName,
      formatMetricValue,
      calculateBarHeight,
      getStatusLabel
    }
  }
})
</script>

<style scoped>
.training-dashboard {
  max-width: 1200px;
  margin: 0 auto;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.dashboard-header h2 {
  margin: 0;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-xl) 0;
  text-align: center;
}

.empty-icon {
  font-size: 3rem;
  margin-bottom: var(--spacing-md);
}

.jobs-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: var(--spacing-lg);
}

.job-card {
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  padding: var(--spacing-md);
  background-color: white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.job-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: var(--spacing-md);
}

.job-title h3 {
  margin: 0 0 var(--spacing-xs) 0;
  font-size: 1.125rem;
}

.job-status {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
}

.status-initializing {
  background-color: rgba(0, 0, 0, 0.05);
  color: var(--color-text);
}

.status-training, .status-evaluating {
  background-color: rgba(16, 163, 127, 0.1);
  color: var(--color-primary);
}

.status-completed {
  background-color: rgba(52, 211, 153, 0.1);
  color: var(--color-success);
}

.status-error {
  background-color: rgba(239, 68, 68, 0.1);
  color: var(--color-error);
}

.status-stopping, .status-stopped {
  background-color: rgba(245, 158, 11, 0.1);
  color: #f59e0b;
}

.job-info {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-md);
}

.info-label {
  font-size: 0.75rem;
  color: var(--color-text-light);
  margin-bottom: 2px;
}

.info-value {
  font-size: 0.875rem;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.progress-section {
  margin-bottom: var(--spacing-md);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
  font-size: 0.875rem;
}

.progress-percent {
  font-weight: 600;
}

.progress-bar-container {
  height: 8px;
  background-color: #f5f5f5;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 4px;
}

.progress-bar {
  height: 100%;
  background-color: var(--color-primary);
  transition: width 0.3s ease;
}

.progress-details {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--color-text-light);
}

.metrics-section {
  margin-bottom: var(--spacing-md);
}

.metrics-section h4 {
  font-size: 0.875rem;
  margin: 0 0 var(--spacing-sm) 0;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(70px, 1fr));
  gap: var(--spacing-sm);
  margin-bottom: var(--spacing-sm);
}

.metric-item {
  background-color: #f9f9f9;
  padding: var(--spacing-xs);
  border-radius: var(--border-radius);
  text-align: center;
}

.metric-name {
  font-size: 0.675rem;
  color: var(--color-text-light);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 2px;
}

.metric-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--color-primary);
}

.metrics-chart {
  height: 120px;
  margin-top: var(--spacing-md);
}

.chart-placeholder {
  height: 100%;
  display: flex;
  align-items: flex-end;
  justify-content: space-around;
  background-color: #f9f9f9;
  border-radius: var(--border-radius);
  padding: var(--spacing-xs);
}

.chart-bar {
  width: 30px;
  background-color: rgba(16, 163, 127, 0.7);
  border-radius: 4px 4px 0 0;
  position: relative;
  transition: height 0.3s ease;
}

.chart-bar-label {
  position: absolute;
  bottom: -20px;
  left: 0;
  right: 0;
  text-align: center;
  font-size: 0.675rem;
  color: var(--color-text-light);
}

.job-error {
  margin-bottom: var(--spacing-md);
  padding: var(--spacing-sm);
  background-color: rgba(239, 68, 68, 0.05);
  border-radius: var(--border-radius);
  border-left: 3px solid var(--color-error);
}

.job-error h4 {
  font-size: 0.875rem;
  margin: 0 0 var(--spacing-xs) 0;
  color: var(--color-error);
}

.error-message {
  font-size: 0.875rem;
  font-family: 'Roboto Mono', monospace;
  white-space: pre-wrap;
}

.job-details {
  margin-top: var(--spacing-md);
  border-top: 1px solid var(--color-border);
  padding-top: var(--spacing-md);
}

.job-details h4 {
  font-size: 0.875rem;
  margin: 0 0 var(--spacing-sm) 0;
}

.logs-container {
  height: 200px;
  overflow-y: auto;
  background-color: #f5f5f5;
  border-radius: var(--border-radius);
  padding: var(--spacing-xs);
  font-family: 'Roboto Mono', monospace;
  font-size: 0.75rem;
}

.log-entry {
  display: flex;
  margin-bottom: 4px;
}

.log-timestamp {
  flex-shrink: 0;
  color: var(--color-text-light);
  margin-right: var(--spacing-sm);
}

.log-message {
  word-break: break-word;
}

.empty-logs {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-light);
  font-style: italic;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: var(--spacing-xl) 0;
}

.btn-sm {
  font-size: 0.75rem;
  padding: 4px 8px;
}

@media (max-width: 768px) {
  .jobs-container {
    grid-template-columns: 1fr;
  }
  
  .job-info {
    grid-template-columns: 1fr 1fr;
  }
}
</style>