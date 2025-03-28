<template>
  <div class="training-config">
    <h2>Configure Training</h2>
    
    <div v-if="!dataset || !model" class="alert alert-error">
      Please select both a dataset and a model before configuring training.
    </div>
    
    <div v-else>
      <div class="config-summary card">
        <div class="summary-item">
          <h3>Dataset</h3>
          <div class="summary-content">
            <div class="item-label">Name:</div>
            <div class="item-value">{{ dataset.name }}</div>
          </div>
          <div class="summary-content">
            <div class="item-label">Format:</div>
            <div class="item-value">{{ dataset.format.toUpperCase() }}</div>
          </div>
          <div class="summary-content" v-if="dataset.columns && dataset.columns.length > 0">
            <div class="item-label">Columns:</div>
            <div class="item-value">{{ dataset.columns.join(', ') }}</div>
          </div>
        </div>
        
        <div class="summary-item">
          <h3>Model</h3>
          <div class="summary-content">
            <div class="item-label">Name:</div>
            <div class="item-value">{{ model.name }}</div>
          </div>
          <div class="summary-content">
            <div class="item-label">Source:</div>
            <div class="item-value">{{ model.source === 'huggingface' ? 'Hugging Face' : 'Local' }}</div>
          </div>
          <div class="summary-content" v-if="model.type">
            <div class="item-label">Type:</div>
            <div class="item-value">{{ model.type }}</div>
          </div>
          <div class="summary-content" v-if="model.path">
            <div class="item-label">Path:</div>
            <div class="item-value path-value">{{ model.path }}</div>
          </div>
        </div>
      </div>
      
      <div class="training-form card">
        <h3>Training Configuration</h3>
        
        <div class="form-group">
          <label class="form-label required">Learning Type</label>
          <select v-model="config.learning_type" class="form-control" @change="updateConfigBasedOnLearningType">
            <option v-for="type in learningTypes" :key="type.value" :value="type.value">
              {{ type.label }}
            </option>
          </select>
          <div class="learning-type-description">
            {{ getLearningTypeDescription(config.learning_type) }}
          </div>
        </div>
        
        <div class="form-group field-mapping" v-if="dataset.columns && dataset.columns.length > 0">
          <label class="form-label required">Field Mapping</label>
          
          <div class="subform-group">
            <label class="subform-label">Input Field</label>
            <select v-model="config.input_field" class="form-control">
              <option v-for="column in dataset.columns" :key="column" :value="column">
                {{ column }}
              </option>
            </select>
          </div>
          
          <div class="subform-group" v-if="needsOutputField">
            <label class="subform-label">Output Field</label>
            <select v-model="config.output_field" class="form-control">
              <option v-for="column in dataset.columns" :key="column" :value="column">
                {{ column }}
              </option>
            </select>
          </div>
        </div>
        
        <div class="form-group">
          <label class="form-label">Output Directory</label>
          <input type="text" v-model="config.output_dir" class="form-control" placeholder="models/trained">
          <div class="form-hint">Directory where the trained model will be saved</div>
        </div>
        
        <div class="form-divider"></div>
        
        <div class="form-tabs">
          <button 
            :class="['tab-btn', { active: activeTab === 'basic' }]"
            @click="activeTab = 'basic'"
          >
            Basic Parameters
          </button>
          <button 
            :class="['tab-btn', { active: activeTab === 'advanced' }]"
            @click="activeTab = 'advanced'"
          >
            Advanced Parameters
          </button>
        </div>
        
        <div class="tab-content">
          <!-- Basic Parameters Tab -->
          <div v-if="activeTab === 'basic'" class="basic-parameters">
            <div class="form-grid">
              <div class="form-group">
                <label class="form-label">Batch Size</label>
                <input 
                  type="number" 
                  v-model.number="config.batch_size" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Number of samples processed before model update</div>
              </div>
              
              <div class="form-group">
                <label class="form-label">Learning Rate</label>
                <input 
                  type="number" 
                  v-model.number="config.learning_rate" 
                  class="form-control"
                  step="0.0001"
                  min="0"
                >
                <div class="form-hint">Step size for gradient descent optimization</div>
              </div>
              
              <div class="form-group" v-if="showEpochsField">
                <label class="form-label">Training Epochs</label>
                <input 
                  type="number" 
                  v-model.number="config.num_train_epochs" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Number of complete passes through the dataset</div>
              </div>
              
              <div class="form-group" v-if="showMaxStepsField">
                <label class="form-label">Max Steps</label>
                <input 
                  type="number" 
                  v-model.number="config.max_steps" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Maximum number of training steps</div>
              </div>
              
              <div class="form-group">
                <label class="form-label">Weight Decay</label>
                <input 
                  type="number" 
                  v-model.number="config.weight_decay" 
                  class="form-control"
                  step="0.01"
                  min="0"
                >
                <div class="form-hint">L2 regularization to prevent overfitting</div>
              </div>
              
              <div class="form-group">
                <label class="form-label">Max Sequence Length</label>
                <input 
                  type="number" 
                  v-model.number="config.max_seq_length" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Maximum length of sequences for processing</div>
              </div>
              
              <!-- Learning type specific fields - Reinforcement Learning -->
              <div v-if="config.learning_type === 'reinforcement'" class="form-group">
                <label class="form-label">Discount Factor</label>
                <input 
                  type="number" 
                  v-model.number="config.discount_factor" 
                  class="form-control"
                  step="0.01"
                  min="0"
                  max="1"
                >
                <div class="form-hint">Value between 0-1 determining importance of future rewards</div>
              </div>
              
              <!-- Learning type specific fields - Semi-supervised Learning -->
              <div v-if="config.learning_type === 'semi_supervised'" class="form-group">
                <label class="form-label">Unlabeled Weight</label>
                <input 
                  type="number" 
                  v-model.number="config.unlabeled_weight" 
                  class="form-control"
                  step="0.1"
                  min="0"
                >
                <div class="form-hint">Weight of unlabeled data in the loss function</div>
              </div>
              
              <!-- Learning type specific fields - Online Learning -->
              <div v-if="config.learning_type === 'online'" class="form-group">
                <label class="form-label">Window Size</label>
                <input 
                  type="number" 
                  v-model.number="config.window_size" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Size of the sliding window for online learning</div>
              </div>
              
              <!-- Learning type specific fields - Federated Learning -->
              <div v-if="config.learning_type === 'federated'" class="form-group">
                <label class="form-label">Number of Clients</label>
                <input 
                  type="number" 
                  v-model.number="config.num_clients" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Number of simulated clients for federated training</div>
              </div>
            </div>
          </div>
          
          <!-- Advanced Parameters Tab -->
          <div v-if="activeTab === 'advanced'" class="advanced-parameters">
            <div class="form-grid">
              <div class="form-group">
                <label class="form-label">Warmup Steps</label>
                <input 
                  type="number" 
                  v-model.number="config.warmup_steps" 
                  class="form-control"
                  min="0"
                >
                <div class="form-hint">Number of steps for learning rate warmup</div>
              </div>
              
              <div class="form-group" v-if="showSaveStepsField">
                <label class="form-label">Save Steps</label>
                <input 
                  type="number" 
                  v-model.number="config.save_steps" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Save checkpoint every X steps</div>
              </div>
              
              <div class="form-group" v-if="config.learning_type === 'supervised'">
                <label class="form-label">Evaluation Strategy</label>
                <select v-model="config.evaluation_strategy" class="form-control">
                  <option value="no">No evaluation</option>
                  <option value="steps">Evaluate every X steps</option>
                  <option value="epoch">Evaluate every epoch</option>
                </select>
                <div class="form-hint">When to perform model evaluation</div>
              </div>
              
              <!-- Reinforcement Learning specific fields -->
              <div v-if="config.learning_type === 'reinforcement'" class="form-group">
                <label class="form-label">Target Update Interval</label>
                <input 
                  type="number" 
                  v-model.number="config.target_update_interval" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Steps between target network updates</div>
              </div>
              
              <div v-if="config.learning_type === 'reinforcement'" class="form-group">
                <label class="form-label">Replay Buffer Size</label>
                <input 
                  type="number" 
                  v-model.number="config.replay_buffer_size" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Size of experience replay buffer</div>
              </div>
              
              <div v-if="config.learning_type === 'reinforcement'" class="form-group">
                <label class="form-label">Exploration Rate</label>
                <input 
                  type="number" 
                  v-model.number="config.exploration_rate" 
                  class="form-control"
                  step="0.01"
                  min="0"
                  max="1"
                >
                <div class="form-hint">Initial exploration rate (epsilon) for epsilon-greedy policy</div>
              </div>
              
              <!-- Online Learning specific fields -->
              <div v-if="config.learning_type === 'online'" class="form-group">
                <label class="form-label">Forget Factor</label>
                <input 
                  type="number" 
                  v-model.number="config.forget_factor" 
                  class="form-control"
                  step="0.01"
                  min="0"
                  max="1"
                >
                <div class="form-hint">Rate at which old samples are forgotten</div>
              </div>
              
              <div v-if="config.learning_type === 'online'" class="form-group">
                <label class="form-label">Update Interval</label>
                <input 
                  type="number" 
                  v-model.number="config.update_interval" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Steps between model updates</div>
              </div>
              
              <div v-if="config.learning_type === 'online'" class="form-group">
                <label class="form-label">Max Samples</label>
                <input 
                  type="number" 
                  v-model.number="config.max_samples" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Maximum number of samples to process</div>
              </div>
              
              <!-- Federated Learning specific fields -->
              <div v-if="config.learning_type === 'federated'" class="form-group">
                <label class="form-label">Client Fraction</label>
                <input 
                  type="number" 
                  v-model.number="config.client_fraction" 
                  class="form-control"
                  step="0.01"
                  min="0"
                  max="1"
                >
                <div class="form-hint">Fraction of clients to use in each round</div>
              </div>
              
              <div v-if="config.learning_type === 'federated'" class="form-group">
                <label class="form-label">Local Epochs</label>
                <input 
                  type="number" 
                  v-model.number="config.local_epochs" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Number of local training epochs per client</div>
              </div>
              
              <div v-if="config.learning_type === 'federated'" class="form-group">
                <label class="form-label">Number of Rounds</label>
                <input 
                  type="number" 
                  v-model.number="config.num_rounds" 
                  class="form-control"
                  min="1"
                >
                <div class="form-hint">Number of federated training rounds</div>
              </div>
              
              <div v-if="config.learning_type === 'federated'" class="form-group">
                <label class="form-label">Aggregation Method</label>
                <select v-model="config.aggregation" class="form-control">
                  <option value="fedavg">FedAvg</option>
                  <option value="fedsgd">FedSGD</option>
                  <option value="fedprox">FedProx</option>
                </select>
                <div class="form-hint">Method to aggregate client models</div>
              </div>
            </div>
          </div>
        </div>
        
        <div class="form-actions">
          <button class="btn btn-secondary" @click="resetToDefaults">
            Reset to Defaults
          </button>
          <button 
            class="btn btn-primary" 
            :disabled="!isValid"
            @click="saveConfig"
          >
            Save Configuration
          </button>
        </div>
        
        <div v-if="configSaved" class="alert alert-success">
          Configuration saved successfully!
        </div>
        
        <div v-if="validationErrors.length > 0" class="validation-errors">
          <div class="alert alert-error">
            <strong>Please fix the following errors:</strong>
            <ul>
              <li v-for="(error, index) in validationErrors" :key="index">{{ error }}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed, watch, onMounted } from 'vue'
import { useTrainingStore } from '../../store/training'
import type { DatasetInfo, ModelInfo, TrainingConfig } from '../../store/training'

export default defineComponent({
  name: 'TrainingConfiguration',
  
  props: {
    dataset: {
      type: Object as () => DatasetInfo | null,
      default: null
    },
    model: {
      type: Object as () => ModelInfo | null,
      default: null
    }
  },
  
  emits: ['config-saved'],
  
  setup(props, { emit }) {
    const trainingStore = useTrainingStore()
    const activeTab = ref('basic')
    const configSaved = ref(false)
    const validationErrors = ref<string[]>([])
    
    // Define learning types
    const learningTypes = ref([
      { value: 'supervised', label: 'Supervised Learning' },
      { value: 'unsupervised', label: 'Unsupervised Learning' },
      { value: 'reinforcement', label: 'Reinforcement Learning' },
      { value: 'semi_supervised', label: 'Semi-supervised Learning' },
      { value: 'self_supervised', label: 'Self-supervised Learning' },
      { value: 'online', label: 'Online Learning' },
      { value: 'federated', label: 'Federated Learning' }
    ])
    
    // Initialize config with default values
    const config = ref<TrainingConfig>({
      learning_type: 'supervised',
      batch_size: 8,
      learning_rate: 2e-5,
      num_train_epochs: 3,
      weight_decay: 0.01,
      warmup_steps: 500,
      max_seq_length: 128,
      save_steps: 10000,
      evaluation_strategy: 'epoch',
      output_dir: 'models/trained'
    })
    
    // Computed properties for conditional field display
    const needsOutputField = computed(() => {
      return ['supervised', 'semi_supervised'].includes(config.value.learning_type)
    })
    
    const showEpochsField = computed(() => {
      return !['reinforcement', 'online'].includes(config.value.learning_type)
    })
    
    const showMaxStepsField = computed(() => {
      return ['reinforcement', 'online'].includes(config.value.learning_type)
    })
    
    const showSaveStepsField = computed(() => {
      return !['online', 'federated'].includes(config.value.learning_type)
    })
    
    // Validation
    const isValid = computed(() => {
      validateConfig()
      return validationErrors.value.length === 0
    })
    
    // Methods
    const validateConfig = () => {
      const errors: string[] = []
      
      // Required fields for all learning types
      if (!config.value.learning_type) {
        errors.push('Learning type is required')
      }
      
      if (!config.value.batch_size || config.value.batch_size <= 0) {
        errors.push('Batch size must be a positive number')
      }
      
      if (!config.value.learning_rate || config.value.learning_rate <= 0) {
        errors.push('Learning rate must be a positive number')
      }
      
      if (showEpochsField.value && (!config.value.num_train_epochs || config.value.num_train_epochs <= 0)) {
        errors.push('Number of training epochs must be a positive number')
      }
      
      if (showMaxStepsField.value && (!config.value.max_steps || config.value.max_steps <= 0)) {
        errors.push('Max steps must be a positive number')
      }
      
      if (!config.value.max_seq_length || config.value.max_seq_length <= 0) {
        errors.push('Max sequence length must be a positive number')
      }
      
      // Check if we need field mapping
      if (props.dataset?.columns && props.dataset.columns.length > 0) {
        if (!config.value.input_field) {
          errors.push('Input field is required')
        }
        
        if (needsOutputField.value && !config.value.output_field) {
          errors.push('Output field is required for this learning type')
        }
        
        if (config.value.input_field === config.value.output_field && needsOutputField.value) {
          errors.push('Input and output fields must be different')
        }
      }
      
      // Learning type specific validations
      if (config.value.learning_type === 'reinforcement') {
        if (config.value.discount_factor === undefined || config.value.discount_factor < 0 || config.value.discount_factor > 1) {
          errors.push('Discount factor must be between 0 and 1')
        }
        
        if (!config.value.target_update_interval || config.value.target_update_interval <= 0) {
          errors.push('Target update interval must be a positive number')
        }
      }
      
      if (config.value.learning_type === 'federated') {
        if (!config.value.num_clients || config.value.num_clients <= 0) {
          errors.push('Number of clients must be a positive number')
        }
        
        if (config.value.client_fraction === undefined || config.value.client_fraction <= 0 || config.value.client_fraction > 1) {
          errors.push('Client fraction must be between 0 and 1')
        }
      }
      
      validationErrors.value = errors
    }
    
    const updateConfigBasedOnLearningType = () => {
      // Load default configuration for the selected learning type
      if (trainingStore.defaultConfigs[config.value.learning_type]) {
        const defaultConfig = trainingStore.defaultConfigs[config.value.learning_type]
        
        // Preserve current output dir and fields mapping
        const outputDir = config.value.output_dir
        const inputField = config.value.input_field
        const outputField = config.value.output_field
        
        // Apply default config for selected learning type
        Object.assign(config.value, defaultConfig)
        
        // Restore preserved values
        config.value.output_dir = outputDir
        config.value.input_field = inputField
        config.value.output_field = outputField
      }
    }
    
    const resetToDefaults = () => {
      // Load default configuration for the current learning type
      if (trainingStore.defaultConfigs[config.value.learning_type]) {
        const defaultConfig = trainingStore.defaultConfigs[config.value.learning_type]
        
        // Reset to defaults but preserve learning type, output dir, and field mapping
        const learningType = config.value.learning_type
        const outputDir = config.value.output_dir
        const inputField = config.value.input_field
        const outputField = config.value.output_field
        
        Object.assign(config.value, defaultConfig)
        
        config.value.learning_type = learningType
        config.value.output_dir = outputDir
        config.value.input_field = inputField
        config.value.output_field = outputField
      }
    }
    
    const saveConfig = () => {
      if (!isValid.value) return
      
      // Add field mapping to config
      if (props.dataset?.columns && props.dataset.columns.length > 0) {
        // Make sure input_field is set
        if (!config.value.input_field && props.dataset.columns.length > 0) {
          config.value.input_field = props.dataset.columns[0]
        }
        
        // For supervised learning, set output_field if not already set
        if (needsOutputField.value && !config.value.output_field && props.dataset.columns.length > 1) {
          // Try to find a different column than input_field
          const otherColumns = props.dataset.columns.filter(col => col !== config.value.input_field)
          if (otherColumns.length > 0) {
            config.value.output_field = otherColumns[0]
          }
        }
      }
      
      // Save config in store
      trainingStore.updateCurrentConfig(config.value)
      
      // Notify parent component
      emit('config-saved', config.value)
      
      // Show success message briefly
      configSaved.value = true
      setTimeout(() => {
        configSaved.value = false
      }, 3000)
    }
    
    const getLearningTypeDescription = (type: string) => {
      switch (type) {
        case 'supervised':
          return 'Model learns from labeled data with input-output pairs'
        case 'unsupervised':
          return 'Model discovers patterns in unlabeled data'
        case 'reinforcement':
          return 'Model learns through trial and error with rewards/penalties'
        case 'semi_supervised':
          return 'Model learns from a combination of labeled and unlabeled data'
        case 'self_supervised':
          return 'Model generates its own labels from input data'
        case 'online':
          return 'Model learns incrementally from streaming data'
        case 'federated':
          return 'Model trains across multiple decentralized devices or servers'
        default:
          return ''
      }
    }
    
    // Initialize default configs when component mounts
    onMounted(async () => {
      await trainingStore.fetchDefaultConfigs()
    })
    
    // Watch for changes in dataset and update input/output fields
    watch(() => props.dataset, (newDataset) => {
      if (newDataset?.columns && newDataset.columns.length > 0) {
        // Set input field to first column if not already set
        if (!config.value.input_field) {
          config.value.input_field = newDataset.columns[0]
        }
        
        // For supervised learning, set output field to second column if available
        if (needsOutputField.value && !config.value.output_field && newDataset.columns.length > 1) {
          config.value.output_field = newDataset.columns[1]
        }
      }
    }, { immediate: true })
    
    return {
      config,
      activeTab,
      learningTypes,
      configSaved,
      validationErrors,
      needsOutputField,
      showEpochsField,
      showMaxStepsField,
      showSaveStepsField,
      isValid,
      
      updateConfigBasedOnLearningType,
      resetToDefaults,
      saveConfig,
      getLearningTypeDescription
    }
  }
})
</script>

<style scoped>
.training-config h2 {
  margin-top: 0;
  margin-bottom: var(--spacing-lg);
}

.config-summary {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--spacing-md);
  margin-bottom: var(--spacing-lg);
}

.summary-item {
  padding: var(--spacing-md);
  background-color: #f9f9f9;
  border-radius: var(--border-radius);
}

.summary-item h3 {
  margin-top: 0;
  margin-bottom: var(--spacing-sm);
  font-size: 1rem;
  color: var(--color-text-light);
}

.summary-content {
  display: flex;
  margin-bottom: var(--spacing-xs);
}

.item-label {
  width: 80px;
  font-weight: 500;
  color: var(--color-text);
}

.item-value {
  flex: 1;
  word-break: break-word;
}

.path-value {
  font-family: 'Roboto Mono', monospace;
  font-size: 0.875rem;
  overflow-wrap: anywhere;
}

.training-form h3 {
  margin-top: 0;
  margin-bottom: var(--spacing-md);
}

.form-tabs {
  display: flex;
  margin-bottom: var(--spacing-md);
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

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: var(--spacing-md);
}

.form-divider {
  margin: var(--spacing-md) 0;
  border-top: 1px solid var(--color-border);
}

.form-hint {
  font-size: 0.75rem;
  color: var(--color-text-light);
  margin-top: 4px;
}

.form-label.required::after {
  content: '*';
  color: var(--color-error);
  margin-left: 4px;
}

.field-mapping {
  margin-bottom: var(--spacing-md);
}

.subform-group {
  margin-bottom: var(--spacing-sm);
}

.subform-label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.875rem;
  font-weight: 500;
}

.learning-type-description {
  font-size: 0.875rem;
  color: var(--color-text-light);
  margin-top: 4px;
  font-style: italic;
}

.form-actions {
  margin-top: var(--spacing-lg);
  display: flex;
  justify-content: flex-end;
  gap: var(--spacing-md);
}

.validation-errors {
  margin-top: var(--spacing-md);
}

.validation-errors ul {
  margin-top: var(--spacing-xs);
  padding-left: var(--spacing-md);
}

@media (max-width: 768px) {
  .config-summary {
    grid-template-columns: 1fr;
  }
  
  .form-grid {
    grid-template-columns: 1fr;
  }
}
</style>