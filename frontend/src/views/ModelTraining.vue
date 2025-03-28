<template>
  <div class="model-training">
    <h1>Model Training</h1>
    
    <div class="training-steps">
      <div class="step" :class="{ 'active': currentStep === 1, 'completed': currentStep > 1 }">
        <div class="step-number">1</div>
        <div class="step-label">Select Dataset</div>
      </div>
      <div class="step-connector"></div>
      <div class="step" :class="{ 'active': currentStep === 2, 'completed': currentStep > 2 }">
        <div class="step-number">2</div>
        <div class="step-label">Select Base Model</div>
      </div>
      <div class="step-connector"></div>
      <div class="step" :class="{ 'active': currentStep === 3, 'completed': currentStep > 3 }">
        <div class="step-number">3</div>
        <div class="step-label">Configure Training</div>
      </div>
      <div class="step-connector"></div>
      <div class="step" :class="{ 'active': currentStep === 4, 'completed': currentStep > 4 }">
        <div class="step-number">4</div>
        <div class="step-label">Train Model</div>
      </div>
    </div>
    
    <!-- Step 1: Dataset Selection -->
    <div v-if="currentStep === 1" class="step-content">
      <dataset-selection @dataset-selected="onDatasetSelected" />
    </div>
    
    <!-- Step 2: Model Selection -->
    <div v-if="currentStep === 2" class="step-content">
      <model-selection @model-selected="onModelSelected" />
    </div>
    
    <!-- Step 3: Training Configuration -->
    <div v-if="currentStep === 3" class="step-content">
      <training-config 
        :dataset="selectedDataset" 
        :model="selectedModel"
        @config-saved="onConfigSaved" 
      />
    </div>
    
    <!-- Step 4: Training Progress -->
    <div v-if="currentStep === 4" class="step-content">
      <training-progress 
        :dataset="selectedDataset" 
        :model="selectedModel"
        :config="trainingConfig"
        @training-complete="onTrainingComplete"
      />
    </div>
    
    <!-- Navigation buttons -->
    <div class="step-navigation">
      <button 
        v-if="currentStep > 1" 
        class="btn btn-secondary" 
        @click="prevStep"
      >
        Previous
      </button>
      
      <button 
        v-if="currentStep < 4 && canProceed" 
        class="btn btn-primary" 
        @click="nextStep"
      >
        Next
      </button>
      
      <button 
        v-if="currentStep === 4 && !isTraining" 
        class="btn btn-primary" 
        @click="startTraining"
      >
        Start Training
      </button>
    </div>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref, computed } from 'vue'
import { useTrainingStore } from '../store/training'
import type { DatasetInfo, ModelInfo, TrainingConfig } from '../store/training'

import DatasetSelection from '../components/training/DatasetSelection.vue'
import ModelSelection from '../components/training/ModelSelection.vue'
import TrainingConfig from '../components/training/TrainingConfig.vue'
import TrainingProgress from '../components/training/TrainingProgress.vue'

export default defineComponent({
  name: 'ModelTraining',
  
  components: {
    DatasetSelection,
    ModelSelection,
    TrainingConfig,
    TrainingProgress
  },
  
  setup() {
    const trainingStore = useTrainingStore()
    const currentStep = ref(1)
    const selectedDataset = ref<DatasetInfo | null>(null)
    const selectedModel = ref<ModelInfo | null>(null)
    const trainingConfig = ref<TrainingConfig | null>(null)
    const isTraining = ref(false)
    
    // Determine if user can proceed to next step
    const canProceed = computed(() => {
      switch (currentStep.value) {
        case 1:
          return selectedDataset.value !== null
        case 2:
          return selectedModel.value !== null
        case 3:
          return trainingConfig.value !== null
        default:
          return false
      }
    })
    
    // Event handlers
    const onDatasetSelected = (dataset: DatasetInfo) => {
      selectedDataset.value = dataset
      trainingStore.setSelectedDataset(dataset)
    }
    
    const onModelSelected = (model: ModelInfo) => {
      selectedModel.value = model
      trainingStore.setSelectedModel(model)
    }
    
    const onConfigSaved = (config: TrainingConfig) => {
      trainingConfig.value = config
      trainingStore.updateCurrentConfig(config)
    }
    
    const onTrainingComplete = () => {
      isTraining.value = false
      // Reset to first step after training is complete
      // You might want to navigate to the model testing page instead
      currentStep.value = 1
      selectedDataset.value = null
      selectedModel.value = null
      trainingConfig.value = null
      trainingStore.resetTrainingStatus()
    }
    
    // Navigation
    const nextStep = () => {
      if (currentStep.value < 4) {
        currentStep.value++
      }
    }
    
    const prevStep = () => {
      if (currentStep.value > 1) {
        currentStep.value--
      }
    }
    
    const startTraining = async () => {
      if (!selectedDataset.value || !selectedModel.value || !trainingConfig.value) {
        return
      }
      
      isTraining.value = true
      
      // Prepare training parameters
      const trainingParams = {
        model_source: selectedModel.value.source,
        dataset_name: selectedDataset.value.name,
        input_field: trainingConfig.value.input_field || 'input',
        output_field: trainingConfig.value.output_field || 'output',
        learning_type: trainingConfig.value.learning_type,
        config: trainingConfig.value,
        output_dir: trainingConfig.value.output_dir || 'models/trained'
      }
      
      // Add model information based on source
      if (selectedModel.value.source === 'huggingface') {
        trainingParams.model_id = selectedModel.value.id
      } else {
        trainingParams.model_path = selectedModel.value.path
      }
      
      // Start training
      await trainingStore.trainModel(trainingParams)
    }
    
    // Initialize store data when component is mounted
    trainingStore.fetchDatasets()
    trainingStore.getHuggingFaceTasks()
    trainingStore.fetchDefaultConfigs()
    
    return {
      currentStep,
      selectedDataset,
      selectedModel,
      trainingConfig,
      isTraining,
      canProceed,
      onDatasetSelected,
      onModelSelected,
      onConfigSaved,
      onTrainingComplete,
      nextStep,
      prevStep,
      startTraining
    }
  }
})
</script>

<style scoped>
.model-training {
  max-width: 1000px;
  margin: 0 auto;
}

.training-steps {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-xl);
  padding: var(--spacing-md);
  background-color: #f9f9f9;
  border-radius: var(--border-radius);
}

.step {
  display: flex;
  flex-direction: column;
  align-items: center;
  position: relative;
}

.step-number {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: var(--color-border);
  color: var(--color-text);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  margin-bottom: var(--spacing-xs);
  transition: all 0.3s ease;
}

.step.active .step-number {
  background-color: var(--color-primary);
  color: white;
}

.step.completed .step-number {
  background-color: var(--color-success);
  color: white;
}

.step-label {
  font-size: 0.875rem;
  color: var(--color-text-light);
  transition: all 0.3s ease;
}

.step.active .step-label {
  color: var(--color-primary);
  font-weight: 500;
}

.step.completed .step-label {
  color: var(--color-success);
}

.step-connector {
  height: 2px;
  background-color: var(--color-border);
  flex-grow: 1;
  margin: 0 var(--spacing-sm);
}

.step-content {
  background-color: white;
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius);
  padding: var(--spacing-lg);
  margin-bottom: var(--spacing-lg);
  min-height: 400px;
}

.step-navigation {
  display: flex;
  justify-content: space-between;
  margin-top: var(--spacing-lg);
}

@media (max-width: 768px) {
  .training-steps {
    flex-direction: column;
    gap: var(--spacing-md);
  }
  
  .step-connector {
    width: 2px;
    height: 20px;
  }
}
</style>