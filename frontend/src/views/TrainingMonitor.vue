<template>
  <div class="training-monitor">
    <div class="monitor-header">
      <h1>Training Monitor</h1>
      <div class="header-actions">
        <router-link to="/training" class="btn btn-primary">New Training</router-link>
      </div>
    </div>
    
    <div class="monitor-tabs">
      <button 
        class="tab-button" 
        :class="{ active: activeTab === 'dashboard' }" 
        @click="activeTab = 'dashboard'"
      >
        Dashboard
      </button>
      <button 
        class="tab-button" 
        :class="{ active: activeTab === 'history' }" 
        @click="activeTab = 'history'"
      >
        History
      </button>
      <button 
        class="tab-button" 
        :class="{ active: activeTab === 'visualization' }" 
        @click="activeTab = 'visualization'"
      >
        Visualizations
      </button>
    </div>
    
    <div class="tab-content">
      <div v-if="activeTab === 'dashboard'" class="tab-pane active">
        <TrainingDashboard />
      </div>
      
      <div v-else-if="activeTab === 'history'" class="tab-pane active">
        <div class="content-placeholder">
          <h2>Training History</h2>
          <p>View and compare past training runs.</p>
          <p class="coming-soon">Coming soon! 🚀</p>
        </div>
      </div>
      
      <div v-else-if="activeTab === 'visualization'" class="tab-pane active">
        <div class="content-placeholder">
          <h2>Training Visualizations</h2>
          <p>Visualize training metrics and performance.</p>
          <p class="coming-soon">Coming soon! 🚀</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import TrainingDashboard from '../components/training/TrainingDashboard.vue'

const activeTab = ref('dashboard')
</script>

<style scoped>
.training-monitor {
  max-width: 1200px;
  margin: 0 auto;
}

.monitor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--spacing-lg);
}

.monitor-header h1 {
  margin: 0;
  font-size: 1.75rem;
}

.monitor-tabs {
  display: flex;
  border-bottom: 1px solid var(--color-border);
  margin-bottom: var(--spacing-lg);
}

.tab-button {
  padding: var(--spacing-sm) var(--spacing-md);
  background: none;
  border: none;
  font-size: 1rem;
  cursor: pointer;
  position: relative;
  color: var(--color-text-light);
  transition: color 0.3s ease;
}

.tab-button:hover {
  color: var(--color-primary);
}

.tab-button.active {
  color: var(--color-primary);
  font-weight: 500;
}

.tab-button.active::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: var(--color-primary);
}

.tab-content {
  margin-top: var(--spacing-lg);
}

.tab-pane {
  display: none;
}

.tab-pane.active {
  display: block;
}

.content-placeholder {
  text-align: center;
  padding: var(--spacing-xl) 0;
  background-color: #f9f9f9;
  border-radius: var(--border-radius);
}

.content-placeholder h2 {
  margin-top: 0;
}

.coming-soon {
  margin-top: var(--spacing-lg);
  font-weight: 600;
  color: var(--color-primary);
}

@media (max-width: 768px) {
  .monitor-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .header-actions {
    margin-top: var(--spacing-md);
  }
  
  .monitor-tabs {
    overflow-x: auto;
    width: 100%;
  }
}
</style>