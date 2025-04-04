import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'
import ModelConfig from '../views/ModelConfig.vue'
import ModelTest from '../views/ModelTest.vue'
import OpenApi from '../views/OpenApi.vue'
import ModelTraining from '../views/ModelTraining.vue'
import TrainingMonitor from '../views/TrainingMonitor.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: Dashboard
    },
    {
      path: '/model-config',
      name: 'model-config',
      component: ModelConfig
    },
    {
      path: '/model-test/:id',
      name: 'model-test',
      component: ModelTest,
      props: true
    },
    {
      path: '/openapi',
      name: 'openapi',
      component: OpenApi
    },
    {
      path: '/model-training',
      name: 'model-training',
      component: ModelTraining
    },
    {
      path: '/training-monitor',
      name: 'training-monitor',
      component: TrainingMonitor
    }
  ]
})

export default router