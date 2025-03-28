import { createRouter, createWebHistory } from 'vue-router'
import Dashboard from '../views/Dashboard.vue'
import ModelConfig from '../views/ModelConfig.vue'
import ModelTest from '../views/ModelTest.vue'
import OpenApi from '../views/OpenApi.vue'

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
    }
  ]
})

export default router