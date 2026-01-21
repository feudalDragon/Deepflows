import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'
import Layout from '@/layout/index.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: Layout,
    redirect: '/dashboard',
    children: [
      {
        path: 'dashboard',
        name: 'Dashboard',
        component: () => import('@/views/dashboard/index.vue'),
        meta: { title: 'Dashboard', icon: 'Odometer' }
      },
      {
        path: 'model-builder',
        name: 'ModelBuilder',
        component: () => import('@/views/model-builder/index.vue'),
        meta: { title: 'Model Builder', icon: 'Connection' }
      },
      {
        path: 'training-config',
        name: 'TrainingConfig',
        component: () => import('@/views/training-config/index.vue'),
        meta: { title: 'Training Config', icon: 'Setting' }
      }
    ]
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
