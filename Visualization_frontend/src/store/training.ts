import { defineStore } from 'pinia'
import { ref, reactive } from 'vue'
import { startTraining as startTrainingApi, stopTraining as stopTrainingApi, type TrainConfig } from '@/api/modules/training'
import { WebSocketClient } from '@/utils/websocket'
import { ElMessage } from 'element-plus'
import { log } from 'console'

export interface TrainingMetric {
  epoch: number
  batch: number
  loss: number
  accuracy: number
}

export const useTrainingStore = defineStore('training', () => {
  const isTraining = ref(false)
  const metrics = ref<TrainingMetric[]>([])
  const currentEpoch = ref(0)
  const currentBatch = ref(0)
  const currentLoss = ref(0)
  const currentAccuracy = ref(0)
  const systemResources = ref({ cpu: 0, ram: 0, gpu: 0 })
  const wsClient = ref<WebSocketClient | null>(null)

  const initWebSocket = () => {
    if (wsClient.value) return

    // Use current host (proxy) for WS connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = import.meta.env.VITE_WS_URL || `${protocol}//${window.location.host}/ws`
    console.log('WS URL:', wsUrl);
    
    wsClient.value = new WebSocketClient(wsUrl)
    wsClient.value.connect()

    wsClient.value.onMessage((msg: any) => {
      handleMessage(msg)
    })
  }

  const handleMessage = (msg: any) => {
    if (msg.type === 'metrics') {
      const data = msg.data
      metrics.value.push(data)
      // Keep only last 100 metrics to avoid memory issues if running long
      if (metrics.value.length > 500) {
        metrics.value.shift()
      }
      
      currentEpoch.value = data.epoch
      currentBatch.value = data.batch
      currentLoss.value = data.loss
      currentAccuracy.value = data.accuracy
    } else if (msg.type === 'status') {
        if (msg.data === 'stopped') {
            isTraining.value = false
            ElMessage.info('Training stopped by server.')
        }
    } else if (msg.type === 'resources') {
        systemResources.value = msg.data
    } else if (msg.type === 'error') {
        ElMessage.error(`Server Error: ${msg.data}`)
        isTraining.value = false
    }
  }

  const startTraining = async (config: TrainConfig) => {
    try {
      await startTrainingApi(config)
      isTraining.value = true
      metrics.value = [] // Reset metrics on new start
      ElMessage.success('Training started')
    } catch (error) {
      console.error(error)
      isTraining.value = false
    }
  }

  const stopTraining = async () => {
    try {
      await stopTrainingApi()
      // isTraining.value will be updated via WS status message or manually here
      isTraining.value = false 
    } catch (error) {
      console.error(error)
    }
  }

  return {
    isTraining,
    metrics,
    currentEpoch,
    currentBatch,
    currentLoss,
    currentAccuracy,
    systemResources,
    startTraining,
    stopTraining,
    initWebSocket
  }
})
