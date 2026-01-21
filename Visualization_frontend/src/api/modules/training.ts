import request from '@/utils/request'

export interface TrainConfig {
  lr: number
  batch_size: number
  epochs: number
  optimizer: string
  dataset: string
  custom_model_config?: any[] | null
}

export const startTraining = (data: TrainConfig) => {
  return request.post('/train/start', data)
}

export const stopTraining = () => {
  return request.post('/train/stop')
}

export const getModels = () => {
  return request.get('/models')
}
