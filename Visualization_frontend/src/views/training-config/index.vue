<template>
  <div class="training-config">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>Training Configuration</span>
        </div>
      </template>
      <el-form :model="form" label-width="120px">
        <el-form-item label="Model Selection">
          <el-select v-model="form.selectedModelId" placeholder="Select a model" @change="onModelChange">
            <el-option label="Default MNIST_CNN" value="default" />
            <el-option
              v-for="(model, index) in savedModels"
              :key="index"
              :label="model.name"
              :value="index"
            />
          </el-select>
        </el-form-item>
        <el-form-item label="Model Name" v-if="form.selectedModelId === 'default'">
          <el-input v-model="form.name" />
        </el-form-item>
        <el-form-item label="Dataset">
          <el-select v-model="form.dataset" placeholder="Select dataset">
            <el-option label="MNIST" value="mnist" />
            <el-option label="CIFAR-10" value="cifar10" />
            <el-option label="ImageNet" value="imagenet" />
          </el-select>
        </el-form-item>
        <el-row>
          <el-col :span="12">
            <el-form-item label="Batch Size">
              <el-input-number v-model="form.batchSize" :min="1" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item label="Epochs">
              <el-input-number v-model="form.epochs" :min="1" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item label="Learning Rate">
          <el-slider v-model="form.learningRate" :min="0.0001" :max="0.1" :step="0.0001" show-input />
        </el-form-item>
        <el-form-item label="Optimizer">
          <el-radio-group v-model="form.optimizer">
            <el-radio label="SGD">SGD</el-radio>
            <el-radio label="Adam">Adam</el-radio>
            <el-radio label="Adagrad">Adagrad</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="onSubmit" :loading="isTraining" :disabled="isTraining">Start Training</el-button>
          <el-button @click="onStop" type="danger" :disabled="!isTraining">Stop Training</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref, onMounted } from 'vue'
import { useTrainingStore } from '@/store/training'
import { storeToRefs } from 'pinia'
import { useRouter } from 'vue-router'

const router = useRouter()
const trainingStore = useTrainingStore()
const { isTraining } = storeToRefs(trainingStore)

const savedModels = ref<any[]>([])

const form = reactive({
  selectedModelId: 'default',
  name: 'My Model v1',
  dataset: 'mnist',
  batchSize: 64,
  epochs: 5,
  learningRate: 0.001,
  optimizer: 'Adam'
})

onMounted(() => {
  const savedModelsStr = localStorage.getItem('saved_models')
  if (savedModelsStr) {
    savedModels.value = JSON.parse(savedModelsStr)
    // Auto-select the latest saved model if coming from Model Builder
    if (savedModels.value.length > 0) {
        // Simple logic: select the last one if it was just added
        // A better way would be passing a query param
        form.selectedModelId = savedModels.value.length - 1
    }
  }
})

const onModelChange = (val: string | number) => {
    if (val === 'default') {
        form.name = 'My Model v1'
    } else {
        // form.name could be read-only or hidden for custom models
    }
}

const onSubmit = async () => {
  let customConfig = null
  
  if (form.selectedModelId !== 'default') {
      const modelIndex = Number(form.selectedModelId)
      if (savedModels.value[modelIndex]) {
          customConfig = savedModels.value[modelIndex].config
      }
  }

  await trainingStore.startTraining({
    lr: form.learningRate,
    batch_size: form.batchSize,
    epochs: form.epochs,
    optimizer: form.optimizer,
    dataset: form.dataset,
    custom_model_config: customConfig
  })
  if (isTraining.value) {
    router.push('/dashboard')
  }
}

const onStop = async () => {
  await trainingStore.stopTraining()
}
</script>

<style lang="scss" scoped>
.training-config {
  max-width: 800px;
  margin: 0 auto;
}
</style>
