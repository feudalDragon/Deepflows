<template>
  <div class="dashboard">
    <el-row :gutter="20">
      <el-col :span="6">
        <MetricCard title="Current Epoch" :value="currentEpoch" />
      </el-col>
      <el-col :span="6">
        <MetricCard title="Current Batch" :value="currentBatch" />
      </el-col>
      <el-col :span="6">
        <MetricCard title="Current Loss" :value="currentLoss.toFixed(4)" />
      </el-col>
      <el-col :span="6">
        <MetricCard title="Current Accuracy" :value="currentAccuracy.toFixed(2) + '%'" />
      </el-col>
    </el-row>

    <el-row :gutter="20" class="chart-row">
      <el-col :span="16">
        <el-card shadow="hover">
          <template #header>Loss & Accuracy</template>
          <EChartsWrapper :options="lossAccOptions" height="400px" />
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card shadow="hover">
          <template #header>Resource Usage</template>
          <EChartsWrapper :options="resourceOptions" height="400px" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useTrainingStore } from '@/store/training'
import { storeToRefs } from 'pinia'
import EChartsWrapper from '@/components/EChartsWrapper.vue'
import MetricCard from './components/MetricCard.vue'

const trainingStore = useTrainingStore()
const { metrics, currentEpoch, currentBatch, currentLoss, currentAccuracy, systemResources } = storeToRefs(trainingStore)

onMounted(() => {
  trainingStore.initWebSocket()
})

const lossAccOptions = computed(() => {
  const epochs = metrics.value.map(m => `E${m.epoch}-B${m.batch}`)
  const lossData = metrics.value.map(m => m.loss)
  const accData = metrics.value.map(m => m.accuracy / 100) // Normalize to 0-1 for chart if needed, or keep as is. Let's use 0-1 for dual axis.

  return {
    tooltip: { trigger: 'axis' },
    legend: { data: ['Loss', 'Accuracy'] },
    xAxis: { type: 'category', data: epochs },
    yAxis: [
      { type: 'value', name: 'Loss' },
      { type: 'value', name: 'Accuracy', max: 1 }
    ],
    series: [
      { name: 'Loss', type: 'line', data: lossData, smooth: true },
      { name: 'Accuracy', type: 'line', yAxisIndex: 1, data: accData, smooth: true }
    ]
  }
})

// Real resource usage from backend
const resourceOptions = computed(() => ({
  tooltip: { trigger: 'axis' },
  xAxis: { type: 'category', data: ['GPU', 'CPU', 'RAM'] },
  yAxis: { type: 'value', max: 100 },
  series: [
    { 
      type: 'bar', 
      data: [systemResources.value.gpu, systemResources.value.cpu, systemResources.value.ram], 
      itemStyle: { color: '#409eff' },
      label: { show: true, position: 'top', formatter: '{c}%' }
    }
  ]
}))
</script>

<style lang="scss" scoped>
.dashboard {
  .chart-row {
    margin-top: 20px;
  }
}
</style>
