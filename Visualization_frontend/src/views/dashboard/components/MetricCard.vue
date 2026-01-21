<template>
  <div class="metric-card">
    <el-card shadow="hover">
      <template #header>
        <div class="card-header">
          <span>{{ title }}</span>
        </div>
      </template>
      <div class="card-value">{{ value }}</div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
  value: string | number
}>()
</script>

<style lang="scss" scoped>
.card-value {
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}
</style>
