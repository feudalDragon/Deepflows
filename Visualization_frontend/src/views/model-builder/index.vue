<template>
  <div class="model-builder">
    <div class="layer-list">
      <div class="title">Layers</div>
      <div class="layer-item" v-for="layer in layers" :key="layer" draggable="true" @dragstart="onDragStart(layer)">
        {{ layer }}
      </div>
    </div>
    <div class="canvas-area" @drop="onDrop" @dragover.prevent>
      <div v-if="modelLayers.length === 0" class="placeholder">
        Drag layers here to build your model
      </div>
      <div v-else class="built-layers">
        <div v-for="(layer, index) in modelLayers" :key="index" class="built-layer-item">
          <div class="layer-content">
            <span class="layer-name">{{ layer.type }}</span>
            <span v-if="layer.params && Object.keys(layer.params).length" class="layer-params">
              {{ JSON.stringify(layer.params).substring(0, 20) + '...' }}
            </span>
          </div>
          <div class="layer-actions">
            <el-icon class="action-btn edit-btn" @click="editLayer(index)"><Edit /></el-icon>
            <el-icon class="action-btn delete-btn" @click="removeLayer(index)"><Delete /></el-icon>
          </div>
        </div>
      </div>
    </div>
    <div class="config-preview">
      <div class="title">JSON Preview</div>
      <pre>{{ JSON.stringify(modelLayers, null, 2) }}</pre>
      <div class="actions">
        <el-button type="primary" @click="saveModel" :disabled="modelLayers.length === 0">Save Model</el-button>
        <el-button @click="clearModel" :disabled="modelLayers.length === 0">Clear</el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Delete, Edit } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useTrainingStore } from '@/store/training'
import { useRouter } from 'vue-router'

const router = useRouter()
const trainingStore = useTrainingStore()

const layers = ['Conv2d', 'ReLU', 'MaxPool2d', 'Linear', 'Flatten', 'Dropout']
// Update modelLayers type to store objects with params
interface LayerConfig {
  type: string
  params?: Record<string, any>
}
const modelLayers = ref<LayerConfig[]>([])

const onDragStart = (layer: string) => {
  (window as any).draggedLayer = layer
}

const getDefaultParams = (layerType: string) => {
  switch (layerType) {
    case 'Conv2d': return { in_channels: 1, out_channels: 32, kernel_size: 3, padding: 1 }
    case 'MaxPool2d': return { kernel_size: 2, stride: 2 }
    case 'Linear': return { in_features: 128, out_features: 10 }
    case 'Dropout': return { p: 0.5 }
    default: return {}
  }
}

const onDrop = () => {
  const layerType = (window as any).draggedLayer
  if (layerType) {
    modelLayers.value.push({
      type: layerType,
      params: getDefaultParams(layerType)
    })
    ;(window as any).draggedLayer = null
  }
}

const removeLayer = (index: number) => {
  modelLayers.value.splice(index, 1)
}

const editLayer = async (index: number) => {
  const layer = modelLayers.value[index]
  if (Object.keys(layer.params || {}).length === 0) return

  // Simple prompt for now, can be improved with a proper dialog form
  const paramsStr = JSON.stringify(layer.params, null, 2)
  try {
    const { value } = await ElMessageBox.prompt('Edit parameters (JSON format)', `Edit ${layer.type}`, {
      inputValue: paramsStr,
      inputType: 'textarea',
      inputValidator: (val) => {
        try {
          JSON.parse(val)
          return true
        } catch (e) {
          return 'Invalid JSON'
        }
      },
      customClass: 'json-editor-dialog', // Add custom class
    })
    layer.params = JSON.parse(value)
  } catch {
    // Cancelled
  }
}

const clearModel = () => {
  modelLayers.value = []
}

const saveModel = async () => {
  if (modelLayers.value.length === 0) return

  try {
    const { value: modelName } = await ElMessageBox.prompt('Please enter a name for your model', 'Save Model', {
      confirmButtonText: 'Save',
      cancelButtonText: 'Cancel',
      inputPattern: /\S+/,
      inputErrorMessage: 'Model name is required'
    })

    const newModel = {
      name: modelName,
      config: modelLayers.value,
      timestamp: Date.now()
    }

    // Get existing saved models
    const savedModelsStr = localStorage.getItem('saved_models')
    const savedModels = savedModelsStr ? JSON.parse(savedModelsStr) : []
    
    // Add new model
    savedModels.push(newModel)
    
    // Save back to localStorage
    localStorage.setItem('saved_models', JSON.stringify(savedModels))
    
    ElMessage.success(`Model "${modelName}" saved successfully!`)
    router.push('/training-config')
  } catch {
    // Cancelled
  }
}
</script>

<style lang="scss">
// Global style override for the message box
.json-editor-dialog {
  width: 500px;
  
  textarea {
    min-height: 200px !important;
    font-family: monospace;
  }
}
</style>

<style lang="scss" scoped>
.model-builder {
  display: flex;
  height: calc(100vh - 100px);
  gap: 20px;

  .layer-list, .config-preview {
    width: 250px;
    background: white;
    padding: 20px;
    border-radius: 4px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    
    .title {
      font-weight: bold;
      margin-bottom: 15px;
    }
    
    .layer-item {
      padding: 10px;
      margin-bottom: 10px;
      background: #ecf5ff;
      border: 1px solid #d9ecff;
      border-radius: 4px;
      cursor: move;
      &:hover {
        background: #409eff;
        color: white;
      }
    }

    .actions {
      margin-top: auto;
      display: flex;
      gap: 10px;
      padding-top: 20px;
    }
  }

  .canvas-area {
    flex: 1;
    background: white;
    border-radius: 4px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    border: 2px dashed #dcdfe6;
    overflow-y: auto;
    
    .placeholder {
      color: #909399;
      margin-top: 200px;
    }

    .built-layers {
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      
      .built-layer-item {
        width: 240px;
        padding: 10px 15px;
        margin: 10px 0;
        background: #f0f9eb;
        border: 1px solid #e1f3d8;
        border-radius: 4px;
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: center;
        
        &:not(:last-child)::after {
          content: '↓';
          position: absolute;
          bottom: -25px;
          left: 50%;
          transform: translateX(-50%);
          color: #909399;
        }

        .layer-content {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          overflow: hidden;
          
          .layer-name {
            font-weight: bold;
          }
          
          .layer-params {
            font-size: 12px;
            color: #606266;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 150px;
          }
        }

        .layer-actions {
          display: flex;
          gap: 5px;
          opacity: 0;
          transition: opacity 0.2s;
        }

        &:hover .layer-actions {
          opacity: 1;
        }

        .action-btn {
          cursor: pointer;
          font-size: 16px;
          
          &.edit-btn { color: #409eff; }
          &.delete-btn { color: #f56c6c; }
        }
      }
    }
  }
}
</style>
