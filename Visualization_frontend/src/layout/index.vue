<template>
  <div class="app-wrapper">
    <div class="sidebar-container" :class="{ 'collapsed': !appStore.sidebarOpen }">
      <el-menu
        :default-active="route.path"
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409eff"
        :collapse="!appStore.sidebarOpen"
        router
      >
        <div class="logo">
          <svg v-if="appStore.sidebarOpen" width="180" height="50" viewBox="0 0 180 50" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#409eff;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#36cfc9;stop-opacity:1" />
              </linearGradient>
            </defs>
            <!-- 模拟手写/流动风格的文字路径 (DeepFlows) -->
            <!-- 新设计：双重流线 (Dual Flow)，象征神经网络的层级与数据流动 -->
            <path d="M10,22 C20,12 30,32 45,18" fill="none" stroke="url(#flowGradient)" stroke-width="3" stroke-linecap="round" />
            <path d="M10,34 C20,24 30,44 45,30" fill="none" stroke="url(#flowGradient)" stroke-width="2" stroke-linecap="round" opacity="0.6" />
            
            <!-- 使用更清晰的手写体，平衡艺术感与可读性 -->
            <text x="55" y="34" font-family="'Segoe Script', 'Lucida Handwriting', 'Gabriola', cursive" font-size="20" font-weight="bold" fill="url(#flowGradient)">DeepFlows</text>
            <!-- 底部装饰流线 -->
            <path d="M10,42 Q80,55 150,42" fill="none" stroke="url(#flowGradient)" stroke-width="2" stroke-opacity="0.5" />
          </svg>
          <span v-else>DF</span>
        </div>
        <el-menu-item index="/dashboard">
          <el-icon><Odometer /></el-icon>
          <template #title>Dashboard</template>
        </el-menu-item>
        <el-menu-item index="/model-builder">
          <el-icon><Connection /></el-icon>
          <template #title>Model Builder</template>
        </el-menu-item>
        <el-menu-item index="/training-config">
          <el-icon><Setting /></el-icon>
          <template #title>Training Config</template>
        </el-menu-item>
      </el-menu>
    </div>
    <div class="main-container">
      <div class="header">
        <el-icon class="hamburger" @click="appStore.toggleSidebar">
          <Fold v-if="appStore.sidebarOpen" />
          <Expand v-else />
        </el-icon>
        <div class="breadcrumb">
          {{ route.meta.title }}
        </div>
        <div class="header-right">
          <el-icon class="fullscreen-btn" @click="toggleFullScreen">
            <FullScreen />
          </el-icon>
        </div>
      </div>
      <div class="app-main">
        <router-view />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useAppStore } from '@/store/app'
import { useRoute } from 'vue-router'
import { Odometer, Connection, Setting, Fold, Expand, FullScreen } from '@element-plus/icons-vue'
import { useFullscreen } from '@vueuse/core'

const appStore = useAppStore()
const route = useRoute()
const { toggle: toggleFullScreen } = useFullscreen()
</script>

<style lang="scss" scoped>
.app-wrapper {
  display: flex;
  height: 100vh;
  width: 100%;
}

.sidebar-container {
  width: $menu-width;
  height: 100%;
  background-color: $menu-bg;
  transition: width 0.3s;
  overflow: hidden;

  &.collapsed {
    width: 64px;
  }

  .logo {
    height: 50px;
    line-height: 50px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: 20px;
  }
  
  :deep(.el-menu) {
    border-right: none;
  }
}

.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;

  .header {
    height: $header-height;
    border-bottom: 1px solid #dcdfe6;
    display: flex;
    align-items: center;
    padding: 0 20px;

    .hamburger {
      font-size: 20px;
      cursor: pointer;
      margin-right: 20px;
    }

    .header-right {
      margin-left: auto;
      
      .fullscreen-btn {
        font-size: 20px;
        cursor: pointer;
        &:hover {
          color: #409eff;
        }
      }
    }
  }

  .app-main {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background-color: #f0f2f5;
  }
}
</style>
