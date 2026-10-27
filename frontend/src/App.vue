<template>
  <div id="app">
    <el-container class="layout">
      <el-aside width="220px" class="sidebar">
        <div class="brand">
          <span class="brand-icon">📚</span>
          <div class="brand-text">
            <div class="brand-title">智能知识库</div>
            <div class="brand-sub">Multi-Agent RAG</div>
          </div>
        </div>

        <el-menu :default-active="activeMenu" router class="side-menu">
          <el-menu-item index="/">
            <el-icon><HomeFilled /></el-icon><span>概览</span>
          </el-menu-item>
          <el-menu-item index="/chat">
            <el-icon><ChatDotRound /></el-icon><span>智能问答</span>
          </el-menu-item>
          <el-menu-item index="/search">
            <el-icon><Search /></el-icon><span>语义检索</span>
          </el-menu-item>
          <el-menu-item index="/documents">
            <el-icon><Document /></el-icon><span>文档管理</span>
          </el-menu-item>
          <el-menu-item index="/graph">
            <el-icon><Share /></el-icon><span>知识图谱</span>
          </el-menu-item>
        </el-menu>

        <div class="sidebar-footer">
          <el-tag :type="healthType" size="small" effect="dark">
            {{ healthText }}
          </el-tag>
        </div>
      </el-aside>

      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { getHealthDetail } from './api'

const route = useRoute()
// /chat/12 这类带参数的路径也要让侧边栏保持高亮
const activeMenu = computed(() => {
  const top = route.path.split('/')[1] || ''
  return top ? `/${top}` : '/'
})

const health = ref({ status: 'unknown' })
const healthText = computed(() => {
  if (health.value.status === 'ok') return '服务正常'
  if (health.value.status === 'degraded') return '部分依赖异常'
  return '连接中...'
})
const healthType = computed(() => {
  if (health.value.status === 'ok') return 'success'
  if (health.value.status === 'degraded') return 'warning'
  return 'info'
})

onMounted(async () => {
  try {
    health.value = await getHealthDetail()
  } catch {
    health.value = { status: 'degraded' }
  }
})
</script>

<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

html, body, #app { height: 100%; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
    'Microsoft YaHei', sans-serif;
  background: #f4f6fa;
  color: #1f2937;
}

.layout { height: 100%; }

.sidebar {
  background: #111827;
  display: flex;
  flex-direction: column;
  padding: 0;
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 20px 18px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.brand-icon { font-size: 24px; }
.brand-title { color: #f9fafb; font-size: 15px; font-weight: 600; }
.brand-sub { color: #6b7280; font-size: 11px; letter-spacing: 0.5px; }

.side-menu {
  flex: 1;
  border-right: none;
  background: transparent;
  padding-top: 8px;
}
.side-menu .el-menu-item {
  color: #9ca3af;
  height: 46px;
}
.side-menu .el-menu-item:hover {
  background: rgba(255, 255, 255, 0.06);
  color: #f3f4f6;
}
.side-menu .el-menu-item.is-active {
  background: rgba(59, 130, 246, 0.18);
  color: #60a5fa;
  border-left: 3px solid #3b82f6;
}

.sidebar-footer {
  padding: 14px 18px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.main {
  padding: 24px;
  overflow-y: auto;
}

.page {
  max-width: 1180px;
  margin: 0 auto;
}

.page-title {
  font-size: 22px;
  font-weight: 650;
  margin-bottom: 6px;
}
.page-desc {
  color: #6b7280;
  font-size: 13px;
  margin-bottom: 20px;
}
</style>
