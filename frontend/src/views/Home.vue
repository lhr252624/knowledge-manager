<template>
  <div class="page">
    <h2 class="page-title">概览</h2>
    <p class="page-desc">多 Agent 驱动的本地知识库：上传文档即可检索与问答</p>

    <el-row :gutter="16" class="stat-row">
      <el-col :span="6" v-for="card in cards" :key="card.label">
        <el-card shadow="hover" class="stat-card">
          <div class="stat-label">{{ card.label }}</div>
          <div class="stat-value">{{ card.value }}</div>
          <div class="stat-hint">{{ card.hint }}</div>
        </el-card>
      </el-col>
    </el-row>

    <el-card class="block" shadow="never">
      <template #header>
        <div class="card-header">
          <span>Agent 分工</span>
          <el-button text :icon="Refresh" @click="load" :loading="loading">刷新</el-button>
        </div>
      </template>
      <el-table :data="agents" size="small">
        <el-table-column prop="name" label="Agent" width="180" />
        <el-table-column prop="description" label="职责" />
        <el-table-column label="状态" width="110">
          <template #default>
            <el-tag type="success" size="small">就绪</el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card class="block" shadow="never">
      <template #header>快捷入口</template>
      <div class="actions">
        <el-button type="primary" :icon="ChatDotRound" @click="$router.push('/chat')">
          开始问答
        </el-button>
        <el-button :icon="Document" @click="$router.push('/documents')">
          上传文档
        </el-button>
        <el-button :icon="Search" @click="$router.push('/search')">
          语义检索
        </el-button>
        <el-button :icon="Share" @click="$router.push('/graph')">
          知识图谱
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { ChatDotRound, Document, Refresh, Search, Share } from '@element-plus/icons-vue'
import { getAgents, getStats } from '../api'

const stats = ref({})
const agents = ref([])
const loading = ref(false)

const cards = computed(() => [
  {
    label: '文档总数',
    value: stats.value.total_documents ?? '-',
    hint: `就绪 ${stats.value.ready_documents ?? 0} / 处理中 ${stats.value.processing_documents ?? 0}`
  },
  {
    label: '文本分块',
    value: stats.value.total_chunks ?? '-',
    hint: '检索的最小语义单元'
  },
  {
    label: '向量条数',
    value: stats.value.total_vectors ?? '-',
    hint: 'Milvus 中已索引向量'
  },
  {
    label: '总字符数',
    value: (stats.value.total_characters ?? 0).toLocaleString(),
    hint: '知识库正文规模'
  }
])

async function load() {
  loading.value = true
  try {
    const [s, a] = await Promise.all([getStats(), getAgents()])
    stats.value = s
    agents.value = a.agents || []
  } catch (e) {
    /* 拦截器已提示 */
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.stat-row { margin-bottom: 18px; }
.stat-card { border-radius: 10px; }
.stat-label { color: #6b7280; font-size: 13px; }
.stat-value { font-size: 30px; font-weight: 650; margin: 6px 0 2px; }
.stat-hint { color: #9ca3af; font-size: 12px; }
.block { margin-bottom: 18px; border-radius: 10px; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
.actions { display: flex; gap: 10px; flex-wrap: wrap; }
</style>
