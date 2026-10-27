<template>
  <div class="page">
    <h2 class="page-title">语义检索</h2>
    <p class="page-desc">
      BGE-M3 向量召回 + 文件名/标题匹配：既能按意思找，也能直接输入文档名
    </p>

    <el-card shadow="never" class="search-card">
      <div class="search-bar">
        <el-input
          v-model="query"
          placeholder="输入自然语言问题或关键词，例如：这个项目的部署方式是？"
          size="large"
          clearable
          @keyup.enter="doSearch"
        />
        <el-button type="primary" size="large" :icon="Search" :loading="loading" @click="doSearch">
          检索
        </el-button>
      </div>
      <div class="options">
        <span class="opt-label">返回条数</span>
        <el-input-number v-model="topK" :min="1" :max="30" size="small" />
        <el-checkbox v-model="groupByDoc">按文档聚合</el-checkbox>
        <el-checkbox v-model="useNameMatch">
          <el-tooltip
            placement="top"
            content="同时用文件名/标题匹配，输入文档名也能搜到"
          >
            <span class="opt-name">文件名匹配</span>
          </el-tooltip>
        </el-checkbox>
        <span class="result-count" v-if="searched">
          命中 {{ result.total_chunks }} 个片段 / {{ result.documents?.length || 0 }} 篇文档
        </span>
      </div>
    </el-card>

    <!-- 名称通道命中：告诉用户"你搜的名字确实对上了哪篇文档" -->
    <el-alert
      v-if="searched && nameMatches.length"
      type="success"
      :closable="false"
      show-icon
      class="name-hit"
    >
      <template #title>
        <span class="name-hit-title">文件名命中 {{ nameMatches.length }} 篇：</span>
        <el-tag
          v-for="m in nameMatches"
          :key="m.document_id"
          size="small"
          type="success"
          effect="plain"
          class="name-hit-tag"
        >
          {{ m.field === 'title' ? '标题' : '文件' }} · {{ m.text }}
          <em>{{ m.score.toFixed(2) }}</em>
        </el-tag>
      </template>
    </el-alert>

    <!-- 文档聚合视图 -->
    <template v-if="groupByDoc && result.documents?.length">
      <el-card
        v-for="doc in result.documents"
        :key="doc.document_id"
        shadow="hover"
        class="doc-card"
      >
        <div class="doc-head">
          <div class="doc-title-wrap">
            <div class="doc-title">
              <el-icon><Document /></el-icon>
              {{ doc.title || doc.filename }}
            </div>
            <div v-if="showFilename(doc)" class="doc-file">{{ doc.filename }}</div>
          </div>
          <div class="doc-meta">
            <el-tag size="small">{{ doc.file_type }}</el-tag>
            <el-tag
              v-if="doc.match_type && doc.match_type !== 'vector'"
              size="small"
              type="warning"
              effect="plain"
            >文件名命中 {{ (doc.name_score || 0).toFixed(2) }}</el-tag>
            <el-tag size="small" type="success">相似度 {{ doc.best_score.toFixed(3) }}</el-tag>
            <el-tag size="small" type="info">{{ doc.hit_count }} 处命中</el-tag>
          </div>
        </div>

        <p v-if="doc.summary" class="doc-summary">{{ doc.summary }}</p>

        <div class="chunk" v-for="c in doc.matched_chunks" :key="c.chunk_index">
          <span class="chunk-tag">片段 {{ c.chunk_index + 1 }}</span>
          <span class="chunk-text">{{ c.preview }}</span>
        </div>

        <el-button text type="primary" @click="openDoc(doc.document_id)">查看全文</el-button>
      </el-card>
    </template>

    <!-- 片段视图 -->
    <template v-else-if="result.chunks?.length">
      <el-card v-for="c in result.chunks" :key="c.milvus_pk" shadow="hover" class="doc-card">
        <div class="doc-head">
          <div class="doc-title-wrap">
            <div class="doc-title">{{ c.title || c.filename }}</div>
            <div v-if="showFilename(c)" class="doc-file">{{ c.filename }}</div>
          </div>
          <div class="doc-meta">
            <el-tag
              v-if="c.match_type && c.match_type !== 'vector'"
              size="small"
              type="warning"
              effect="plain"
            >文件名命中</el-tag>
            <el-tag size="small" type="success">{{ c.score.toFixed(3) }}</el-tag>
          </div>
        </div>
        <p class="chunk-content">{{ c.content }}</p>
      </el-card>
    </template>

    <el-empty
      v-else-if="searched && !loading"
      description="没有找到相关内容，试试换个说法，或先上传文档"
    />

    <el-drawer v-model="drawer" :title="drawerTitle" size="46%">
      <div class="doc-content">{{ drawerContent }}</div>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { Document, Search } from '@element-plus/icons-vue'
import { getDocumentContent, search } from '../api'

const query = ref('')
const topK = ref(5)
const groupByDoc = ref(true)
const useNameMatch = ref(true)
const loading = ref(false)
const searched = ref(false)
const result = ref({ total_chunks: 0, chunks: [], documents: [], name_matches: [] })

const nameMatches = computed(() => result.value.name_matches || [])

const drawer = ref(false)
const drawerTitle = ref('')
const drawerContent = ref('')

const UUID_NAME_RE = /^[0-9a-f]{32}\.[a-z0-9]+$/i

/** 文件名与标题不同且不是 UUID 占位名时才单独显示 */
function showFilename(row) {
  const name = row.filename || ''
  if (!name || UUID_NAME_RE.test(name)) return false
  return name.replace(/\.[^.]+$/, '') !== (row.title || '')
}

async function doSearch() {
  const q = query.value.trim()
  if (!q) return

  loading.value = true
  searched.value = true
  try {
    result.value = await search({
      query: q,
      top_k: topK.value,
      group_by_document: groupByDoc.value,
      use_name_match: useNameMatch.value
    })
  } catch {
    result.value = { total_chunks: 0, chunks: [], documents: [], name_matches: [] }
  } finally {
    loading.value = false
  }
}

async function openDoc(documentId) {
  try {
    const data = await getDocumentContent(documentId)
    drawerTitle.value = data.title || `文档 ${documentId}`
    drawerContent.value = data.content
    drawer.value = true
  } catch {
    /* 已提示 */
  }
}
</script>

<style scoped>
.search-card { border-radius: 10px; margin-bottom: 18px; }
.search-bar { display: flex; gap: 10px; }
.options {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 12px;
  font-size: 13px;
}
.opt-label { color: #6b7280; }
.opt-name { border-bottom: 1px dashed #9ca3af; cursor: help; }
.result-count { margin-left: auto; color: #9ca3af; font-size: 12px; }

.name-hit {
  margin-bottom: 14px;
  border-radius: 8px;
}
.name-hit :deep(.el-alert__title) {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 13px;
}
.name-hit-title { margin-right: 2px; }
.name-hit-tag { margin: 2px 4px 2px 0; }
.name-hit-tag em { font-style: normal; opacity: 0.6; margin-left: 4px; }

.doc-card { border-radius: 10px; margin-bottom: 14px; }
.doc-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.doc-title-wrap { min-width: 0; }
.doc-title {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 600;
  font-size: 15px;
  color: #111827;
}
.doc-file {
  color: #9ca3af;
  font-size: 12px;
  margin-top: 2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-meta { display: flex; gap: 6px; flex-shrink: 0; flex-wrap: wrap; justify-content: flex-end; }
.doc-summary {
  color: #6b7280;
  font-size: 13px;
  line-height: 1.7;
  margin-bottom: 10px;
  padding-left: 10px;
  border-left: 3px solid #e5e7eb;
}
.chunk {
  background: #f9fafb;
  border-radius: 8px;
  padding: 8px 10px;
  margin-bottom: 6px;
  font-size: 13px;
  line-height: 1.7;
  color: #4b5563;
}
.chunk-tag {
  color: #3b82f6;
  font-size: 12px;
  margin-right: 8px;
  font-weight: 600;
}
.chunk-content { font-size: 14px; line-height: 1.8; color: #374151; }
.doc-content { white-space: pre-wrap; line-height: 1.9; font-size: 14px; }
</style>
