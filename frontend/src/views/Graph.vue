<template>
  <div class="page">
    <h2 class="page-title">知识图谱</h2>
    <p class="page-desc">
      文档为节点；实线＝纯语义相似，虚线＝名字也对得上（同系列/同名的文档即使正文不同也连起来）
    </p>

    <el-card shadow="never" class="ctrl-card">
      <div class="controls">
        <div class="ctrl">
          <span class="ctrl-label">相似度阈值</span>
          <el-slider
            v-model="minScore"
            :min="0.25"
            :max="0.85"
            :step="0.01"
            style="width: 170px"
            @change="load"
          />
          <span class="ctrl-value">{{ minScore.toFixed(2) }}</span>
        </div>
        <div class="ctrl">
          <span class="ctrl-label">最多节点</span>
          <el-input-number v-model="maxNodes" :min="2" :max="500" size="small" @change="load" />
        </div>
        <div class="ctrl">
          <el-tooltip placement="top" content="文件名高度相似的文档也连线（虚线）">
            <span class="ctrl-label">文件名关联</span>
          </el-tooltip>
          <el-switch v-model="useNameEdges" size="small" @change="load" />
        </div>
        <el-button type="primary" :icon="Refresh" :loading="loading" @click="load">重新计算</el-button>
        <el-tooltip
          v-if="graph.vector_stats"
          placement="top"
          :content="`文档向量按文档缓存：复用了 ${graph.vector_stats.reused} 篇，本次新编码 ${graph.vector_stats.encoded} 篇（${graph.vector_stats.encode_ms}ms）`"
        >
          <span class="vec-stat">
            <el-icon><Coin /></el-icon>
            编码 {{ graph.vector_stats.encoded }} / 复用 {{ graph.vector_stats.reused }}
          </span>
        </el-tooltip>
        <span class="summary">
          {{ graph.count }} 个节点 / {{ graph.edges.length }} 条关联
          <em v-if="graph.name_edge_count">（含文件名关联 {{ graph.name_edge_count }} 条）</em>
        </span>
      </div>
    </el-card>

    <!-- 阈值过高导致一条边都连不出来时，给出可操作的建议 -->
    <el-alert
      v-if="!loading && graph.nodes.length > 1 && graph.edges.length === 0"
      type="info"
      :closable="false"
      class="hint-bar"
      show-icon
    >
      <template #title>
        <span>
          当前阈值 {{ minScore.toFixed(2) }} 下没有文档达到相似度要求
          <template v-if="graph.max_score">（最高相似度 {{ Number(graph.max_score).toFixed(2) }}）</template>
        </span>
        <el-button
          v-if="graph.suggested_score"
          text
          type="primary"
          size="small"
          @click="applySuggested"
        >按建议改用 {{ Number(graph.suggested_score).toFixed(2) }}</el-button>
      </template>
    </el-alert>

    <!-- 图表容器始终存在：v-show/display:none 会让 echarts.init 拿到 0 尺寸画布，
         导致图表只渲染出一小块。空状态改为绝对定位的浮层。 -->
    <el-card shadow="never" class="graph-card">
      <div class="chart-wrap">
        <div ref="chartRef" class="chart"></div>

        <div v-if="!loading && graph.nodes.length === 0" class="empty-overlay">
          <el-empty description="还没有足够的文档，先上传并索引几篇文档吧" />
        </div>

        <div v-if="loading" class="loading-overlay">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>正在计算文档关联…</span>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'
import { Loading, Refresh, Coin } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import { getGraph } from '../api'

const chartRef = ref(null)
const chart = shallowRef(null)
const loading = ref(false)
const minScore = ref(0.45)
const maxNodes = ref(200)
const useNameEdges = ref(true)
const graph = ref({
  nodes: [],
  edges: [],
  count: 0,
  max_score: null,
  suggested_score: null,
  name_edge_count: 0,
  vector_stats: null
})

let resizeObserver = null
let disposed = false

const palette = {
  pdf: '#ef4444',
  docx: '#3b82f6',
  doc: '#3b82f6',
  md: '#10b981',
  markdown: '#10b981',
  txt: '#6b7280',
  csv: '#f59e0b',
  json: '#8b5cf6',
  other: '#9ca3af'
}

const ESC_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ESC_MAP[c])

function shortName(name, max = 14) {
  const text = String(name ?? '未命名')
  return text.length > max ? text.slice(0, max) + '…' : text
}

function categoryList() {
  const types = [...new Set(graph.value.nodes.map((n) => n.category || 'other'))]
  return types.map((t) => ({ name: t, itemStyle: { color: palette[t] || palette.other } }))
}

/** 等 DOM 真正有尺寸后再初始化，避免 0×0 画布 */
async function ensureChart() {
  await nextTick()
  if (disposed || !chartRef.value) return null

  if (!chart.value) {
    chart.value = echarts.init(chartRef.value, null, { renderer: 'canvas' })
    // 开发模式下暴露实例，方便调试与自动化测试（生产构建会被 tree-shake 掉）
    if (import.meta.env.DEV) {
      window.__kmChart = chart.value
    }
  }
  // 容器尺寸可能刚变化（首次显示、窗口缩放、侧栏展开）
  chart.value.resize()
  return chart.value
}

function renderChart() {
  const inst = chart.value
  if (!inst) return

  const categories = categoryList()
  const catIndex = Object.fromEntries(categories.map((c, i) => [c.name, i]))

  const option = {
    animation: true,
    animationDuration: 600,
    // 让 tooltip 始终留在画布内，并强制换行，避免长文本把浮框撑爆
    tooltip: {
      trigger: 'item',
      confine: true,
      appendToBody: false,
      enterable: false,
      backgroundColor: 'rgba(17, 24, 39, 0.94)',
      borderWidth: 0,
      padding: [8, 10],
      textStyle: { color: '#f9fafb', fontSize: 12, lineHeight: 18 },
      extraCssText: 'max-width: 280px; white-space: normal; word-break: break-word; border-radius: 8px;',
      formatter: (p) => {
        if (p.dataType === 'edge') {
          const nameScore = Number(p.data.name_score ?? 0)
          const value = Number(p.data.value ?? 0)
          if (p.data.relation === 'filename') {
            return `<div>文件名关联</div><div style="margin-top:2px;color:#d1d5db;font-size:11px;">名称相似度 ${nameScore.toFixed(3)}</div>`
          }
          const extra = nameScore
            ? `<div style="margin-top:2px;color:#9ca3af;font-size:11px;">文件名相似度 ${nameScore.toFixed(3)}</div>`
            : ''
          return `<div>语义相似度 <b>${value.toFixed(3)}</b></div>${extra}`
        }
        const name = esc(p.data.fullName || p.data.name)
        const type = esc(p.data.categoryName || 'other')
        const filename =
          p.data.filename && p.data.filename !== p.data.fullName
            ? `<div style="margin-top:2px;color:#9ca3af;font-size:11px;word-break:break-all;">文件：${esc(
                p.data.filename
              )}</div>`
            : ''
        const summary = p.data.summary
          ? `<div style="margin-top:6px;color:#d1d5db;font-size:11px;line-height:1.6;">${esc(
              p.data.summary.slice(0, 90)
            )}${p.data.summary.length > 90 ? '…' : ''}</div>`
          : ''
        const chunks = p.data.chunkCount
          ? `<div style="margin-top:4px;color:#9ca3af;font-size:11px;">${p.data.chunkCount} 个分块</div>`
          : ''
        return `<div style="font-weight:600;font-size:13px;">${name}</div>
                <div style="margin-top:2px;color:#9ca3af;font-size:11px;">类型：${type}</div>
                ${filename}${chunks}${summary}`
      }
    },
    legend: [
      {
        data: categories.map((c) => c.name),
        bottom: 0,
        type: 'scroll',
        textStyle: { fontSize: 11, color: '#6b7280' },
        itemWidth: 10,
        itemHeight: 10
      }
    ],
    series: [
      {
        type: 'graph',
        layout: 'force',
        // 给图例留出底部空间，否则节点会被挤到图例上
        left: 24,
        right: 24,
        top: 24,
        bottom: 40,
        // 环形初始布局比随机分布更稳定，节点不易被甩到视野外
        force: {
          initLayout: 'circular',
          repulsion: 700,
          edgeLength: [170, 300],
          gravity: 0.05,
          friction: 0.15,
          layoutAnimation: true
        },
        roam: true,
        draggable: true,
        scaleLimit: { min: 0.3, max: 4 },
        zoom: 0.95,
        symbolSize: 30,
        // 中文标题普遍较长，节点挨在一起时标签会互相压住，交给 echarts 自动隐藏
        labelLayout: { hideOverlap: true },
        label: {
          show: true,
          position: 'right',
          distance: 6,
          fontSize: 11,
          color: '#374151',
          formatter: (p) => shortName(p.data.fullName || p.data.name, 12)
        },
        edgeSymbol: ['none', 'none'],
        edgeLabel: { show: false },
        lineStyle: { color: 'source', curveness: 0.1, opacity: 0.5, width: 1.2 },
        emphasis: {
          focus: 'adjacency',
          lineStyle: { width: 3, opacity: 0.9 },
          label: { fontSize: 12, fontWeight: 600 }
        },
        blur: { itemStyle: { opacity: 0.35 }, lineStyle: { opacity: 0.15 } },
        categories,
        data: graph.value.nodes.map((n) => ({
          id: String(n.id),
          name: n.name,
          fullName: n.name,
          filename: n.filename || '',
          category: catIndex[n.category || 'other'] ?? 0,
          categoryName: n.category || 'other',
          summary: n.summary || '',
          chunkCount: n.value || 0,
          symbolSize: 26 + Math.min(18, (n.value || 1) * 1.2)
        })),
        links: graph.value.edges.map((e) => ({
          source: String(e.source),
          target: String(e.target),
          value: e.value,
          relation: e.relation || 'semantic',
          name_score: e.name_score || 0,
          lineStyle: e.lineStyle
        }))
      }
    ]
  }

  inst.setOption(option, true)
  // setOption 后容器可能刚完成布局（例如从空状态切到有数据），再补一次 resize
  requestAnimationFrame(() => {
    if (!disposed && chart.value) chart.value.resize()
  })
}

async function load() {
  loading.value = true
  try {
    graph.value = await getGraph({
      min_score: minScore.value,
      max_nodes: maxNodes.value,
      // 关闭开关时把阈值设为 0 以上但不等于 0 的"关闭"语义：后端约定 0 = 不生成
      name_min_score: useNameEdges.value ? 0.6 : 0
    })
    await ensureChart()
    renderChart()
  } catch {
    /* 错误已由 api 拦截器提示 */
  } finally {
    loading.value = false
  }
}

/** 采用后端给出的建议阈值重新计算 */
function applySuggested() {
  const suggested = Number(graph.value.suggested_score)
  if (!Number.isFinite(suggested)) return
  minScore.value = Math.min(0.85, Math.max(0.25, suggested))
  load()
}

onMounted(() => {
  if (chartRef.value && typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => {
      if (!disposed && chart.value) chart.value.resize()
    })
    resizeObserver.observe(chartRef.value)
  }
  window.addEventListener('resize', onWindowResize)
  load()
})

function onWindowResize() {
  if (!disposed && chart.value) chart.value.resize()
}

onBeforeUnmount(() => {
  disposed = true
  window.removeEventListener('resize', onWindowResize)
  if (resizeObserver) {
    resizeObserver.disconnect()
    resizeObserver = null
  }
  chart.value?.dispose()
  chart.value = null
})
</script>

<style scoped>
.ctrl-card { border-radius: 10px; margin-bottom: 16px; }
.controls { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
.ctrl { display: flex; align-items: center; gap: 8px; }
.ctrl-label { font-size: 13px; color: #6b7280; }
.ctrl-value { font-size: 13px; color: #3b82f6; font-weight: 600; width: 36px; }
.summary { margin-left: auto; font-size: 12px; color: #9ca3af; }
.summary em { font-style: normal; color: #b0b7c3; }
.vec-stat {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #6b7280;
  cursor: help;
}

.hint-bar {
  margin-bottom: 14px;
  border-radius: 8px;
}
.hint-bar :deep(.el-alert__title) {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  font-size: 13px;
}

.graph-card { border-radius: 10px; }
.graph-card :deep(.el-card__body) { padding: 0; }

.chart-wrap {
  position: relative;
  width: 100%;
  height: 640px;
}

.chart {
  width: 100%;
  height: 100%;
}

.empty-overlay,
.loading-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: #6b7280;
  font-size: 13px;
  background: #fff;
  pointer-events: none;
}

.loading-overlay {
  background: rgba(255, 255, 255, 0.72);
}
</style>
