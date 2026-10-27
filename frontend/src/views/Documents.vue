<template>
  <div class="page">
    <h2 class="page-title">文档管理</h2>
    <p class="page-desc">上传 PDF / Word / Markdown / TXT，后台自动解析、分块并建立向量索引</p>

    <el-card shadow="never" class="upload-card">
      <el-upload
        drag
        multiple
        :show-file-list="false"
        :before-upload="beforeUpload"
        :http-request="customUpload"
      >
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-text">拖拽文件到此处，或 <em>点击选择文件</em></div>
        <div class="upload-tip">
          支持 PDF、DOCX、MD、TXT、CSV、JSON，单文件不超过 50MB
        </div>
      </el-upload>

      <div v-if="uploading.length" class="uploading">
        <div v-for="u in uploading" :key="u.name" class="upload-item">
          <span class="up-name">{{ u.name }}</span>
          <el-progress :percentage="u.percent" :stroke-width="6" style="flex: 1" />
        </div>
      </div>
    </el-card>

    <el-card shadow="never" class="list-card">
      <template #header>
        <div class="list-head">
          <div class="filters">
            <el-input
              v-model="keyword"
              placeholder="搜索文件名/标题"
              clearable
              size="small"
              style="width: 200px"
              @keyup.enter="load()"
              @clear="load()"
            />
            <el-select
              v-model="status"
              placeholder="全部状态"
              clearable
              size="small"
              style="width: 130px"
              @change="load()"
            >
              <el-option label="就绪" value="ready" />
              <el-option label="索引中" value="indexing" />
              <el-option label="失败" value="failed" />
            </el-select>
            <el-button size="small" :icon="Refresh" @click="load()">刷新</el-button>
          </div>

          <div class="list-right">
            <span class="last-updated" v-if="autoRefresh">
              <span class="dot" :class="{ live: hasPending }"></span>
              {{ hasPending ? '处理中，自动刷新' : '自动刷新' }}
              <em v-if="lastUpdated">· {{ lastUpdated }}</em>
            </span>
            <el-switch
              v-model="autoRefresh"
              active-text="自动"
              size="small"
              @change="toggleAuto"
            />
          </div>
        </div>
      </template>

      <el-table :data="items" v-loading="loading" size="small">
        <el-table-column label="文档" min-width="240">
          <template #default="{ row }">
            <div class="doc-name">{{ row.title || row.filename }}</div>
            <!-- 早期版本入库时没保留原始文件名，磁盘名是 32 位 UUID，
                 这类占位名不必展示（标题已经足够识别文档） -->
            <div v-if="showFilename(row)" class="doc-file">{{ row.filename }}</div>
          </template>
        </el-table-column>

        <el-table-column prop="file_type" label="类型" width="80" />
        <el-table-column label="大小" width="90">
          <template #default="{ row }">{{ formatSize(row.file_size) }}</template>
        </el-table-column>
        <el-table-column prop="chunk_count" label="分块" width="70" />
        <el-table-column label="摘要" min-width="230">
          <template #default="{ row }">
            <!-- 摘要是入库时由 ReAct agent 生成的，点击看完整内容与决策轨迹 -->
            <span
              v-if="row.summary"
              class="summary-cell"
              :title="row.summary"
              @click="openSummary(row)"
            >{{ row.summary }}</span>
            <span v-else-if="isPending(row.status)" class="summary-empty">生成中…</span>
            <span v-else class="summary-empty">—</span>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="110">
          <template #default="{ row }">
            <!-- 失败时把后端原因挂在 tooltip 上，否则用户只看到"失败"无从排查 -->
            <el-tooltip
              v-if="row.status === 'failed'"
              placement="top"
              effect="dark"
              :show-after="200"
            >
              <template #content>
                <div class="err-tip">{{ row.error_msg || '索引失败，可点「重建」重试' }}</div>
              </template>
              <el-tag type="danger" size="small" effect="plain" class="clickable">
                失败
              </el-tag>
            </el-tooltip>

            <el-tag
              v-else
              :type="statusType(row.status)"
              size="small"
              :effect="row.status === 'ready' ? 'light' : 'plain'"
            >
              <el-icon v-if="isPending(row.status)" class="is-loading tag-icon"><Loading /></el-icon>
              {{ statusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="上传时间" width="150">
          <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
        </el-table-column>
        <el-table-column label="操作" width="300" fixed="right">
          <template #default="{ row }">
            <el-button text size="small" :icon="View" @click="preview(row)">查看</el-button>
            <el-button text size="small" :icon="Edit" @click="openRename(row)">改名</el-button>
            <el-button
              text
              size="small"
              :icon="RefreshRight"
              :disabled="isPending(row.status)"
              @click="reindex(row)"
            >重建</el-button>
            <el-button text size="small" type="danger" :icon="Delete" @click="remove(row)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>

      <div class="pager">
        <el-pagination
          v-model:current-page="page"
          :page-size="size"
          :total="total"
          layout="total, prev, pager, next"
          @current-change="load()"
        />
      </div>
    </el-card>

    <el-drawer v-model="drawer" :title="drawerTitle" size="50%">
      <div class="doc-content">{{ drawerContent }}</div>
    </el-drawer>

    <!-- 摘要：正文摘要 + 关键词 + agent 的决策轨迹 -->
    <el-drawer v-model="summaryDrawer" :title="summaryTitle" size="46%">
      <div v-loading="summaryLoading" class="summary-panel">
        <div class="sum-block">
          <div class="sum-label">摘要</div>
          <div class="sum-text">{{ summaryData.summary || '（暂无摘要）' }}</div>
        </div>

        <div v-if="summaryData.keywords?.length" class="sum-block">
          <div class="sum-label">关键词</div>
          <div class="sum-tags">
            <el-tag
              v-for="k in summaryData.keywords"
              :key="k"
              size="small"
              effect="plain"
              type="info"
            >{{ k }}</el-tag>
          </div>
        </div>

        <div class="sum-block">
          <div class="sum-label">
            生成过程
            <span class="sum-meta">
              {{ summaryData.mode === 'react' ? 'ReAct 工具循环' : '单次生成' }}
              <template v-if="summaryData.steps"> · {{ summaryData.steps }} 步</template>
              <template v-if="readTimes"> · 读取 {{ readTimes }} 次 / 累计 {{ totalChars }} 字</template>
            </span>
          </div>

          <el-timeline v-if="steps.length" class="sum-timeline">
            <el-timeline-item
              v-for="(s, i) in steps"
              :key="i"
              :type="stepType(s.action)"
              size="normal"
            >
              <div class="step-title">{{ stepText(s) }}</div>
              <div v-if="stepDesc(s)" class="step-desc">{{ stepDesc(s) }}</div>
              <div v-if="s.chars" class="step-chars">累计读取 {{ s.chars }} 字</div>
            </el-timeline-item>
          </el-timeline>

          <el-empty
            v-else-if="!summaryLoading"
            :image-size="56"
            description="短文档走单次生成，没有工具调用"
          />
        </div>
      </div>
    </el-drawer>

    <!-- 改名：文件名参与检索（向量里带了名字前缀），所以改完要重算向量 -->
    <el-dialog v-model="renameVisible" title="修改文件名 / 标题" width="520px">
      <el-form label-width="76px">
        <el-form-item label="文件名">
          <el-input v-model="renameForm.filename" placeholder="例如：作业2-王小明.pdf" />
        </el-form-item>
        <el-form-item label="标题">
          <el-input v-model="renameForm.title" placeholder="留空则用文件名" />
        </el-form-item>
      </el-form>
      <el-alert
        type="info"
        :closable="false"
        show-icon
        title="文件名会参与语义检索与知识图谱，保存后会重算该文档的向量（约几秒）"
      />
      <template #footer>
        <el-button @click="renameVisible = false">取消</el-button>
        <el-button type="primary" :loading="renaming" @click="submitRename">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Delete,
  Edit,
  Loading,
  Refresh,
  RefreshRight,
  UploadFilled,
  View
} from '@element-plus/icons-vue'
import {
  deleteDocument,
  getDocumentContent,
  getSummaryTrace,
  listDocuments,
  reindexDocument,
  updateDocument,
  uploadDocument
} from '../api'

const PENDING_STATES = ['pending', 'parsing', 'indexing']

const items = ref([])
const total = ref(0)
const page = ref(1)
const size = ref(10)
const keyword = ref('')
const status = ref('')
const loading = ref(false)
const uploading = ref([])
const autoRefresh = ref(true)
const lastUpdated = ref('')

let timer = null
let disposed = false

const drawer = ref(false)
const drawerTitle = ref('')
const drawerContent = ref('')

/* ---------------- 摘要抽屉 ---------------- */
const summaryDrawer = ref(false)
const summaryTitle = ref('')
const summaryLoading = ref(false)
const summaryData = ref({ summary: '', keywords: [], mode: '', steps: 0, trace: [] })

const steps = computed(() => summaryData.value.trace || [])
// 读取次数：轨迹里 read_chunks 出现的次数（一次可能读多块）
const readTimes = computed(() => steps.value.filter((s) => s.action === 'read_chunks').length)
const totalChars = computed(() => Math.max(0, ...steps.value.map((s) => s.chars || 0)))

const STEP_LABELS = {
  get_outline: '查看文档结构',
  finish: '提交摘要',
  thought: '思考（未调用工具）',
  reduce_fallback: '预算用尽，用已读内容收尾'
}

function stepText(s) {
  const action = s.action || ''
  if (action === 'read_chunks') {
    const start = Number(s.args?.start ?? 0) + 1
    const end = Number(s.args?.end ?? 0) + 1
    return start === end ? `读取第 ${start} 块` : `读取第 ${start}~${end} 块`
  }
  if (action === 'search_in_document') return `文档内检索「${s.args?.query || ''}」`
  return STEP_LABELS[action] || action
}

function stepDesc(s) {
  if (s.action === 'get_outline') return String(s.observation || '').split('\n')[0]
  if (s.action === 'thought') return s.observation || ''
  return ''
}

function stepType(action) {
  if (action === 'finish') return 'success'
  if (action === 'reduce_fallback') return 'warning'
  if (action === 'thought') return 'info'
  return 'primary'
}

async function openSummary(row) {
  summaryTitle.value = row.title || row.filename
  // 先用列表里的摘要占位，轨迹随后补上，避免抽屉空白一瞬
  summaryData.value = {
    summary: row.summary || '',
    keywords: [],
    mode: '',
    steps: 0,
    trace: []
  }
  summaryDrawer.value = true
  summaryLoading.value = true
  try {
    summaryData.value = await getSummaryTrace(row.id)
  } catch {
    /* 错误提示由 api 拦截器统一处理 */
  } finally {
    summaryLoading.value = false
  }
}

const renameVisible = ref(false)
const renaming = ref(false)
const renameForm = ref({ id: null, filename: '', title: '' })

const statusMap = {
  ready: { text: '就绪', type: 'success' },
  indexing: { text: '索引中', type: 'warning' },
  parsing: { text: '解析中', type: 'warning' },
  pending: { text: '排队中', type: 'info' },
  failed: { text: '失败', type: 'danger' }
}
const statusText = (s) => statusMap[s]?.text || s
const statusType = (s) => statusMap[s]?.type || 'info'
const isPending = (s) => PENDING_STATES.includes(s)

const hasPending = computed(() => items.value.some((i) => isPending(i.status)))

function formatSize(bytes) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let i = 0
  let n = bytes
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024
    i += 1
  }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`
}

function formatTime(iso) {
  if (!iso) return '-'
  return iso.replace('T', ' ').slice(0, 16)
}

const UUID_NAME_RE = /^[0-9a-f]{32}\.[a-z0-9]+$/i

/** 是否值得展示文件名（过滤掉 UUID 占位名与和标题重复的情况） */
function showFilename(row) {
  const name = row.filename || ''
  if (!name) return false
  if (UUID_NAME_RE.test(name)) return false
  return name.replace(/\.[^.]+$/, '') !== (row.title || '')
}

function stamp() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false })
}

/* ---------------- 轮询 ---------------- */
// 用 setTimeout 链而不是 setInterval：避免请求慢时任务堆积，
// 并且能让"有文档处理中"时用更短的间隔。
function scheduleNext() {
  clearTimeout(timer)
  if (disposed || !autoRefresh.value) return
  const delay = hasPending.value ? 2000 : 8000
  timer = setTimeout(async () => {
    await refresh(true)
    scheduleNext()
  }, delay)
}

/**
 * 拉取列表
 * @param {boolean} silent 静默刷新：不显示 loading 遮罩（自动轮询时使用）
 */
async function refresh(silent = false) {
  if (!silent) loading.value = true
  const pendingBefore = items.value.filter((i) => isPending(i.status)).length
  try {
    const data = await listDocuments({
      page: page.value,
      size: size.value,
      keyword: keyword.value || undefined,
      status: status.value || undefined
    })
    items.value = data.items
    total.value = data.total
    lastUpdated.value = stamp()

    // 处理中的文档全部完成时给一次提示，避免用户以为界面没反应
    const pendingAfter = data.items.filter((i) => isPending(i.status)).length
    if (pendingBefore > 0 && pendingAfter === 0) {
      ElMessage.success('文档索引已完成')
    }
  } catch {
    /* 错误提示由 api 拦截器统一处理 */
  } finally {
    if (!silent) loading.value = false
  }
}

/** 手动触发（筛选、翻页、点刷新按钮） */
async function load() {
  await refresh(false)
  scheduleNext()
}

function toggleAuto(val) {
  clearTimeout(timer)
  timer = null
  if (val) {
    refresh(true).then(scheduleNext)
  }
}

/* ---------------- 上传 ---------------- */
function beforeUpload(file) {
  const maxMb = 50
  if (file.size > maxMb * 1024 * 1024) {
    ElMessage.error(`文件超过 ${maxMb}MB`)
    return false
  }
  return true
}

async function customUpload({ file }) {
  const entry = { name: file.name, percent: 0 }
  uploading.value.push(entry)
  try {
    await uploadDocument(file, (p) => {
      entry.percent = p
    })
    ElMessage.success(`${file.name} 已上传，正在后台索引`)
    // 立刻拉一次让新文档出现在列表里，随后靠常驻轮询自动翻状态
    await refresh(true)
    scheduleNext()
  } catch {
    /* 已提示 */
  } finally {
    uploading.value = uploading.value.filter((u) => u !== entry)
  }
}

/* ---------------- 行操作 ---------------- */
async function preview(row) {
  try {
    const data = await getDocumentContent(row.id)
    drawerTitle.value = data.title || row.filename
    drawerContent.value = data.content
    drawer.value = true
  } catch {
    /* 已提示 */
  }
}

async function reindex(row) {
  try {
    await reindexDocument(row.id)
    ElMessage.success('已提交重建任务')
    await load()
  } catch {
    /* 已提示 */
  }
}

/* ---------------- 改名 ---------------- */
function openRename(row) {
  renameForm.value = {
    id: row.id,
    // 占位名（32 位 UUID）不要预填，免得用户以为这就是原名
    filename: UUID_NAME_RE.test(row.filename || '') ? '' : row.filename || '',
    title: row.title || ''
  }
  renameVisible.value = true
}

async function submitRename() {
  const { id, filename, title } = renameForm.value
  if (!filename.trim() && !title.trim()) {
    ElMessage.warning('文件名和标题不能都为空')
    return
  }
  renaming.value = true
  try {
    await updateDocument(id, {
      filename: filename.trim() || undefined,
      title: title.trim() || undefined
    })
    ElMessage.success('已保存，向量已按新名字重算')
    renameVisible.value = false
    await load()
  } catch {
    /* 已提示 */
  } finally {
    renaming.value = false
  }
}

async function remove(row) {
  try {
    await ElMessageBox.confirm(
      `确定删除《${row.title || row.filename}》？向量与文件都会被清除。`,
      '删除确认',
      { type: 'warning' }
    )
  } catch {
    return
  }
  try {
    await deleteDocument(row.id)
    ElMessage.success('已删除')
    await load()
  } catch {
    /* 已提示 */
  }
}

onMounted(() => {
  load()
})

onBeforeUnmount(() => {
  disposed = true
  clearTimeout(timer)
  timer = null
})
</script>

<style scoped>
.upload-card { border-radius: 10px; margin-bottom: 18px; }
.upload-icon { font-size: 46px; color: #c0c4cc; }
.upload-text { color: #606266; font-size: 14px; }
.upload-text em { color: #3b82f6; font-style: normal; }
.upload-tip { color: #9ca3af; font-size: 12px; margin-top: 6px; }

.uploading { margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
.upload-item { display: flex; align-items: center; gap: 12px; }
.up-name {
  width: 200px;
  font-size: 13px;
  color: #4b5563;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.list-card { border-radius: 10px; }
.list-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.filters { display: flex; gap: 8px; align-items: center; }
.list-right { display: flex; align-items: center; gap: 10px; }

.last-updated {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: #9ca3af;
}
.last-updated em { font-style: normal; color: #c0c4cc; }
.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #d1d5db;
}
.dot.live {
  background: #22c55e;
  box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7);
  animation: pulse 1.6s infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6); }
  70% { box-shadow: 0 0 0 6px rgba(34, 197, 94, 0); }
  100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}

.doc-name { font-weight: 550; color: #111827; }
.doc-file { color: #9ca3af; font-size: 12px; }
.tag-icon { margin-right: 3px; vertical-align: -2px; }
.clickable { cursor: help; }
.err-tip {
  max-width: 360px;
  line-height: 1.6;
  word-break: break-word;
  white-space: normal;
}
.pager { display: flex; justify-content: flex-end; margin-top: 12px; }
.doc-content { white-space: pre-wrap; line-height: 1.9; font-size: 14px; }

/* ---------------- 摘要列与摘要抽屉 ---------------- */
.summary-cell {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  font-size: 12.5px;
  line-height: 1.55;
  color: #4b5563;
  cursor: pointer;
}
.summary-cell:hover { color: #3b82f6; }
.summary-empty { color: #c0c4cc; font-size: 12px; }

.summary-panel { min-height: 120px; }
.sum-block { margin-bottom: 22px; }
.sum-label {
  font-size: 13px;
  font-weight: 600;
  color: #374151;
  margin-bottom: 10px;
}
.sum-meta { font-weight: 400; color: #9ca3af; margin-left: 8px; font-size: 12px; }
.sum-text {
  font-size: 14px;
  line-height: 1.9;
  color: #1f2937;
  white-space: pre-wrap;
}
.sum-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.sum-timeline { padding-left: 2px; }
.step-title { font-size: 13px; color: #111827; font-weight: 550; }
.step-desc {
  font-size: 12px;
  color: #6b7280;
  margin-top: 3px;
  line-height: 1.6;
  word-break: break-word;
}
.step-chars { font-size: 11px; color: #c0c4cc; margin-top: 2px; }
</style>
