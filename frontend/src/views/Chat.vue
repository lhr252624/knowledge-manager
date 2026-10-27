<template>
  <div class="chat-shell">
    <!-- ============ 历史对话侧边栏 ============ -->
    <aside class="history" :class="{ collapsed }">
      <div class="history-head">
        <el-button class="new-btn" type="primary" :icon="Plus" @click="newChat">新对话</el-button>
        <el-button text :icon="Fold" class="fold-btn" @click="collapsed = true" />
      </div>

      <div class="history-list" v-loading="listLoading">
        <div v-if="!conversations.length" class="history-empty">
          还没有历史对话<br /><span>问点什么，这里就会出现记录</span>
        </div>

        <template v-for="group in grouped" :key="group.label">
          <div class="group-label">{{ group.label }}</div>
          <div
            v-for="c in group.items"
            :key="c.id"
            :class="['conv-item', { active: c.id === currentId }]"
            :title="c.title"
            @click="openConversation(c)"
          >
            <el-icon class="conv-icon"><ChatDotRound /></el-icon>
            <span class="conv-title">{{ c.title }}</span>
            <span class="conv-actions">
              <el-tooltip content="重命名" placement="top" :show-after="300">
                <el-icon @click.stop="renameConv(c)"><EditPen /></el-icon>
              </el-tooltip>
              <el-tooltip content="删除" placement="top" :show-after="300">
                <el-icon @click.stop="removeConv(c)"><Delete /></el-icon>
              </el-tooltip>
            </span>
          </div>
        </template>
      </div>

      <div class="history-foot">
        <div
          class="cache-line"
          :title="`词面阈值 ${cacheStats.similarity_threshold} / 语义召回 ${cacheStats.semantic_recall ?? 0.7} / 重排阈值 ${cacheStats.rerank?.threshold ?? 0.995}`"
        >
          <el-icon><Coin /></el-icon>
          <span>缓存 {{ cacheStats.entries }} 条</span>
          <em v-if="cacheStats.session_hits + cacheStats.session_misses">
            命中率 {{ hitRate }}%
          </em>
          <el-tag
            v-if="cacheStats.rerank"
            size="small"
            effect="plain"
            :type="cacheStats.rerank.ready ? 'success' : 'info'"
            class="rerank-tag"
          >
            {{ cacheStats.rerank.ready ? '重排就绪' : '重排加载中' }}
          </el-tag>
        </div>
        <div class="foot-actions">
          <el-button text size="small" @click="doClearCache">清缓存</el-button>
          <el-divider direction="vertical" />
          <el-button text size="small" @click="doClearHistory">清历史</el-button>
        </div>
      </div>
    </aside>

    <!-- ============ 对话主区 ============ -->
    <section class="chat-main">
      <div class="chat-head">
        <el-button v-if="collapsed" text :icon="Expand" @click="collapsed = false" />
        <div class="chat-title">{{ currentTitle }}</div>
        <el-tag v-if="cacheStats.enabled" size="small" effect="plain" type="info" class="cache-tag">
          语义缓存 {{ cacheStats.entries ? '已启用' : '待积累' }}
        </el-tag>
      </div>

      <div class="messages" ref="scrollRef">
        <div v-if="messages.length === 0 && !streaming" class="empty">
          <div class="empty-icon">💬</div>
          <div class="empty-title">向你的知识库提问</div>
          <div class="empty-hint">
            重复的问题会直接命中缓存（不调用大模型），追问会带上最近几轮上下文
          </div>
          <div class="samples">
            <el-tag v-for="s in samples" :key="s" class="sample" @click="ask(s)">{{ s }}</el-tag>
          </div>
        </div>

        <div v-for="(m, i) in messages" :key="i" :class="['msg', m.role]">
          <div class="avatar">{{ m.role === 'user' ? '我' : 'AI' }}</div>
          <div class="bubble">
            <div v-if="m.role === 'assistant'" class="md" v-html="render(m.content || '')"></div>
            <div v-else class="plain">{{ m.content }}</div>

            <div v-if="m.role === 'assistant' && (m.cached || m.latency_ms)" class="meta-line">
              <el-tooltip
                v-if="m.cached"
                placement="top"
                :content="cacheTip(m)"
              >
                <el-tag size="small" type="success" effect="plain" class="cache-badge">
                  缓存命中 · {{ methodLabel(m.method) }}
                </el-tag>
              </el-tooltip>
              <span v-if="m.latency_ms" class="latency">{{ m.latency_ms }}ms</span>
            </div>

            <div v-if="m.citations && m.citations.length" class="citations">
              <div class="citations-title">参考来源</div>
              <div
                v-for="c in m.citations"
                :key="c.index"
                class="citation"
                @click="openDoc(c.document_id)"
              >
                <span class="cite-index">[{{ c.index }}]</span>
                <span class="cite-main">
                  <span class="cite-title">{{ c.title || c.filename }}</span>
                  <span v-if="showFilename(c)" class="cite-file">{{ c.filename }}</span>
                </span>
                <el-tag
                  v-if="c.match_type && c.match_type !== 'vector'"
                  size="small"
                  type="warning"
                  effect="plain"
                >文件名命中</el-tag>
                <span class="cite-score">{{ (c.score ?? 0).toFixed(2) }}</span>
              </div>
            </div>
          </div>
        </div>

        <div v-if="streaming" class="msg assistant">
          <div class="avatar">AI</div>
          <div class="bubble">
            <div class="typing" v-if="!currentText">
              {{ streamCached ? '正在读取缓存…' : '正在检索并组织答案…' }}
            </div>
            <div v-else class="md" v-html="render(currentText)"></div>
            
            <!-- Agent 步骤时间线 -->
            <div v-if="streamSteps.length" class="agent-steps">
              <div class="steps-title">🔍 检索过程</div>
              <div class="step-timeline">
                <div v-for="(s, i) in streamSteps" :key="i" :class="['step-item', stepType(s)]">
                  <span class="step-icon">{{ stepIcon(s) }}</span>
                  <span class="step-text">{{ stepText(s) }}</span>
                </div>
              </div>
            </div>

            <div v-if="streamCached" class="meta-line">
              <el-tag size="small" type="success" effect="plain" class="cache-badge">
                缓存命中 · {{ methodLabel(streamCache?.method) }}
              </el-tag>
            </div>
          </div>
        </div>
      </div>

      <div class="composer">
        <el-input
          v-model="question"
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          type="textarea"
          :rows="2"
          resize="none"
          @keydown.enter.exact.prevent="ask()"
        />
        <div class="composer-side">
          <el-tooltip placement="top" content="关闭后每次都重新问模型">
            <el-checkbox v-model="useCache" size="small">用缓存</el-checkbox>
          </el-tooltip>
          <el-button
            type="primary"
            :icon="Promotion"
            :loading="streaming"
            @click="ask()"
            class="send-btn"
          >发送</el-button>
        </div>
      </div>
    </section>

    <!-- 文档预览 -->
    <el-drawer v-model="drawer" :title="drawerTitle" size="46%">
      <div class="doc-content">{{ drawerContent }}</div>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  ChatDotRound,
  Coin,
  Delete,
  EditPen,
  Expand,
  Fold,
  Plus,
  Promotion
} from '@element-plus/icons-vue'
import MarkdownIt from 'markdown-it'
import {
  askQuestionStream,
  clearCache,
  clearConversations,
  deleteConversation,
  getCacheStats,
  getConversation,
  getDocumentContent,
  listConversations,
  renameConversation
} from '../api'

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })
const route = useRoute()
const router = useRouter()

const messages = ref([])
const question = ref('')
const streaming = ref(false)
const currentText = ref('')
const scrollRef = ref(null)

const conversations = ref([])
const currentId = ref(null)
const listLoading = ref(false)
const collapsed = ref(false)
const useCache = ref(true)

const cacheStats = ref({ entries: 0, session_hits: 0, session_misses: 0, enabled: true, similarity_threshold: 0.85 })
const streamCached = ref(false)
const streamCache = ref(null)

const drawer = ref(false)
const drawerTitle = ref('')
const drawerContent = ref('')

const samples = [
  '总结一下知识库里的主要内容',
  '有哪些关键结论？',
  '这个项目的技术架构是什么？'
]

const streamSteps = ref([])

const hitRate = computed(() => {
  const total = cacheStats.value.session_hits + cacheStats.value.session_misses
  if (!total) return 0
  return Math.round((cacheStats.value.session_hits / total) * 100)
})

// 缓存命中方式：精确（分词一致）/ 词面（相似度 >= 0.85）/ 语义（向量召回 + 重排判定）
const METHOD_LABEL = {
  exact: '完全相同',
  lexical: '词面匹配',
  rerank: '语义判定'
}

function methodLabel(method) {
  return METHOD_LABEL[method] || '相似'
}

function cacheTip(m) {
  const method = methodLabel(m.method)
  if (m.method === 'rerank') {
    return `词面并不相同，但语义判定为同一个问题「${m.matched || ''}」（重排分 ${(m.score ?? 0).toFixed(3)}）`
  }
  if (m.method === 'lexical') {
    return `与历史问题「${m.matched || ''}」词面相似度 ${(m.similarity ?? 0).toFixed(2)}`
  }
  return `与历史问题「${m.matched || ''}」是同一个问题（${method}）`
}

const currentTitle = computed(() => {
  if (!currentId.value) return '新对话'
  const found = conversations.value.find((c) => c.id === currentId.value)
  return found?.title || '对话'
})

function render(text) {
  return md.render(text || '')
}

const UUID_NAME_RE = /^[0-9a-f]{32}\.[a-z0-9]+$/i

/** 引用卡片里，文件名与标题不同且不是 UUID 占位名时才单独显示 */
function showFilename(c) {
  const name = c.filename || ''
  if (!name || UUID_NAME_RE.test(name)) return false
  return name.replace(/\.[^.]+$/, '') !== (c.title || '')
}

/* ---------------- 侧边栏分组（仿 DeepSeek 的今天/昨天/最近7天） ---------------- */
function dayStart(offset = 0) {
  const d = new Date()
  d.setHours(0, 0, 0, 0)
  d.setDate(d.getDate() - offset)
  return d.getTime()
}

const grouped = computed(() => {
  const buckets = [
    { label: '今天', min: dayStart(0), items: [] },
    { label: '昨天', min: dayStart(1), items: [] },
    { label: '最近 7 天', min: dayStart(7), items: [] },
    { label: '更早', min: 0, items: [] }
  ]
  for (const c of conversations.value) {
    const ts = new Date(c.updated_at || c.created_at || 0).getTime()
    const bucket = buckets.find((b) => ts >= b.min) || buckets[buckets.length - 1]
    bucket.items.push(c)
  }
  return buckets.filter((b) => b.items.length)
})

/* ---------------- 数据加载 ---------------- */
async function loadList() {
  listLoading.value = true
  try {
    const data = await listConversations({ limit: 200 })
    conversations.value = data.items || []
  } catch {
    /* 已提示 */
  } finally {
    listLoading.value = false
  }
}

async function loadCacheStats() {
  try {
    cacheStats.value = await getCacheStats()
  } catch {
    /* 统计失败不影响对话 */
  }
}

async function scrollToBottom() {
  await nextTick()
  const el = scrollRef.value
  if (el) el.scrollTop = el.scrollHeight
}

function toMessage(m) {
  return {
    role: m.role,
    content: m.content,
    citations: m.citations || [],
    cached: !!m.cached,
    similarity: m.similarity,
    latency_ms: m.latency_ms
  }
}

async function openConversation(c) {
  if (streaming.value || c.id === currentId.value) return
  try {
    const data = await getConversation(c.id)
    currentId.value = data.id
    messages.value = (data.messages || []).map(toMessage)
    router.replace(`/chat/${data.id}`)
    scrollToBottom()
  } catch {
    /* 已提示 */
  }
}

function newChat() {
  if (streaming.value) return
  currentId.value = null
  messages.value = []
  question.value = ''
  router.replace('/chat')
}

async function renameConv(c) {
  try {
    const { value } = await ElMessageBox.prompt('输入新的会话名称', '重命名', {
      inputValue: c.title,
      confirmButtonText: '保存',
      cancelButtonText: '取消',
      inputValidator: (v) => (v && v.trim() ? true : '名称不能为空')
    })
    const updated = await renameConversation(c.id, value.trim())
    // 先本地更新，避免网络往返期间列表还显示旧名字
    const item = conversations.value.find((x) => x.id === c.id)
    if (item && updated?.title) item.title = updated.title
    await loadList()
  } catch {
    /* 取消或失败 */
  }
}

async function removeConv(c) {
  try {
    await ElMessageBox.confirm(`删除会话「${c.title}」？`, '删除确认', { type: 'warning' })
  } catch {
    return
  }
  try {
    await deleteConversation(c.id)
    if (c.id === currentId.value) newChat()
    await loadList()
    ElMessage.success('已删除')
  } catch {
    /* 已提示 */
  }
}

async function doClearHistory() {
  try {
    await ElMessageBox.confirm('清空全部历史对话？此操作不可恢复。', '清空历史', {
      type: 'warning'
    })
  } catch {
    return
  }
  try {
    await clearConversations()
    newChat()
    await loadList()
    ElMessage.success('历史对话已清空')
  } catch {
    /* 已提示 */
  }
}

async function doClearCache() {
  try {
    await ElMessageBox.confirm(
      '清空问答缓存？清空后相同问题会重新调用大模型。',
      '清空缓存',
      { type: 'warning' }
    )
  } catch {
    return
  }
  try {
    const res = await clearCache()
    await loadCacheStats()
    ElMessage.success(res.message || '缓存已清空')
  } catch {
    /* 已提示 */
  }
}

/* ---------------- Agent 步骤展示 ---------------- */
const STEP_ICONS = {
  search_kb: '🔍',
  search_by_filename: '📂',
  read_document: '📄',
  list_documents: '📚',
  finish: '✅',
  give_up: '❌',
  thought: '💭'
}

function stepIcon(s) {
  return STEP_ICONS[s.tool] || '🔧'
}

function stepType(s) {
  if (s.tool === 'finish') return 'success'
  if (s.tool === 'give_up') return 'warning'
  return 'info'
}

function stepText(s) {
  const tool = s.tool
  const args = s.args || {}
  if (tool === 'search_kb') return `检索：${args.query || ''}`
  if (tool === 'search_by_filename') return `查找文档：${args.name || ''}`
  if (tool === 'read_document') {
    const start = args.start_chunk ?? 0
    const end = args.end_chunk ?? start
    return `读取文档 ${args.document_id} 第 ${start + 1}~${end + 1} 块`
  }
  if (tool === 'list_documents') return '列出所有文档'
  if (tool === 'finish') return `收集完成${args.reason ? `: ${args.reason}` : ''}`
  if (tool === 'give_up') return `未找到${args.reason ? `: ${args.reason}` : ''}`
  if (tool === 'thought') return '思考中'
  return `${tool}(${JSON.stringify(args)})`
}

/* ---------------- 提问 ---------------- */
function ask(preset) {
  const q = (preset ?? question.value).trim()
  if (!q || streaming.value) return

  messages.value.push({ role: 'user', content: q })
  question.value = ''
  streaming.value = true
  currentText.value = ''
  streamCached.value = false
  streamCache.value = null
  streamSteps.value = []

  let citations = []
  let answer = ''
  let cacheInfo = null
  let latencyMs = null
  let finished = false
  let convId = currentId.value

  const finalize = () => {
    if (finished) return
    finished = true
    streaming.value = false
    messages.value.push({
      role: 'assistant',
      content: answer || currentText.value || '（未获得回答）',
      citations,
      cached: !!cacheInfo,
      similarity: cacheInfo?.similarity,
      matched: cacheInfo?.matched_question,
      method: cacheInfo?.method,
      latency_ms: latencyMs
    })
    currentText.value = ''
    streamCached.value = false
    scrollToBottom()
    loadList()
    loadCacheStats()
  }

  askQuestionStream(
    {
      question: q,
      top_k: 5,
      conversation_id: convId || undefined,
      use_cache: useCache.value
    },
    {
      onMeta: (meta) => {
        citations = meta.citations || []
        cacheInfo = meta.cache || null
        if (cacheInfo) {
          streamCached.value = true
          streamCache.value = cacheInfo
        }
        streamSteps.value = meta.steps || []
        // 第一句话由后端建会话，这里把地址栏和当前会话同步过来
        if (meta.conversation_id && !convId) {
          convId = meta.conversation_id
          currentId.value = convId
          router.replace(`/chat/${convId}`)
        }
      },
      onDelta: (text) => {
        answer += text
        currentText.value = answer
        scrollToBottom()
      },
      onError: (msg) => {
        answer = answer || `回答失败：${msg}`
        finalize()
      },
      onDone: (data) => {
        if (data && data.latency_ms != null) latencyMs = data.latency_ms
        // 保存最终步骤列表到消息（可选，暂不实现历史消息的步骤展示）
        finalize()
      }
    }
  ).finally(() => {
    if (!finished) finalize()
  })

  scrollToBottom()
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

watch(
  () => route.params.id,
  async (id) => {
    if (!id) {
      if (currentId.value) {
        currentId.value = null
        messages.value = []
      }
      return
    }
    if (Number(id) === currentId.value) return
    try {
      const data = await getConversation(id)
      currentId.value = data.id
      messages.value = (data.messages || []).map(toMessage)
      scrollToBottom()
    } catch {
      router.replace('/chat')
    }
  }
)

onMounted(async () => {
  await Promise.all([loadList(), loadCacheStats()])
  const id = route.params.id
  if (id) {
    try {
      const data = await getConversation(id)
      currentId.value = data.id
      messages.value = (data.messages || []).map(toMessage)
      scrollToBottom()
    } catch {
      router.replace('/chat')
    }
  }
})
</script>

<style scoped>
.chat-shell {
  display: flex;
  gap: 14px;
  height: calc(100vh - 48px);
  max-width: none;
}

/* ---------------- 侧边栏 ---------------- */
.history {
  width: 236px;
  flex-shrink: 0;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: width 0.18s ease, opacity 0.18s ease;
}
.history.collapsed {
  width: 0;
  opacity: 0;
  pointer-events: none;
}

.history-head {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px;
  border-bottom: 1px solid #f1f3f7;
}
.new-btn { flex: 1; }
.fold-btn { flex-shrink: 0; }

.history-list {
  flex: 1;
  overflow-y: auto;
  padding: 6px 8px;
}
.history-empty {
  color: #b0b7c3;
  font-size: 12px;
  text-align: center;
  padding: 40px 10px;
  line-height: 1.9;
}
.history-empty span { color: #cbd2dc; font-size: 11px; }

.group-label {
  font-size: 11px;
  color: #9ca3af;
  padding: 10px 8px 4px;
  letter-spacing: 0.5px;
}

.conv-item {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 8px 9px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  color: #374151;
  position: relative;
}
.conv-item:hover { background: #f3f6fb; }
.conv-item.active {
  background: #eaf1ff;
  color: #1d4ed8;
  font-weight: 550;
}
.conv-icon { font-size: 14px; color: #9ca3af; flex-shrink: 0; }
.conv-item.active .conv-icon { color: #3b82f6; }
.conv-title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.conv-actions {
  display: none;
  gap: 4px;
  flex-shrink: 0;
  color: #9ca3af;
}
.conv-item:hover .conv-actions { display: flex; }
.conv-actions .el-icon:hover { color: #3b82f6; }

.history-foot {
  border-top: 1px solid #f1f3f7;
  padding: 8px 10px;
}
.cache-line {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 5px;
  font-size: 11px;
  color: #9ca3af;
}
.cache-line em { font-style: normal; color: #10b981; }
.rerank-tag { transform: scale(0.85); transform-origin: left center; }
.foot-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  margin-top: 2px;
}

/* ---------------- 主区 ---------------- */
.chat-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.chat-head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 2px 10px;
}
.chat-title {
  font-size: 15px;
  font-weight: 600;
  color: #111827;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cache-tag { margin-left: auto; }

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 8px 4px 16px;
}

.empty { text-align: center; padding: 70px 20px; color: #6b7280; }
.empty-icon { font-size: 44px; }
.empty-title { font-size: 17px; font-weight: 600; margin: 12px 0 4px; color: #374151; }
.empty-hint { font-size: 13px; color: #9ca3af; margin-bottom: 18px; }
.samples { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
.sample { cursor: pointer; }

.msg { display: flex; gap: 10px; margin-bottom: 16px; align-items: flex-start; }
.msg.user { flex-direction: row-reverse; }

.avatar {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: #e5e7eb;
  color: #374151;
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.msg.assistant .avatar { background: #3b82f6; color: #fff; }

.bubble {
  max-width: 78%;
  background: #fff;
  border-radius: 10px;
  padding: 12px 14px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
  font-size: 14px;
  line-height: 1.75;
  overflow-wrap: break-word;
}
.msg.user .bubble { background: #3b82f6; color: #fff; }

.plain { white-space: pre-wrap; }

.meta-line {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
}
.cache-badge { cursor: help; }
.latency { font-size: 11px; color: #c0c4cc; }

.md :deep(p) { margin: 0 0 8px; }
.md :deep(p:last-child) { margin-bottom: 0; }
.md :deep(ul), .md :deep(ol) { padding-left: 22px; margin: 6px 0; }
.md :deep(code) { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 13px; }
.md :deep(pre) {
  background: #111827;
  color: #e5e7eb;
  padding: 12px;
  border-radius: 8px;
  overflow-x: auto;
  margin: 8px 0;
}
.md :deep(pre code) { background: transparent; color: inherit; }
.md :deep(strong) { font-weight: 650; }
.md :deep(blockquote) {
  border-left: 3px solid #d1d5db;
  padding-left: 10px;
  color: #6b7280;
  margin: 8px 0;
}

.typing { color: #9ca3af; font-size: 13px; }

.citations {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed #e5e7eb;
}
.citations-title { font-size: 12px; color: #9ca3af; margin-bottom: 6px; }
.citation {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  padding: 4px 6px;
  border-radius: 6px;
  cursor: pointer;
}
.citation:hover { background: #f3f4f6; }
.cite-index { color: #3b82f6; font-weight: 600; }
.cite-main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.cite-title { color: #374151; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cite-file { color: #9ca3af; font-size: 11px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cite-score { color: #9ca3af; }

/* Agent 步骤时间线 */
.agent-steps {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid #f3f4f6;
}
.steps-title {
  font-size: 11px;
  color: #6b7280;
  margin-bottom: 6px;
  font-weight: 600;
}
.step-timeline {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.step-item {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  font-size: 12px;
  padding: 3px 6px;
  border-radius: 4px;
  background: #f9fafb;
}
.step-item.success { background: #ecfdf5; color: #065f46; }
.step-item.warning { background: #fef3c7; color: #92400e; }
.step-item.info { background: #eff6ff; color: #1e40af; }
.step-icon {
  flex-shrink: 0;
  font-size: 13px;
}
.step-text {
  flex: 1;
  line-height: 1.4;
  word-break: break-word;
}

.composer {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  background: #fff;
  padding: 12px;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
}
.composer-side {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}
.send-btn { height: 44px; width: 92px; }

.doc-content { white-space: pre-wrap; line-height: 1.9; font-size: 14px; color: #374151; }
</style>
