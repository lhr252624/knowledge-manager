import axios from 'axios'
import { ElMessage } from 'element-plus'

const http = axios.create({
  baseURL: '/api/v1',
  timeout: 180000
})

// 统一错误提示
http.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail =
      error?.response?.data?.detail ||
      error?.message ||
      '请求失败'
    ElMessage.error(String(detail).slice(0, 200))
    return Promise.reject(error)
  }
)

/* ---------------- 系统 ---------------- */
export const getHealthDetail = () => http.get('/health/detail')
export const getStats = () => http.get('/stats')
export const getAgents = () => http.get('/agents')

/* ---------------- 文档 ---------------- */
export const listDocuments = (params) => http.get('/documents', { params })

export const getDocument = (id, withChunks = false) =>
  http.get(`/documents/${id}`, { params: { with_chunks: withChunks } })

export const getDocumentContent = (id, maxChars = 20000) =>
  http.get(`/documents/${id}/content`, { params: { max_chars: maxChars } })

// 摘要 Agent 的决策轨迹：读了哪几段、每步累计多少字
export const getSummaryTrace = (id) => http.get(`/documents/${id}/summary-trace`)

export const deleteDocument = (id) => http.delete(`/documents/${id}`)

export const reindexDocument = (id) => http.post(`/documents/${id}/reindex`)

// 改文件名/标题：文件名参与检索，后端会顺带重算该文档的向量
export const updateDocument = (id, payload) => http.patch(`/documents/${id}`, payload)

export const uploadDocument = (file, onProgress) => {
  const form = new FormData()
  form.append('file', file)
  return http.post('/documents/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    }
  })
}

/* ---------------- 检索 ---------------- */
export const search = (payload) => http.post('/search', payload)
export const askQuestion = (payload) => http.post('/qa', payload)

/* ---------------- 历史对话 ---------------- */
export const listConversations = (params) => http.get('/conversations', { params })

export const getConversation = (id) => http.get(`/conversations/${id}`)

export const renameConversation = (id, title) =>
  http.patch(`/conversations/${id}`, { title })

export const deleteConversation = (id) => http.delete(`/conversations/${id}`)

export const clearConversations = () => http.delete('/conversations')

/* ---------------- 问答缓存 ---------------- */
export const getCacheStats = () => http.get('/cache/stats')

export const listCacheEntries = (params) => http.get('/cache', { params })

export const clearCache = (params) => http.delete('/cache', { params })

/* ---------------- 知识图谱 ---------------- */
export const getGraph = (params) => http.get('/graph', { params })

/* ---------------- 流式问答 (SSE) ---------------- */
export async function askQuestionStream(payload, { onMeta, onStep, onDelta, onDone, onReplace, onError } = {}) {
  try {
    const response = await fetch('/api/v1/qa/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })

    if (!response.ok || !response.body) {
      const text = await response.text()
      throw new Error(text || `HTTP ${response.status}`)
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // 按 SSE 空行切分事件
      let sepIndex
      while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
        const raw = buffer.slice(0, sepIndex)
        buffer = buffer.slice(sepIndex + 2)

        let event = 'message'
        const dataLines = []
        for (const line of raw.split('\n')) {
          if (line.startsWith('event:')) event = line.slice(6).trim()
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
        }
        const dataStr = dataLines.join('\n')
        if (!dataStr) continue

        let data = {}
        try {
          data = JSON.parse(dataStr)
        } catch {
          data = { text: dataStr }
        }

        if (event === 'meta') onMeta?.(data)
        else if (event === 'step') onStep?.(data)
        else if (event === 'delta') onDelta?.(data.text || '')
        else if (event === 'replace') onReplace?.(data.text || '', data.issues || [])
        else if (event === 'error') onError?.(data.message || '未知错误')
        else if (event === 'done') onDone?.(data)
      }
    }
    onDone?.()
  } catch (err) {
    onError?.(err?.message || String(err))
  }
}

export default http
