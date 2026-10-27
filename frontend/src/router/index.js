import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'Home', component: () => import('../views/Home.vue') },
  { path: '/chat/:id?', name: 'Chat', component: () => import('../views/Chat.vue') },
  { path: '/search', name: 'Search', component: () => import('../views/Search.vue') },
  { path: '/documents', name: 'Documents', component: () => import('../views/Documents.vue') },
  { path: '/graph', name: 'Graph', component: () => import('../views/Graph.vue') }
]

export default createRouter({
  history: createWebHistory(),
  routes
})
