// app/web/js/chat/config.js
// 配置和全局状态模块

// API基础路径
export const API_BASE = '/api';

// 本地存储key - 确保与其他地方一致
export const STORAGE_KEY_SESSION = 'rag_current_session_id';

// DOM 元素引用
export const elements = {
    messagesContainer: null,
    chatInput: null,
    sendBtn: null,
    retrievalResults: null,
    similarityThreshold: null,
    thresholdValue: null,
    topKSelect: null,
    enableRerank: null,
    enableQueryRewrite: null,
    enableMemory: null,
    sessionBadge: null,
    newSessionBtn: null
};

// 全局状态
export const state = {
    isProcessing: false,
    useStreamMode: true,
    thinkingAnimationInterval: null,
    currentSessionId: null
};

// 初始化 DOM 元素
export function initElements() {
    elements.messagesContainer = document.getElementById('messagesContainer');
    elements.chatInput = document.getElementById('chatInput');
    elements.sendBtn = document.getElementById('sendBtn');
    elements.retrievalResults = document.getElementById('retrievalResults');
    elements.similarityThreshold = document.getElementById('similarityThreshold');
    elements.thresholdValue = document.getElementById('thresholdValue');
    elements.topKSelect = document.getElementById('topK');
    elements.enableRerank = document.getElementById('enableRerank');
    elements.enableQueryRewrite = document.getElementById('enableQueryRewrite');
    elements.enableMemory = document.getElementById('enableMemory');
    elements.sessionBadge = document.getElementById('sessionBadge');
    elements.newSessionBtn = document.getElementById('newSessionBtn');
}

// 更新状态
export function updateState(newState) {
    Object.assign(state, newState);
}