// app/web/js/chat/session.js
/* app/web/js/chat/session.js */
// 会话管理模块 - 支持 URL 参数和 PostgreSQL 持久化

import { elements, state, updateState, STORAGE_KEY_SESSION, API_BASE } from './config.js';
import { clearAllMessages, addWelcomeMessage, addMessage } from './messages.js';
import {showToast, showSessionToast, escapeHtml, formatContent} from './utils.js';

// 从 URL 获取 session_id
export function getSessionIdFromURL() {
    const urlParams = new URLSearchParams(window.location.search);
    const session = urlParams.get('session');
    console.log('从 URL 获取 session_id:', session);
    return session;
}

// 更新 URL 中的 session_id（不刷新页面）
export function updateURLWithSessionId(sessionId) {
    if (!sessionId || sessionId === 'default' || sessionId === 'null' || sessionId === 'undefined') {
        console.log('跳过更新 URL: session_id 无效', sessionId);
        return;
    }

    const url = new URL(window.location.href);
    const currentSession = url.searchParams.get('session');

    // 只有当 session_id 不同时才更新
    if (currentSession !== sessionId) {
        url.searchParams.set('session', sessionId);
        window.history.replaceState({}, '', url);
        console.log('URL 已更新:', url.toString());
    }
}

// 清除 URL 中的 session_id
export function clearURLSessionId() {
    const url = new URL(window.location.href);
    url.searchParams.delete('session');
    window.history.replaceState({}, '', url);
    console.log('已清除 URL 中的 session 参数');
}

// 滚动到底部（本地函数，避免重复导入）
function scrollToBottom(container) {
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

// 格式化时间
function formatTime(isoString) {
    if (!isoString) return new Date().toLocaleTimeString();
    try {
        const date = new Date(isoString);
        return date.toLocaleTimeString();
    } catch (e) {
        return new Date().toLocaleTimeString();
    }
}

// 更新会话徽章显示
export function updateSessionBadge(hasSession) {
    if (elements.sessionBadge) {
        if (hasSession && state.currentSessionId && state.currentSessionId !== 'default') {
            const shortId = state.currentSessionId.substring(0, 8) + '...';
            elements.sessionBadge.innerHTML = `📝 会话: ${shortId}`;
            elements.sessionBadge.classList.add('has-session');
            elements.sessionBadge.title = `会话ID: ${state.currentSessionId}`;
        } else {
            elements.sessionBadge.innerHTML = `📝 新会话`;
            elements.sessionBadge.classList.remove('has-session');
            elements.sessionBadge.title = '点击发送问题开始新会话';
        }
    }
}

// 保存会话ID到本地和 URL
export function saveSessionId(sessionId) {
    if (sessionId && sessionId !== 'default' && sessionId !== 'null' && sessionId !== 'undefined') {
        const oldSessionId = state.currentSessionId;
        updateState({ currentSessionId: sessionId });

        // 保存到 localStorage（备用）
        try {
            localStorage.setItem(STORAGE_KEY_SESSION, sessionId);
            console.log('会话已保存到 localStorage:', sessionId);
        } catch (e) {
            console.warn('保存会话失败:', e);
        }

        // 更新 URL
        updateURLWithSessionId(sessionId);
        updateSessionBadge(true);
        console.log('会话已保存:', sessionId, '(旧会话:', oldSessionId, ')');
    }
}

// 清除保存的会话
export function clearSavedSession() {
    updateState({ currentSessionId: null });
    try {
        localStorage.removeItem(STORAGE_KEY_SESSION);
        console.log('已清除 localStorage 中的会话');
    } catch (e) {
        console.warn('清除会话失败:', e);
    }
    clearURLSessionId();
    updateSessionBadge(false);
}

// 验证会话是否存在
// app/web/js/chat/session.js (修复验证函数)

// 验证会话是否存在
async function verifySessionExists(sessionId) {
    if (!sessionId || sessionId === 'default' || sessionId === 'null') return false;

    try {
        console.log('验证会话:', sessionId);
        const response = await fetch(`${API_BASE}/chat/session/${sessionId}`);
        if (response.ok) {
            const data = await response.json();
            // 修复：检查 data.success 和 data.info 是否为真
            const exists = data.success === true && data.info !== null && data.info !== undefined;
            console.log('会话验证结果:', exists ? '存在' : '不存在', data);
            return exists;
        }
        return false;
    } catch (error) {
        console.warn('验证会话失败:', error);
        return false;
    }
}

// 加载会话历史消息
async function loadSessionHistory(sessionId) {
    if (!sessionId || sessionId === 'default') return [];

    console.log('加载会话历史:', sessionId);
    try {
        const response = await fetch(`${API_BASE}/chat/session/${sessionId}/history?limit=50`);
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.messages) {
                console.log(`加载会话历史成功: ${data.messages.length} 条消息`);
                return data.messages;
            }
        }
        return [];
    } catch (error) {
        console.warn('加载会话历史失败:', error);
        return [];
    }
}

// 渲染加载的会话历史
export function renderSessionHistory(messages) {
    if (!messages || messages.length === 0) {
        console.log('没有历史消息可渲染');
        return false;
    }

    console.log('渲染会话历史:', messages.length, '条消息');

    // 清空当前消息
    const existingMessages = document.querySelectorAll('.message:not(.thinking)');
    existingMessages.forEach(msg => msg.remove());

    let hasUserMessages = false;

    for (const msg of messages) {
        // 跳过系统消息
        if (msg.role === 'system') continue;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.role}`;
        const avatar = msg.role === 'user' ? '👤' : '🤖';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${formatContent(msg.content)}</div>
                <div class="message-meta">${formatTime(msg.created_at)}</div>
            </div>
        `;

        elements.messagesContainer.appendChild(messageDiv);
        if (msg.role === 'user') hasUserMessages = true;
    }

    if (hasUserMessages) {
        scrollToBottom(elements.messagesContainer);
    } else if (messages.length === 0) {
        // 没有历史，显示欢迎消息
        addWelcomeMessage();
    }

    return hasUserMessages;
}

// app/web/js/chat/session.js (修复 loadSavedSession 函数)

// 加载保存的会话（优先从 URL 读取）
export async function loadSavedSession() {
    console.log('开始加载保存的会话...');
    console.log('当前 URL:', window.location.href);

    // 1. 优先从 URL 参数获取 session_id
    let sessionId = getSessionIdFromURL();
    console.log('从 URL 获取的 session_id:', sessionId);

    // 2. 如果 URL 没有，从 localStorage 获取
    if (!sessionId) {
        const saved = localStorage.getItem(STORAGE_KEY_SESSION);
        if (saved && saved !== 'null' && saved !== 'undefined' && saved !== 'default') {
            sessionId = saved;
            console.log('从 localStorage 获取会话:', sessionId);
        }
    }

    // 3. 如果有 session_id，验证并加载历史
    if (sessionId && sessionId !== 'null' && sessionId !== 'undefined') {
        console.log('找到会话 ID:', sessionId);
        const isValid = await verifySessionExists(sessionId);
        if (isValid) {
            updateState({ currentSessionId: sessionId });

            // 确保 URL 中有 session_id
            updateURLWithSessionId(sessionId);

            // 加载历史消息
            const history = await loadSessionHistory(sessionId);
            console.log('加载到的历史消息数量:', history.length);

            if (history.length > 0) {
                renderSessionHistory(history);
                showSessionToast(`✅ 已恢复会话 (${history.length} 条消息)`, 'info');
            } else {
                // 即使没有历史消息，也保留会话
                updateSessionBadge(true);
                showSessionToast('✅ 已恢复会话', 'info');
            }
            console.log('加载保存的会话成功:', sessionId);
            return sessionId;
        } else {
            console.log('保存的会话已失效，清除');
            clearSavedSession();
        }
    }

    // 4. 没有有效会话，创建新会话
    console.log('没有有效会话，创建新会话...');
    updateState({ currentSessionId: null });
    updateSessionBadge(false);

    // 创建新会话并更新 URL
    const newSessionId = await createAndSetNewSession();
    return newSessionId;
}

// 创建新会话并更新 URL
export async function createAndSetNewSession() {
    try {
        console.log('创建新会话...');
        const response = await fetch(`${API_BASE}/chat/session/create`, {
            method: 'GET'
        });

        if (response.ok) {
            const data = await response.json();
            if (data.success && data.session_id) {
                updateState({ currentSessionId: data.session_id });
                updateURLWithSessionId(data.session_id);
                updateSessionBadge(true);
                console.log('创建新会话成功:', data.session_id);
                return data.session_id;
            }
        }
        console.error('创建会话响应失败:', response.status);
    } catch (e) {
        console.error('创建会话失败:', e);
    }
    return null;
}

// 新建会话
export async function newSession() {
    if (state.isProcessing) {
        showToast('请等待当前回答完成', 'warning');
        return;
    }

    const messageCount = document.querySelectorAll('.message:not(.thinking)').length;
    if (messageCount > 1) {
        if (!confirm('新建会话将清除当前对话历史，确定要继续吗？')) {
            return;
        }
    }

    console.log('创建新会话...');

    // 清除前端状态
    clearSavedSession();

    // 清空消息
    clearAllMessages();
    addWelcomeMessage();

    // 创建新会话
    await createAndSetNewSession();

    // 清空检索结果
    const retrievalResults = document.getElementById('retrievalResults');
    if (retrievalResults) {
        retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">点击发送问题后，相关文档将显示在这里</div>';
    }

    showSessionToast('✨ 已创建新会话', 'info');
    if (elements.chatInput) elements.chatInput.focus();
}

// 导出获取当前会话ID的函数
export function getCurrentSessionId() {
    return state.currentSessionId;
}