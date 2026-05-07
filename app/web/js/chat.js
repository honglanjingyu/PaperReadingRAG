/* app/web/js/chat.js */
/* 聊天页面逻辑 - 支持流式输出，包含思考动画和会话管理 */

// DOM 元素
const messagesContainer = document.getElementById('messagesContainer');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const retrievalResults = document.getElementById('retrievalResults');
const similarityThreshold = document.getElementById('similarityThreshold');
const thresholdValue = document.getElementById('thresholdValue');
const topKSelect = document.getElementById('topK');
const enableRerank = document.getElementById('enableRerank');
const enableQueryRewrite = document.getElementById('enableQueryRewrite');
const enableMemory = document.getElementById('enableMemory');
const sessionBadge = document.getElementById('sessionBadge');
const newSessionBtn = document.getElementById('newSessionBtn');  // 只保留新建会话按钮

let isProcessing = false;
let useStreamMode = true;
let thinkingAnimationInterval = null;
let currentSessionId = null;  // 当前会话ID

// 本地存储key
const STORAGE_KEY_SESSION = 'rag_current_session_id';

// ========== 会话管理函数 ==========

// 加载保存的会话ID
async function loadSavedSession() {
    try {
        const saved = localStorage.getItem(STORAGE_KEY_SESSION);
        if (saved && saved !== 'null' && saved !== 'undefined' && saved !== 'default') {
            // 验证会话是否仍然有效
            const isValid = await verifySessionExists(saved);
            if (isValid) {
                currentSessionId = saved;
                updateSessionBadge(true);
                console.log('加载保存的会话:', currentSessionId);
                showSessionToast('✅ 已恢复上次会话', 'info');
                return;
            } else {
                // 会话已失效，清除本地存储
                console.log('保存的会话已失效，清除');
                clearSavedSession();
            }
        }
    } catch (e) {
        console.warn('加载会话失败:', e);
        clearSavedSession();
    }

    // 如果没有有效会话，显示欢迎消息但持有null
    currentSessionId = null;
    updateSessionBadge(false);
}

// 验证会话是否存在
async function verifySessionExists(sessionId) {
    if (!sessionId || sessionId === 'default' || sessionId === 'null') return false;

    try {
        const response = await fetch(`${API_BASE}/chat/session/${sessionId}`);
        if (response.ok) {
            const data = await response.json();
            return data.success === true;
        }
        return false;
    } catch (error) {
        console.warn('验证会话失败:', error);
        return false;
    }
}

// 保存会话ID到本地
function saveSessionId(sessionId) {
    if (sessionId && sessionId !== 'default' && sessionId !== 'null' && sessionId !== 'undefined') {
        currentSessionId = sessionId;
        try {
            localStorage.setItem(STORAGE_KEY_SESSION, sessionId);
            console.log('保存会话:', sessionId);
        } catch (e) {
            console.warn('保存会话失败:', e);
        }
        updateSessionBadge(true);
    }
}

// 清除保存的会话
function clearSavedSession() {
    currentSessionId = null;
    try {
        localStorage.removeItem(STORAGE_KEY_SESSION);
        console.log('清除保存的会话');
    } catch (e) {
        console.warn('清除会话失败:', e);
    }
    updateSessionBadge(false);
}

// 更新会话徽章显示
function updateSessionBadge(hasSession) {
    if (sessionBadge) {
        if (hasSession && currentSessionId && currentSessionId !== 'default') {
            const shortId = currentSessionId.substring(0, 8) + '...';
            sessionBadge.innerHTML = `📝 会话: ${shortId}`;
            sessionBadge.classList.add('has-session');
            sessionBadge.title = `会话ID: ${currentSessionId}`;
        } else {
            sessionBadge.innerHTML = `📝 新会话`;
            sessionBadge.classList.remove('has-session');
            sessionBadge.title = '点击发送问题开始新会话';
        }
    }
}

// 显示会话提示
function showSessionToast(message, type = 'info') {
    // 移除已存在的toast
    const existingToasts = document.querySelectorAll('.session-toast');
    existingToasts.forEach(toast => toast.remove());

    const toast = document.createElement('div');
    toast.className = `session-toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2000);
}

// 新建会话
async function newSession() {
    if (isProcessing) {
        showToast('请等待当前回答完成', 'warning');
        return;
    }

    // 确认是否清除当前会话（如果有对话历史）
    const messageCount = document.querySelectorAll('.message:not(.thinking)').length;
    if (messageCount > 1) {  // 有对话历史（超过欢迎消息）
        if (!confirm('新建会话将清除当前对话历史，确定要继续吗？')) {
            return;
        }
    }

    // 清除当前会话（如果存在）
    if (currentSessionId && currentSessionId !== 'default') {
        try {
            const response = await fetch(`${API_BASE}/chat/session/${currentSessionId}`, {
                method: 'DELETE'
            });
            if (response.ok) {
                console.log('后端会话已清除');
            }
        } catch (e) {
            console.warn('清除会话失败:', e);
        }
    }

    // 重置状态
    clearSavedSession();

    // 清空消息区域（保留欢迎消息）
    clearAllMessages();

    // 添加欢迎消息
    addWelcomeMessage();

    // 清空检索结果
    if (retrievalResults) {
        retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">点击发送问题后，相关文档将显示在这里</div>';
    }

    // 显示提示
    showSessionToast('✨ 已创建新会话', 'info');

    // 聚焦输入框
    chatInput.focus();
}

// 清空所有消息
function clearAllMessages() {
    const messages = document.querySelectorAll('.message');
    messages.forEach(msg => {
        msg.remove();
    });
}

// 添加欢迎消息
function addWelcomeMessage() {
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'message assistant';
    welcomeDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-text">您好！我是RAG智能问答助手。请上传文档后，向我提问任何关于文档内容的问题。</div>
            <div class="message-meta">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    messagesContainer.appendChild(welcomeDiv);
    scrollToBottom();
}

// 显示阈值
if (similarityThreshold) {
    similarityThreshold.addEventListener('input', () => {
        thresholdValue.textContent = similarityThreshold.value;
    });
}

// 自动调整textarea高度
if (chatInput) {
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 100) + 'px';
    });
}

// 新建会话按钮事件
if (newSessionBtn) {
    newSessionBtn.addEventListener('click', newSession);
}

// 发送消息 - 主入口
async function sendMessage() {
    const question = chatInput.value.trim();
    if (!question || isProcessing) return;

    // 清空输入框
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // 添加用户消息
    addMessage('user', question);

    // 显示加载状态
    isProcessing = true;
    sendBtn.disabled = true;

    // 清空之前的检索结果
    if (retrievalResults) {
        retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">正在检索...</div>';
    }

    try {
        if (useStreamMode) {
            await sendMessageStream(question);
        } else {
            await sendMessageNormal(question);
        }
    } catch (error) {
        console.error('发送消息失败:', error);
        removeAllThinkingIndicators();
        addMessage('assistant', `网络错误：${error.message}`);
        if (retrievalResults) {
            retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">检索失败</div>';
        }
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

// ========== 思考动画相关函数 ==========

// 添加思考动画消息
function addThinkingMessage() {
    const messageId = `thinking_${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant thinking';
    messageDiv.id = messageId;
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-text">
                <span class="thinking-text">正在思考</span>
                <span class="thinking-dots">...</span>
            </div>
            <div class="message-meta"></div>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);

    startThinkingAnimation(messageId);
    scrollToBottom();
    return messageId;
}

// 启动思考动画
function startThinkingAnimation(messageId) {
    if (thinkingAnimationInterval) {
        clearInterval(thinkingAnimationInterval);
    }

    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;

    const dotsSpan = messageDiv.querySelector('.thinking-dots');
    if (!dotsSpan) return;

    let dotCount = 1;
    let increasing = true;

    thinkingAnimationInterval = setInterval(() => {
        const currentDiv = document.getElementById(messageId);
        if (!currentDiv) {
            if (thinkingAnimationInterval) {
                clearInterval(thinkingAnimationInterval);
                thinkingAnimationInterval = null;
            }
            return;
        }

        const currentDotsSpan = currentDiv.querySelector('.thinking-dots');
        if (!currentDotsSpan) return;

        if (increasing) {
            dotCount++;
            if (dotCount >= 3) {
                dotCount = 3;
                increasing = false;
            }
        } else {
            dotCount--;
            if (dotCount <= 1) {
                dotCount = 1;
                increasing = true;
            }
        }

        currentDotsSpan.textContent = '.'.repeat(dotCount);
    }, 400);
}

// 停止思考动画
function stopThinkingAnimation() {
    if (thinkingAnimationInterval) {
        clearInterval(thinkingAnimationInterval);
        thinkingAnimationInterval = null;
    }
}

// 移除所有思考指示器
function removeAllThinkingIndicators() {
    stopThinkingAnimation();
    const indicators = document.querySelectorAll('[id^="thinking_"], [id^="loading_"]');
    indicators.forEach(el => el.remove());
}

// 添加消息
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.innerHTML = `
        <div class="message-avatar">${role === 'user' ? '👤' : '🤖'}</div>
        <div class="message-content">
            <div class="message-text">${formatContent(content)}</div>
            <div class="message-meta">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

// 添加消息并返回ID
function addMessageAndReturnId(role, content) {
    const messageId = `msg_${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    messageDiv.id = messageId;
    messageDiv.innerHTML = `
        <div class="message-avatar">${role === 'user' ? '👤' : '🤖'}</div>
        <div class="message-content">
            <div class="message-text">${formatContent(content)}</div>
            <div class="message-meta">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    return messageId;
}

// 添加系统消息
function addSystemMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-avatar">ℹ️</div>
        <div class="message-content" style="background: #e7f3ff; font-size: 12px;">${escapeHtml(content)}</div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

// ========== 非流式模式 ==========

async function sendMessageNormal(question) {
    const thinkingId = addThinkingMessage();

    try {
        const response = await fetch(`${API_BASE}/chat/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                session_id: currentSessionId,
                top_k: parseInt(topKSelect?.value || 5),
                similarity_threshold: parseFloat(similarityThreshold?.value || 0.3),
                enable_rerank: enableRerank?.checked ?? true,
                enable_query_rewrite: enableQueryRewrite?.checked ?? true,
                enable_memory: enableMemory?.checked ?? true,
                template_name: 'detailed'
            })
        });

        const data = await response.json();

        if (data.success) {
            // 保存返回的session_id
            if (data.session_id && data.session_id !== 'default') {
                saveSessionId(data.session_id);
            }
            replaceThinkingWithContent(thinkingId, data.answer);
            if (retrievalResults) {
                displayRetrievalResults(data.results, data.retrieval_info);
            }
            if (data.rewritten_query && data.rewritten_query !== question) {
                addSystemMessage(`✨ Query改写: "${data.rewritten_query}"`);
            }
            if (data.has_history) {
                addSystemMessage(`📝 已结合对话历史回答问题`);
            }
        } else {
            replaceThinkingWithContent(thinkingId, `抱歉，处理您的问题时出错：${data.error || '未知错误'}`);
            if (retrievalResults) {
                retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">检索失败</div>';
            }
        }
    } catch (error) {
        removeThinkingIndicator(thinkingId);
        throw error;
    }
}

// 将思考消息替换为实际内容
function replaceThinkingWithContent(thinkingMessageId, content) {
    stopThinkingAnimation();

    const thinkingDiv = document.getElementById(thinkingMessageId);
    if (!thinkingDiv) {
        return addMessageAndReturnId('assistant', content);
    }

    thinkingDiv.classList.remove('thinking');

    const textDiv = thinkingDiv.querySelector('.message-text');
    if (textDiv) {
        textDiv.innerHTML = formatContent(content);
    }

    const metaSpan = thinkingDiv.querySelector('.message-meta');
    if (metaSpan) {
        metaSpan.textContent = new Date().toLocaleTimeString();
    }

    const newId = `msg_${Date.now()}`;
    thinkingDiv.id = newId;

    return newId;
}

// 移除思考指示器
function removeThinkingIndicator(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

// ========== 流式模式 ==========

async function sendMessageStream(question) {
    const thinkingId = addThinkingMessage();
    let assistantMessageId = null;
    let fullResponse = '';
    let hasReceivedFirstChunk = false;
    let newSessionId = null;

    try {
        const response = await fetch(`${API_BASE}/chat/ask/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                session_id: currentSessionId,
                history: getChatHistory(),
                top_k: parseInt(topKSelect?.value || 5),
                recall_k: (parseInt(topKSelect?.value || 5) * 2),
                similarity_threshold: parseFloat(similarityThreshold?.value || 0.3),
                enable_rerank: enableRerank?.checked ?? true,
                enable_query_rewrite: enableQueryRewrite?.checked ?? true,
                enable_memory: enableMemory?.checked ?? true,
                template_name: 'detailed'
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const data = JSON.parse(line);

                        if (data.type === 'start') {
                            // 开始信号，保存潜在的session_id
                            if (data.session_id && data.session_id !== 'default') {
                                newSessionId = data.session_id;
                            }
                        } else if (data.type === 'answer') {
                            const chunk = data.content;
                            if (chunk) {
                                if (!hasReceivedFirstChunk) {
                                    hasReceivedFirstChunk = true;
                                    stopThinkingAnimation();

                                    const thinkingDiv = document.getElementById(thinkingId);
                                    if (thinkingDiv) {
                                        thinkingDiv.style.display = 'none';
                                    }

                                    assistantMessageId = createAssistantMessageContainer();
                                }

                                if (assistantMessageId) {
                                    fullResponse += chunk;
                                    updateAssistantMessage(assistantMessageId, fullResponse);
                                    scrollToBottom();
                                }
                            }
                        } else if (data.type === 'info') {
                            if (retrievalResults && data.results_count !== undefined) {
                                retrievalResults.innerHTML = `<div style="font-size: 12px; color: #667eea; padding: 8px; text-align: center;">
                                    找到 ${data.results_count} 个相关文档，正在生成回答...
                                </div>`;
                            }
                        } else if (data.type === 'end') {
                            // 保存session_id
                            if (data.session_id && data.session_id !== 'default') {
                                saveSessionId(data.session_id);
                            } else if (newSessionId && newSessionId !== 'default') {
                                saveSessionId(newSessionId);
                            }

                            if (fullResponse && assistantMessageId) {
                                saveToHistory(question, fullResponse);
                            }

                            if (!hasReceivedFirstChunk) {
                                stopThinkingAnimation();
                                const thinkingDiv = document.getElementById(thinkingId);
                                if (thinkingDiv) {
                                    thinkingDiv.style.display = 'none';
                                }
                                addMessage('assistant', '未收到响应，请稍后重试。');
                            }
                        } else if (data.type === 'error') {
                            const errorMsg = data.content || '未知错误';
                            if (hasReceivedFirstChunk && assistantMessageId) {
                                updateAssistantMessage(assistantMessageId, `错误: ${errorMsg}`);
                            } else {
                                stopThinkingAnimation();
                                const thinkingDiv = document.getElementById(thinkingId);
                                if (thinkingDiv) {
                                    thinkingDiv.style.display = 'none';
                                }
                                addMessage('assistant', `错误: ${errorMsg}`);
                            }
                        }
                    } catch (e) {
                        console.warn('解析 SSE 数据失败:', e, line);
                    }
                }
            }
        }

        if (!hasReceivedFirstChunk) {
            stopThinkingAnimation();
            const thinkingDiv = document.getElementById(thinkingId);
            if (thinkingDiv) {
                thinkingDiv.style.display = 'none';
            }
            addMessage('assistant', '未收到响应，请稍后重试。');
        }

    } catch (error) {
        console.error('流式请求失败:', error);
        stopThinkingAnimation();
        const thinkingDiv = document.getElementById(thinkingId);
        if (thinkingDiv) {
            thinkingDiv.style.display = 'none';
        }
        addMessage('assistant', `网络错误: ${error.message}`);
    }
}

// 创建助手消息容器
function createAssistantMessageContainer() {
    const messageId = `msg_${Date.now()}`;
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = messageId;
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-text"></div>
            <div class="message-meta">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    return messageId;
}

// 更新助手消息内容
function updateAssistantMessage(messageId, content) {
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        const textDiv = messageDiv.querySelector('.message-text');
        if (textDiv) {
            textDiv.innerHTML = formatContent(content);
        }
    }
}

// 获取对话历史
function getChatHistory() {
    const messages = [];
    const messageElements = document.querySelectorAll('.message');
    messageElements.forEach(el => {
        const role = el.classList.contains('user') ? 'user' : 'assistant';
        const contentEl = el.querySelector('.message-text');
        if (contentEl && contentEl.textContent && !contentEl.textContent.startsWith('错误:')) {
            messages.push({ role: role, content: contentEl.textContent });
        }
    });
    return messages.slice(-10);
}

// 保存到历史记录
function saveToHistory(question, answer) {
    try {
        const history = JSON.parse(localStorage.getItem('chat_history') || '[]');
        history.push({ role: 'user', content: question });
        history.push({ role: 'assistant', content: answer });
        while (history.length > 50) history.shift();
        localStorage.setItem('chat_history', JSON.stringify(history));
    } catch (e) {
        console.warn('保存历史失败:', e);
    }
}

// 格式化内容（支持Markdown简单解析）
function formatContent(content) {
    if (!content) return '';

    // 转义HTML
    let html = escapeHtml(content);

    // 代码块
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // 行内代码
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // 粗体
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // 斜体
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // 换行转br
    html = html.replace(/\n/g, '<br>');

    return html;
}

// 显示检索结果
function displayRetrievalResults(results, info) {
    if (!results || results.length === 0) {
        retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">未找到相关文档</div>';
        return;
    }

    let html = '';
    if (info) {
        html += `<div style="font-size: 12px; color: #999; padding: 8px; background: #f8f9fa; border-radius: 8px; margin-bottom: 12px;">
            召回: ${info.total_recalled} | 返回: ${info.total_returned} | 重排序: ${info.enable_rerank ? '启用' : '禁用'}
        </div>`;
    }

    for (let i = 0; i < results.length; i++) {
        const result = results[i];
        const score = (result.score * 100).toFixed(1);
        html += `
            <div class="result-card" onclick="copyToInput('${escapeJs(result.content.substring(0, 200))}')">
                <div class="result-score">📊 相关度: ${score}%</div>
                <div class="result-content">${escapeHtml(result.content.substring(0, 300))}${result.content.length > 300 ? '...' : ''}</div>
                <div class="result-source">📄 ${escapeHtml(result.document_name || '未知文档')}</div>
            </div>
        `;
    }

    retrievalResults.innerHTML = html;
}

// 滚动到底部
function scrollToBottom() {
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// HTML转义
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// JS字符串转义
function escapeJs(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
}

// 全局函数供onclick调用
window.copyToInput = function(text) {
    if (chatInput) {
        chatInput.value = text;
        chatInput.focus();
    }
};

// 切换流式/非流式模式
function setStreamMode(enabled) {
    useStreamMode = enabled;
    console.log(`流式模式: ${enabled ? '开启' : '关闭'}`);
}

// 事件监听
if (sendBtn) {
    sendBtn.addEventListener('click', sendMessage);
}
if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    loadSavedSession();
});