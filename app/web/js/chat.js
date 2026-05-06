/* app/web/js/chat.js */
/* 聊天页面逻辑 - 支持流式输出，包含思考动画 */

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

let isProcessing = false;
let useStreamMode = true;  // 默认使用流式输出
let thinkingAnimationInterval = null;  // 思考动画定时器

// 显示阈值
similarityThreshold?.addEventListener('input', () => {
    thresholdValue.textContent = similarityThreshold.value;
});

// 自动调整textarea高度
chatInput?.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 100) + 'px';
});

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

// 添加思考动画消息（三个闪烁的点）
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

    // 启动点动画
    startThinkingAnimation(messageId);

    scrollToBottom();
    return messageId;
}

// 启动思考动画（... 循环闪烁）
function startThinkingAnimation(messageId) {
    // 清除之前的动画
    if (thinkingAnimationInterval) {
        clearInterval(thinkingAnimationInterval);
    }

    const messageDiv = document.getElementById(messageId);
    if (!messageDiv) return;

    const dotsSpan = messageDiv.querySelector('.thinking-dots');
    if (!dotsSpan) return;

    let dotCount = 3;
    let increasing = false;

    thinkingAnimationInterval = setInterval(() => {
        const currentDiv = document.getElementById(messageId);
        if (!currentDiv) {
            // 消息已被替换，清除动画
            if (thinkingAnimationInterval) {
                clearInterval(thinkingAnimationInterval);
                thinkingAnimationInterval = null;
            }
            return;
        }

        const currentDotsSpan = currentDiv.querySelector('.thinking-dots');
        if (!currentDotsSpan) return;

        // 更新点数（1到3之间循环）
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

        // 显示对应的点
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

// 将思考消息替换为实际内容
function replaceThinkingWithContent(thinkingMessageId, content) {
    stopThinkingAnimation();

    const thinkingDiv = document.getElementById(thinkingMessageId);
    if (!thinkingDiv) {
        // 如果思考消息不存在，直接创建新消息
        return addMessageAndReturnId('assistant', content);
    }

    // 移除思考样式
    thinkingDiv.classList.remove('thinking');

    // 更新内容
    const textDiv = thinkingDiv.querySelector('.message-text');
    if (textDiv) {
        textDiv.innerHTML = formatContent(content);
    }

    // 更新时间
    const metaSpan = thinkingDiv.querySelector('.message-meta');
    if (metaSpan) {
        metaSpan.textContent = new Date().toLocaleTimeString();
    }

    // 修改ID
    const newId = `msg_${Date.now()}`;
    thinkingDiv.id = newId;

    return newId;
}

// 移除所有思考指示器
function removeAllThinkingIndicators() {
    stopThinkingAnimation();
    const indicators = document.querySelectorAll('[id^="thinking_"], [id^="loading_"]');
    indicators.forEach(el => el.remove());
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

// ========== 非流式模式 ==========

async function sendMessageNormal(question) {
    const thinkingId = addThinkingMessage();

    try {
        const response = await fetch(`${API_BASE}/chat/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                top_k: parseInt(topKSelect?.value || 5),
                similarity_threshold: parseFloat(similarityThreshold?.value || 0.3),
                enable_rerank: enableRerank?.checked ?? true,
                enable_query_rewrite: enableQueryRewrite?.checked ?? true,
                template_name: 'detailed'
            })
        });

        const data = await response.json();

        if (data.success) {
            replaceThinkingWithContent(thinkingId, data.answer);
            if (retrievalResults) {
                displayRetrievalResults(data.results, data.retrieval_info);
            }
            if (data.rewritten_query && data.rewritten_query !== question) {
                addSystemMessage(`✨ Query改写: "${data.rewritten_query}"`);
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

function removeThinkingIndicator(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

// ========== 流式模式 ==========

async function sendMessageStream(question) {
    // 添加思考动画
    const thinkingId = addThinkingMessage();
    let assistantMessageId = null;
    let fullResponse = '';
    let hasReceivedFirstChunk = false;

    try {
        const response = await fetch(`${API_BASE}/chat/ask/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                history: getChatHistory(),
                top_k: parseInt(topKSelect?.value || 5),
                recall_k: (parseInt(topKSelect?.value || 5) * 2),
                similarity_threshold: parseFloat(similarityThreshold?.value || 0.3),
                enable_rerank: enableRerank?.checked ?? true,
                enable_query_rewrite: enableQueryRewrite?.checked ?? true,
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
                            // 收到开始信号，但保留思考动画直到收到第一个字符
                        } else if (data.type === 'answer') {
                            const chunk = data.content;
                            if (chunk) {
                                // 收到第一个字符，停止思考动画并创建真实消息
                                if (!hasReceivedFirstChunk) {
                                    hasReceivedFirstChunk = true;
                                    stopThinkingAnimation();

                                    // 隐藏思考消息（或替换为真实消息）
                                    const thinkingDiv = document.getElementById(thinkingId);
                                    if (thinkingDiv) {
                                        thinkingDiv.style.display = 'none';
                                    }

                                    // 创建新的消息容器
                                    assistantMessageId = createAssistantMessageContainer();
                                }

                                if (assistantMessageId) {
                                    fullResponse += chunk;
                                    updateAssistantMessage(assistantMessageId, fullResponse);
                                    scrollToBottom();
                                }
                            }
                        } else if (data.type === 'info') {
                            // 显示检索信息
                            if (retrievalResults && data.results_count !== undefined) {
                                retrievalResults.innerHTML = `<div style="font-size: 12px; color: #667eea; padding: 8px; text-align: center;">
                                    找到 ${data.results_count} 个相关文档，正在生成回答...
                                </div>`;
                            }
                        } else if (data.type === 'end') {
                            // 流结束
                            if (fullResponse && assistantMessageId) {
                                saveToHistory(question, fullResponse);
                            }
                            // 如果没有收到任何字符
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

        // 如果从未收到任何字符
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

function scrollToBottom() {
    if (messagesContainer) {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

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
sendBtn?.addEventListener('click', sendMessage);
chatInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});