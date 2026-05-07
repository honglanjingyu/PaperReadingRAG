/* app/web/js/chat/messages.js */
// 消息渲染模块

import { elements } from './config.js';
import { scrollToBottom, formatContent, escapeHtml } from './utils.js';

// 添加消息（支持系统消息）
export function addMessage(role, content, isSystem = false) {
    const messageDiv = document.createElement('div');

    if (isSystem) {
        // 系统消息样式
        messageDiv.className = 'message assistant';
        messageDiv.innerHTML = `
            <div class="message-avatar">ℹ️</div>
            <div class="message-content" style="background: #e7f3ff; font-size: 12px;">
                <div class="message-text">${escapeHtml(content)}</div>
                <div class="message-meta">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
    } else {
        // 普通用户/助手消息
        messageDiv.className = `message ${role}`;
        const avatar = role === 'user' ? '👤' : '🤖';
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <div class="message-text">${formatContent(content)}</div>
                <div class="message-meta">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
    }

    elements.messagesContainer.appendChild(messageDiv);
    scrollToBottom(elements.messagesContainer);
    return messageDiv;
}

// 添加系统消息（使用专门样式）
export function addSystemMessage(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-avatar">ℹ️</div>
        <div class="message-content" style="background: #e7f3ff; font-size: 12px;">
            <div class="message-text">${escapeHtml(content)}</div>
            <div class="message-meta">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    elements.messagesContainer.appendChild(messageDiv);
    scrollToBottom(elements.messagesContainer);
    return messageDiv;
}

// 添加欢迎消息
export function addWelcomeMessage() {
    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'message assistant';
    welcomeDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-text">您好！我是RAG智能问答助手。请上传文档后，向我提问任何关于文档内容的问题。</div>
            <div class="message-meta">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    elements.messagesContainer.appendChild(welcomeDiv);
    scrollToBottom(elements.messagesContainer);
}

// 清空所有消息
export function clearAllMessages() {
    const messages = document.querySelectorAll('.message');
    messages.forEach(msg => msg.remove());
}

// 创建助手消息容器（用于流式输出）
export function createAssistantMessageContainer() {
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
    elements.messagesContainer.appendChild(messageDiv);
    scrollToBottom(elements.messagesContainer);
    return messageId;
}

// 更新助手消息内容（流式）
export function updateAssistantMessage(messageId, content) {
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        const textDiv = messageDiv.querySelector('.message-text');
        if (textDiv) {
            textDiv.innerHTML = formatContent(content);
            scrollToBottom(elements.messagesContainer);
        }
    }
}

// 获取对话历史
export function getChatHistory() {
    const messages = [];
    const messageElements = document.querySelectorAll('.message');
    messageElements.forEach(el => {
        // 跳过系统消息
        if (el.querySelector('.message-content[style*="#e7f3ff"]')) {
            return;
        }
        const role = el.classList.contains('user') ? 'user' : 'assistant';
        const contentEl = el.querySelector('.message-text');
        if (contentEl && contentEl.textContent && !contentEl.textContent.startsWith('错误:')) {
            messages.push({ role: role, content: contentEl.textContent });
        }
    });
    return messages.slice(-10);
}

// 保存到本地历史记录
export function saveToLocalHistory(question, answer) {
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

export { scrollToBottom } from './utils.js';