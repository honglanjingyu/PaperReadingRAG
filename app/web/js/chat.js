// app/web/js/chat.js
// 聊天页面主入口 - 仅支持流式输出

import { elements, initElements, state, updateState } from './chat/config.js';
import { loadSavedSession, newSession, saveSessionId, createAndSetNewSession } from './chat/session.js';
import { sendMessageStream } from './chat/stream.js';
import { removeAllThinkingIndicators } from './chat/thinking.js';
import { addMessage } from './chat/messages.js';
import { showRetrievingStatus } from './chat/retrieval.js';
import { showToast } from './chat/utils.js';

// 发送消息 - 统一使用流式
async function sendMessage() {
    const question = elements.chatInput ? elements.chatInput.value.trim() : '';

    console.log('sendMessage called, question:', question, 'isProcessing:', state.isProcessing);

    if (!question || state.isProcessing) return;

    // 确保有有效的 session_id
    if (!state.currentSessionId || state.currentSessionId === 'default') {
        await createAndSetNewSession();
    }

    // 清空输入框
    if (elements.chatInput) {
        elements.chatInput.value = '';
        elements.chatInput.style.height = 'auto';
    }

    // 添加用户消息
    addMessage('user', question);

    // 显示加载状态
    updateState({ isProcessing: true });
    if (elements.sendBtn) elements.sendBtn.disabled = true;

    // 清空之前的检索结果
    showRetrievingStatus();

    try {
        await sendMessageStream(question);
    } catch (error) {
        console.error('发送消息失败:', error);
        removeAllThinkingIndicators();
        addMessage('assistant', `网络错误：${error.message}`);
        if (elements.retrievalResults) {
            elements.retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">检索失败</div>';
        }
    } finally {
        updateState({ isProcessing: false });
        if (elements.sendBtn) elements.sendBtn.disabled = false;
        if (elements.chatInput) elements.chatInput.focus();
    }
}

// 初始化事件监听
function initEventListeners() {
    console.log('初始化事件监听...');
    console.log('sendBtn 元素:', elements.sendBtn);
    console.log('chatInput 元素:', elements.chatInput);

    // 阈值滑块
    if (elements.similarityThreshold && elements.thresholdValue) {
        elements.similarityThreshold.addEventListener('input', () => {
            if (elements.thresholdValue) {
                elements.thresholdValue.textContent = elements.similarityThreshold.value;
            }
        });
    }

    // 自动调整textarea高度
    if (elements.chatInput) {
        elements.chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });
    }

    // 发送按钮
    if (elements.sendBtn) {
        console.log('绑定发送按钮事件');
        elements.sendBtn.addEventListener('click', sendMessage);
    } else {
        console.error('sendBtn 元素未找到！');
    }

    // 回车发送
    if (elements.chatInput) {
        console.log('绑定回车事件');
        elements.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('回车键触发发送');
                sendMessage();
            }
        });
    } else {
        console.error('chatInput 元素未找到！');
    }

    // 新建会话按钮
    if (elements.newSessionBtn) {
        elements.newSessionBtn.addEventListener('click', newSession);
    }
}
// app/web/js/chat.js - 修改 DOMContentLoaded 部分

document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOMContentLoaded 事件触发');

    // 首先显示用户名
    displayCurrentUser();

    // 添加这行：显示用户等级
    await displayUserRole();

    // 检查登录状态 - 增强版
    const token = localStorage.getItem('rag_token');
    if (!token || token === 'null' || token === 'undefined') {
        console.log('未找到 token，跳转到登录页');
        window.location.href = '/login.html';
        return;
    }

    // 验证 token 有效性
    try {
        console.log('验证 token 有效性...');
        const isValid = await verifyToken();
        if (!isValid) {
            console.log('Token 无效，跳转到登录页');
            logout();
            return;
        }
        console.log('Token 验证通过');
    } catch (error) {
        console.error('验证失败:', error);
        logout();
        return;
    }

    // 确保用户名显示
    displayCurrentUser();

    // 初始化 DOM 元素
    initElements();

    // 绑定退出登录按钮事件
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }

    // 检查 DOM 元素是否存在
    console.log('检查 DOM 元素:');
    console.log('  sendBtn:', elements.sendBtn);
    console.log('  chatInput:', elements.chatInput);
    console.log('  messagesContainer:', elements.messagesContainer);

    // 初始化事件监听
    initEventListeners();

    // 加载会话（从 URL 参数或 localStorage）
    await loadSavedSession();
});