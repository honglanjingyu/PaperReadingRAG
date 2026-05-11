// app/web/js/chat.js
// 聊天页面主入口 - 支持 URL session 参数，添加模式切换功能

import { elements, initElements, state, updateState } from './chat/config.js';
import { loadSavedSession, newSession, saveSessionId, createAndSetNewSession } from './chat/session.js';
import { sendMessageNormal, sendMessageStream } from './chat/stream.js';
import { removeAllThinkingIndicators } from './chat/thinking.js';
import { addMessage } from './chat/messages.js';
import { showRetrievingStatus } from './chat/retrieval.js';
import { showToast } from './chat/utils.js';

// 当前选中的模式
let currentMode = 'advanced'; // 'advanced' 或 'graph'

// 切换模式
function switchMode(mode) {
    if (state.isProcessing) {
        showToast('请等待当前回答完成后再切换模式', 'warning');
        return;
    }

    currentMode = mode;

    // 更新按钮样式
    const advancedBtn = document.getElementById('advancedModeBtn');
    const graphBtn = document.getElementById('graphModeBtn');

    if (advancedBtn && graphBtn) {
        if (mode === 'advanced') {
            advancedBtn.classList.add('active');
            graphBtn.classList.remove('active');
        } else {
            advancedBtn.classList.remove('active');
            graphBtn.classList.add('active');
        }
    }

    // 保存模式到 localStorage（静默）
    localStorage.setItem('rag_chat_mode', mode);
}

// 发送消息 - 主入口（确保 session_id 存在，并传递模式参数）
async function sendMessage() {
    const question = elements.chatInput ? elements.chatInput.value.trim() : '';

    console.log('sendMessage called, question:', question, 'isProcessing:', state.isProcessing, 'mode:', currentMode);

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
        if (state.useStreamMode) {
            await sendMessageStream(question, currentMode);
        } else {
            await sendMessageNormal(question, currentMode);
        }
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

    // ========== 模式切换按钮事件 ==========
    const advancedModeBtn = document.getElementById('advancedModeBtn');
    const graphModeBtn = document.getElementById('graphModeBtn');

    if (advancedModeBtn) {
        advancedModeBtn.addEventListener('click', () => switchMode('advanced'));
    }

    if (graphModeBtn) {
        graphModeBtn.addEventListener('click', () => switchMode('graph'));
    }
}

// 加载保存的模式
function loadSavedMode() {
    const savedMode = localStorage.getItem('rag_chat_mode');
    if (savedMode === 'graph') {
        switchMode('graph');
    } else {
        switchMode('advanced');
    }
}

// 切换流式/非流式模式
window.setStreamMode = function(enabled) {
    updateState({ useStreamMode: enabled });
    console.log(`流式模式: ${enabled ? '开启' : '关闭'}`);
};

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOMContentLoaded 事件触发');

    // 先初始化 DOM 元素
    initElements();

    // 检查 DOM 元素是否存在
    console.log('检查 DOM 元素:');
    console.log('  sendBtn:', elements.sendBtn);
    console.log('  chatInput:', elements.chatInput);
    console.log('  messagesContainer:', elements.messagesContainer);

    // 初始化事件监听
    initEventListeners();

    // 加载保存的模式
    loadSavedMode();

    // 加载会话（从 URL 参数或 localStorage）
    await loadSavedSession();
});