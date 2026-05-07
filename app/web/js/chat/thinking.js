/* app/web/js/chat/thinking.js */
// 思考动画模块

import { elements, state, updateState } from './config.js';
import { scrollToBottom, formatContent } from './utils.js';

// 添加思考动画消息
export function addThinkingMessage() {
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
    elements.messagesContainer.appendChild(messageDiv);
    startThinkingAnimation(messageId);
    scrollToBottom(elements.messagesContainer);
    return messageId;
}

// 启动思考动画
function startThinkingAnimation(messageId) {
    if (state.thinkingAnimationInterval) {
        clearInterval(state.thinkingAnimationInterval);
    }

    let dotCount = 1;
    let increasing = true;

    const interval = setInterval(() => {
        const messageDiv = document.getElementById(messageId);
        if (!messageDiv) {
            clearInterval(interval);
            return;
        }

        const dotsSpan = messageDiv.querySelector('.thinking-dots');
        if (!dotsSpan) return;

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

        dotsSpan.textContent = '.'.repeat(dotCount);
    }, 400);

    updateState({ thinkingAnimationInterval: interval });
}

// 停止思考动画
export function stopThinkingAnimation() {
    if (state.thinkingAnimationInterval) {
        clearInterval(state.thinkingAnimationInterval);
        updateState({ thinkingAnimationInterval: null });
    }
}

// 移除所有思考指示器
export function removeAllThinkingIndicators() {
    stopThinkingAnimation();
    const indicators = document.querySelectorAll('[id^="thinking_"], [id^="loading_"]');
    indicators.forEach(el => el.remove());
}

// 将思考消息替换为实际内容
export function replaceThinkingWithContent(thinkingMessageId, content) {
    stopThinkingAnimation();

    const thinkingDiv = document.getElementById(thinkingMessageId);
    if (!thinkingDiv) {
        return false;
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

    thinkingDiv.id = `msg_${Date.now()}`;
    return true;
}