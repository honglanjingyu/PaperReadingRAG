// app/web/js/chat/stream.js - 添加模式参数支持
// 流式和非流式请求模块

import { elements, state, updateState, API_BASE } from './config.js';
import {
    addMessage,
    createAssistantMessageContainer,
    updateAssistantMessage,
    getChatHistory,
    saveToLocalHistory
} from './messages.js';
import {
    addThinkingMessage,
    removeAllThinkingIndicators,
    stopThinkingAnimation,
    replaceThinkingWithContent
} from './thinking.js';
import { displayRetrievalResults, showRetrievingStatus } from './retrieval.js';
import { saveSessionId } from './session.js';
import { showToast } from './utils.js';

// 非流式模式（支持模式参数）
export async function sendMessageNormal(question, mode = 'advanced') {
    const thinkingId = addThinkingMessage();

    try {
        // 根据模式选择不同的 API 端点
        const apiEndpoint = mode === 'graph'
            ? `${API_BASE}/chat/graph/ask`
            : `${API_BASE}/chat/ask`;

        const response = await fetch(apiEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                session_id: state.currentSessionId,
                mode: mode,
                top_k: parseInt(elements.topKSelect?.value || 5),
                similarity_threshold: parseFloat(elements.similarityThreshold?.value || 0.3),
                enable_rerank: elements.enableRerank?.checked ?? true,
                enable_query_rewrite: elements.enableQueryRewrite?.checked ?? true,
                enable_memory: elements.enableMemory?.checked ?? true,
                template_name: 'detailed'
            })
        });

        const data = await response.json();

        if (data.success) {
            if (data.session_id && data.session_id !== 'default') {
                saveSessionId(data.session_id);
            }

            let finalAnswer = data.answer;

            // GraphRAG 模式：添加推理路径
            if (mode === 'graph' && data.reasoning_path) {
                finalAnswer = data.reasoning_path + '\n\n' + data.answer;
            }

            replaceThinkingWithContent(thinkingId, finalAnswer);

            if (elements.retrievalResults && data.results) {
                displayRetrievalResults(data.results, data.retrieval_info);
            }

            if (data.rewritten_query && data.rewritten_query !== question && mode !== 'graph') {
                addMessage('system', `✨ Query改写: "${data.rewritten_query}"`, true);
            }
            if (data.has_history) {
                addMessage('system', `📝 已结合对话历史回答问题`, true);
            }
        } else {
            replaceThinkingWithContent(thinkingId, `抱歉，处理您的问题时出错：${data.error || '未知错误'}`);
            if (elements.retrievalResults) {
                elements.retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">检索失败</div>';
            }
        }
    } catch (error) {
        const thinkingDiv = document.getElementById(thinkingId);
        if (thinkingDiv) thinkingDiv.remove();
        throw error;
    }
}

// 流式模式（支持模式参数）
export async function sendMessageStream(question, mode = 'advanced') {
    const thinkingId = addThinkingMessage();
    let assistantMessageId = null;
    let fullResponse = '';
    let hasReceivedFirstChunk = false;
    let newSessionId = null;
    let reasoningPath = '';

    // 根据模式选择不同的 API 端点
    const apiEndpoint = mode === 'graph'
        ? `${API_BASE}/chat/graph/ask/stream`
        : `${API_BASE}/chat/ask/stream`;

    try {
        const response = await fetch(apiEndpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                session_id: state.currentSessionId,
                mode: mode,
                history: getChatHistory(),
                top_k: parseInt(elements.topKSelect?.value || 5),
                recall_k: (parseInt(elements.topKSelect?.value || 5) * 2),
                similarity_threshold: parseFloat(elements.similarityThreshold?.value || 0.3),
                enable_rerank: elements.enableRerank?.checked ?? true,
                enable_query_rewrite: elements.enableQueryRewrite?.checked ?? true,
                enable_memory: elements.enableMemory?.checked ?? true,
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
                            if (data.session_id && data.session_id !== 'default') {
                                newSessionId = data.session_id;
                            }
                        } else if (data.type === 'reasoning_path') {
                            // GraphRAG 模式的推理路径
                            reasoningPath = data.content;
                            if (mode === 'graph') {
                                const pathDiv = document.createElement('div');
                                pathDiv.className = 'reasoning-path';
                                pathDiv.innerHTML = `<details>
                                    <summary>🔗 推理路径</summary>
                                    <div style="padding: 12px; background: #f0f4ff; border-radius: 8px; margin: 8px 0; font-family: monospace;">
                                        ${formatContent(reasoningPath)}
                                    </div>
                                </details>`;
                                const thinkingDiv = document.getElementById(thinkingId);
                                if (thinkingDiv) {
                                    thinkingDiv.insertAdjacentElement('afterend', pathDiv);
                                }
                            }
                        } else if (data.type === 'answer') {
                            const chunk = data.content;
                            if (chunk) {
                                if (!hasReceivedFirstChunk) {
                                    hasReceivedFirstChunk = true;
                                    stopThinkingAnimation();
                                    const thinkingDiv = document.getElementById(thinkingId);
                                    if (thinkingDiv) thinkingDiv.style.display = 'none';
                                    assistantMessageId = createAssistantMessageContainer();

                                    // 如果有推理路径，添加到消息中
                                    if (reasoningPath && mode === 'graph') {
                                        const msgDiv = document.getElementById(assistantMessageId);
                                        const reasoningDiv = document.createElement('div');
                                        reasoningDiv.className = 'reasoning-path-block';
                                        reasoningDiv.innerHTML = `<details style="margin-bottom: 12px;">
                                            <summary style="cursor: pointer; color: #4263eb; font-weight: 500;">🔗 推理路径</summary>
                                            <div style="padding: 12px; background: #f0f4ff; border-radius: 8px; margin-top: 8px; font-family: monospace; font-size: 13px;">
                                                ${formatContent(reasoningPath)}
                                            </div>
                                        </details>`;
                                        const contentDiv = msgDiv?.querySelector('.message-content');
                                        if (contentDiv) {
                                            contentDiv.insertBefore(reasoningDiv, contentDiv.firstChild);
                                        }
                                    }
                                }
                                if (assistantMessageId) {
                                    fullResponse += chunk;
                                    updateAssistantMessage(assistantMessageId, fullResponse);
                                }
                            }
                        } else if (data.type === 'info') {
                            if (elements.retrievalResults && data.results_count !== undefined) {
                                elements.retrievalResults.innerHTML = `<div style="font-size: 12px; color: #4263eb; padding: 8px; text-align: center;">
                                    🔍 ${data.content || '正在检索...'}
                                </div>`;
                            }
                        } else if (data.type === 'retrieval_results') {
                            if (elements.retrievalResults && data.results) {
                                displayRetrievalResults(data.results, data.retrieval_info);
                            }
                        } else if (data.type === 'end') {
                            if (data.session_id && data.session_id !== 'default') {
                                saveSessionId(data.session_id);
                            } else if (newSessionId && newSessionId !== 'default') {
                                saveSessionId(newSessionId);
                            }
                            if (fullResponse && assistantMessageId) {
                                saveToLocalHistory(question, fullResponse);
                            }
                            if (!hasReceivedFirstChunk) {
                                stopThinkingAnimation();
                                const thinkingDiv = document.getElementById(thinkingId);
                                if (thinkingDiv) thinkingDiv.style.display = 'none';
                                addMessage('assistant', '未收到响应，请稍后重试。');
                            }
                        } else if (data.type === 'error') {
                            const errorMsg = data.content || '未知错误';
                            if (hasReceivedFirstChunk && assistantMessageId) {
                                updateAssistantMessage(assistantMessageId, `错误: ${errorMsg}`);
                            } else {
                                stopThinkingAnimation();
                                const thinkingDiv = document.getElementById(thinkingId);
                                if (thinkingDiv) thinkingDiv.style.display = 'none';
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
            if (thinkingDiv) thinkingDiv.style.display = 'none';
            addMessage('assistant', '未收到响应，请稍后重试。');
        }

    } catch (error) {
        console.error('流式请求失败:', error);
        stopThinkingAnimation();
        const thinkingDiv = document.getElementById(thinkingId);
        if (thinkingDiv) thinkingDiv.style.display = 'none';
        addMessage('assistant', `网络错误: ${error.message}`);
    }
}