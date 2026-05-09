/* app/web/js/chat/stream.js */
// 流式和非流式请求模块

import { elements, state, API_BASE } from './config.js';
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

// 非流式模式
export async function sendMessageNormal(question) {
    const thinkingId = addThinkingMessage();

    try {
        const response = await fetch(`${API_BASE}/chat/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                session_id: state.currentSessionId,
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
            replaceThinkingWithContent(thinkingId, data.answer);
            if (elements.retrievalResults) {
                displayRetrievalResults(data.results, data.retrieval_info);
            }
            if (data.rewritten_query && data.rewritten_query !== question) {
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

// 流式模式
export async function sendMessageStream(question) {
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
                session_id: state.currentSessionId,
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
                        } else if (data.type === 'retrieval_results') {
                            // 显示检索结果
                            if (elements.retrievalResults && data.results) {
                                displayRetrievalResults(data.results, data.retrieval_info);

                                // 如果没有检索结果，显示提示
                                if (!data.results || data.results.length === 0) {
                                    elements.retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">未找到相关文档</div>';
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
                                }
                                if (assistantMessageId) {
                                    fullResponse += chunk;
                                    updateAssistantMessage(assistantMessageId, fullResponse);
                                }
                            }
                        } else if (data.type === 'info') {
                            if (elements.retrievalResults && data.results_count !== undefined) {
                                elements.retrievalResults.innerHTML = `<div style="font-size: 12px; color: #667eea; padding: 8px; text-align: center;">
                                    找到 ${data.results_count} 个相关文档，正在生成回答...
                                </div>`;
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
                                if (data.no_results) {
                                    addMessage('assistant', '未找到与问题相关的文档内容，请尝试其他问题或上传更多相关文档。');
                                } else {
                                    addMessage('assistant', '未收到响应，请稍后重试。');
                                }
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
                            if (elements.retrievalResults) {
                                elements.retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">检索失败</div>';
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
        }

    } catch (error) {
        console.error('流式请求失败:', error);
        stopThinkingAnimation();
        const thinkingDiv = document.getElementById(thinkingId);
        if (thinkingDiv) thinkingDiv.style.display = 'none';
        addMessage('assistant', `网络错误: ${error.message}`);
        if (elements.retrievalResults) {
            elements.retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">检索失败</div>';
        }
    }
}