// app/web/js/chat/retrieval.js
// 检索结果显示模块 - 恢复备份样式（简化的卡片形式）

import { elements } from './config.js';
import { escapeHtml, escapeJs } from './utils.js';

// 显示检索结果（备份样式：简洁卡片）
export function displayRetrievalResults(results, info) {
    if (!results || results.length === 0) {
        elements.retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">未找到相关文档</div>';
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
            <div class="result-card" onclick="window.copyToInput('${escapeJs(result.content.substring(0, 200))}')">
                <div class="result-score">📊 相关度: ${score}%</div>
                <div class="result-content">${escapeHtml(result.content.substring(0, 300))}${result.content.length > 300 ? '...' : ''}</div>
                <div class="result-source">📄 ${escapeHtml(result.document_name || '未知文档')}</div>
            </div>
        `;
    }

    elements.retrievalResults.innerHTML = html;
}

// 显示检索中状态
export function showRetrievingStatus() {
    if (elements.retrievalResults) {
        elements.retrievalResults.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">正在检索...</div>';
    }
}

// 全局复制函数
window.copyToInput = function(text) {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.value = text;
        chatInput.focus();
    }
};