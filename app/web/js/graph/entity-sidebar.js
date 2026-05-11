// app/web/js/graph/entity-sidebar.js - 实体详情侧边栏
// 实体详情侧边栏模块

import { escapeHtml } from '../chat/utils.js';
import { getEntityCategory } from './graph-data.js';

export function showEntityDetail(entity, graphData, sidebar, detailContainer, relatedContainer, onRelatedClick) {
    if (!detailContainer || !relatedContainer) return;

    if (sidebar) sidebar.style.display = 'flex';

    let entityTypeLabel = '';
    let typeIcon = '';
    switch (entity.type) {
        case 'PERSON': entityTypeLabel = '👤 人物'; typeIcon = '👤'; break;
        case 'ORGANIZATION': entityTypeLabel = '🏢 组织'; typeIcon = '🏢'; break;
        case 'LOCATION': entityTypeLabel = '📍 地点'; typeIcon = '📍'; break;
        case 'CONCEPT': entityTypeLabel = '💡 概念'; typeIcon = '💡'; break;
        case 'PRODUCT': entityTypeLabel = '📦 产品'; typeIcon = '📦'; break;
        case 'DATE': entityTypeLabel = '📅 日期'; typeIcon = '📅'; break;
        case 'NUMBER': entityTypeLabel = '🔢 数值'; typeIcon = '🔢'; break;
        default: entityTypeLabel = '📌 其他'; typeIcon = '📌';
    }

    const degree = entity.degree || 0;
    const freq = entity.frequency || entity.value || 1;

    detailContainer.innerHTML = `
        <div class="entity-name">${typeIcon} ${escapeHtml(entity.name)}</div>
        <span class="entity-type">${entityTypeLabel}</span>
        <div class="entity-frequency">📊 出现频率: ${freq} 次</div>
        <div class="entity-frequency">🔗 连接数: ${degree} 个关联</div>
        <div class="entity-description">${entity.summary || '暂无描述信息'}</div>
    `;

    const relations = graphData?.relations || [];
    const related = relations.filter(r => r.source === entity.name || r.target === entity.name);

    related.sort((a, b) => (b.weight || 0) - (a.weight || 0));

    if (related.length === 0) {
        relatedContainer.innerHTML = `
            <div class="related-title">🔗 相关实体 (0)</div>
            <div style="color: #adb5bd; font-size: 13px; padding: 12px;">无关联实体</div>
        `;
        return;
    }

    let relatedHtml = `<div class="related-title">🔗 相关实体 (${related.length})</div><ul class="related-list">`;
    related.forEach(rel => {
        const relatedName = rel.source === entity.name ? rel.target : rel.source;
        const relationType = rel.relation_type || '相关';
        const weight = rel.weight || 1;
        relatedHtml += `
            <li class="related-item" data-entity="${escapeHtml(relatedName)}">
                <span class="related-name">${escapeHtml(relatedName)}</span>
                <span class="related-relation">${escapeHtml(relationType)} (权重: ${weight})</span>
            </li>
        `;
    });
    relatedHtml += `</ul>`;

    relatedContainer.innerHTML = relatedHtml;

    document.querySelectorAll('.related-item').forEach(item => {
        item.addEventListener('click', () => {
            const entityName = item.dataset.entity;
            const targetEntity = graphData?.entities?.find(e => e.name === entityName);
            if (targetEntity && onRelatedClick) {
                // 递归显示实体详情
                showEntityDetail(targetEntity, graphData, sidebar, detailContainer, relatedContainer, onRelatedClick);
                if (onRelatedClick) onRelatedClick(entityName);
            }
        });
    });
}