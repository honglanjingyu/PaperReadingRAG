// app/web/js/graph/graph-data.js - 数据加载模块
// 图谱数据加载

import {  showToast } from '../chat/utils.js';

const API_BASE = '/api';

export let graphData = null;

// 节点颜色映射
export const NODE_COLORS = {
    'PERSON': { color: '#4caf50', borderColor: '#2e7d32' },
    'ORGANIZATION': { color: '#2196f3', borderColor: '#1565c0' },
    'LOCATION': { color: '#ff9800', borderColor: '#c66900' },
    'CONCEPT': { color: '#9c27b0', borderColor: '#6a1b9a' },
    'PRODUCT': { color: '#f44336', borderColor: '#c62828' },
    'DATE': { color: '#607d8b', borderColor: '#37474f' },
    'NUMBER': { color: '#00bcd4', borderColor: '#00838f' },
    'default': { color: '#757575', borderColor: '#424242' }
};

export async function loadGraphData(forceRebuild = false) {
    try {
        console.log('开始构建知识图谱, forceRebuild=', forceRebuild);

        const buildResponse = await fetch(`${API_BASE}/chat/graph/build`, {
            method: 'POST',
            headers: {
                ...getAuthHeaders(),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ all_documents: true,force_rebuild: forceRebuild })
        });

        if (!buildResponse.ok) {
            throw new Error(`HTTP ${buildResponse.status}`);
        }

        const buildData = await buildResponse.json();
        console.log('图谱构建响应:', buildData);

        // 关键修复：检查响应结构并正确返回数据
        if (buildData && buildData.success) {
            // 确保数据结构完整
            graphData = {
                success: true,
                statistics: buildData.statistics || {
                    entity_count: buildData.entities?.length || 0,
                    relation_count: buildData.relations?.length || 0,
                    community_count: buildData.communities?.length || 0
                },
                entities: buildData.entities || [],
                relations: buildData.relations || [],
                communities: buildData.communities || [],
                total_documents: buildData.total_documents || 0
            };
            return graphData;
        }

        console.error('图谱构建失败:', buildData);
        return null;

    } catch (error) {
        console.error('加载图谱失败:', error);
        throw error;
    }
}

export function updateStats(data, entitySpanId, relationSpanId, communitySpanId, docSpanId) {
    const stats = data.statistics || {};
    const entitySpan = document.getElementById(entitySpanId);
    const relationSpan = document.getElementById(relationSpanId);
    const communitySpan = document.getElementById(communitySpanId);
    const docSpan = document.getElementById(docSpanId);

    if (entitySpan) entitySpan.textContent = stats.entity_count || 0;
    if (relationSpan) relationSpan.textContent = stats.relation_count || 0;
    if (communitySpan) communitySpan.textContent = stats.community_count || 0;
    if (docSpan) docSpan.textContent = data.total_documents || 0;
}

export function setGraphData(data) {
    graphData = data;
}

export function getEntityCategory(type) {
    const categoryMap = {
        'PERSON': 0, 'ORGANIZATION': 1, 'LOCATION': 2,
        'CONCEPT': 3, 'PRODUCT': 4, 'DATE': 5, 'NUMBER': 6
    };
    return categoryMap[type] !== undefined ? categoryMap[type] : 7;
}

export function getCategoryName(categoryIndex) {
    const categories = [
        '👤 人物', '🏢 组织', '📍 地点',
        '💡 概念', '📦 产品', '📅 日期', '🔢 数值', '📌 其他'
    ];
    return categories[categoryIndex] || '📌 其他';
}

export function getCategoryColor(categoryIndex) {
    const colors = [
        '#4caf50', '#2196f3', '#ff9800',
        '#9c27b0', '#f44336', '#607d8b', '#00bcd4', '#757575'
    ];
    return colors[categoryIndex] || '#757575';
}