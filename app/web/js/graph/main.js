// app/web/js/graph/main.js - 主入口模块
// 知识图谱页面主入口

import { showToast } from '../chat/utils.js';
import { initGraph, resizeGraph } from './graph-core.js';
import { loadGraphData, updateStats, graphData, setGraphData } from './graph-data.js';
import { renderGraph, renderCommunities, highlightCommunity } from './graph-render.js';
import { showEntityDetail } from './entity-sidebar.js';
import { initCollapsiblePanels } from './collapsible.js';

const API_BASE = '/api';

// 状态管理
let graphChart = null;
let currentLayout = 'force';
let currentDisplayMode = 'entity';
let selectedEntity = null;

// DOM 元素
let buildGraphBtn, rebuildGraphBtn, layoutType, displayMode;
let graphStatusDot, graphStatusText;
let graphCanvas, graphLoading;
let communitiesPanel, communitiesList;
let entitySidebar, entityDetail, relatedEntities;
let toggleCommunitiesBtn, closeSidebarBtn;
let statsCards, statsHeader, collapseStatsBtn;

// 初始化
// 在 DOMContentLoaded 事件中，删除或注释掉 loadGraphDataUI(false) 调用

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Graph page initializing...');

    displayCurrentUser();
    await displayUserRole();

    const token = localStorage.getItem('rag_token');
    if (!token || token === 'null') {
        window.location.href = '/login.html';
        return;
    }

    const isValid = await verifyToken();
    if (!isValid) {
        logout();
        return;
    }

    initElements();
    bindEvents();
    initGraphUI();
    initCollapsiblePanels();
    await autoLoadGraphWithCache();
});

// app/web/js/graph/main.js

// 在 showEmptyGraphState 函数之后添加此函数

async function autoLoadGraphWithCache() {
    if (!graphLoading) return;

    console.log('自动加载知识图谱（使用缓存）...');

    // 显示加载状态
    graphLoading.style.display = 'flex';
    updateGraphStatus('loading', '正在加载知识图谱...');

    try {
        // 先检查缓存状态
        const statusResponse = await fetch(`${API_BASE}/chat/graph/status`, {
            headers: getAuthHeaders()
        });

        let hasCache = false;
        let cacheInfo = null;

        if (statusResponse.ok) {
            const statusData = await statusResponse.json();
            hasCache = statusData.has_cache;
            cacheInfo = statusData.cache_info;
            console.log('图谱缓存状态:', { hasCache, cacheInfo });
        }

        // 如果有缓存，直接加载（force_rebuild = false）
        if (hasCache) {
            console.log('发现图谱缓存，正在加载...');
            const result = await loadGraphData(false);  // force_rebuild = false

            if (result && result.success) {
                setGraphData(result);

                console.log('缓存图谱加载成功:', {
                    entities: result.entities?.length,
                    relations: result.relations?.length,
                    communities: result.communities?.length,
                    cached_at: result._cached_at
                });

                updateStatsUI(result);
                renderGraphUI();
                renderCommunitiesUI(result);

                const cacheTime = result._cached_at ? new Date(result._cached_at).toLocaleString() : '未知';
                updateGraphStatus('success', `图谱已从缓存加载 (${result.statistics?.entity_count || 0} 实体, 缓存时间: ${cacheTime})`);

                if (statsCards) statsCards.style.display = 'grid';
                if (communitiesPanel) communitiesPanel.style.display = 'block';

                showToast('知识图谱已从缓存加载', 'success', 2000);
                return;
            }
        }

        // 没有缓存，尝试构建（force_rebuild = false，会使用缓存构建）
        console.log('没有缓存或缓存无效，正在构建知识图谱...');
        const result = await loadGraphData(false);

        if (result && result.success) {
            setGraphData(result);

            updateStatsUI(result);
            renderGraphUI();
            renderCommunitiesUI(result);

            updateGraphStatus('success', `图谱已构建 (${result.statistics?.entity_count || 0} 实体, ${result.statistics?.relation_count || 0} 关系)`);

            if (statsCards) statsCards.style.display = 'grid';
            if (communitiesPanel) communitiesPanel.style.display = 'block';

            showToast('知识图谱构建成功', 'success');
        } else {
            // 知识库为空
            updateGraphStatus('idle', result?.error || '知识库为空，请先上传文档');
            showEmptyGraphState();
        }

    } catch (error) {
        console.error('自动加载图谱失败:', error);
        updateGraphStatus('error', '加载失败: ' + error.message);
        showEmptyGraphState();
    } finally {
        if (graphLoading) graphLoading.style.display = 'none';
    }
}

// 显示空图谱状态
function showEmptyGraphState() {
    if (graphChart) {
        graphChart.setOption({
            title: {
                show: true,
                text: '点击「构建/刷新图谱」按钮开始\n知识图谱将基于已上传文档自动构建',
                left: 'center',
                top: 'center',
                textStyle: { color: '#adb5bd', fontSize: 14, fontWeight: 'normal' }
            },
            backgroundColor: 'transparent'
        });
    }

    if (communitiesList) {
        communitiesList.innerHTML = '<div style="text-align: center; padding: 20px; color: #adb5bd;">点击构建按钮加载社区数据</div>';
    }

    updateGraphStatus('idle', '未加载，点击构建');
}

function initElements() {
    buildGraphBtn = document.getElementById('buildGraphBtn');
    rebuildGraphBtn = document.getElementById('rebuildGraphBtn');
    layoutType = document.getElementById('layoutType');
    displayMode = document.getElementById('displayMode');

    graphStatusDot = document.getElementById('graphStatusDot');
    graphStatusText = document.getElementById('graphStatusText');

    graphCanvas = document.getElementById('graphCanvas');
    graphLoading = document.getElementById('graphLoading');

    communitiesPanel = document.getElementById('communitiesPanel');
    communitiesList = document.getElementById('communitiesList');

    entitySidebar = document.getElementById('entitySidebar');
    entityDetail = document.getElementById('entityDetail');
    relatedEntities = document.getElementById('relatedEntities');

    toggleCommunitiesBtn = document.getElementById('toggleCommunitiesBtn');
    closeSidebarBtn = document.getElementById('closeSidebarBtn');
    statsCards = document.getElementById('statsCards');
}

// app/web/js/graph/main.js

// 在 initElements() 后面添加
document.addEventListener('visibilitychange', async () => {
    // 当页面从隐藏变为可见时（从其他标签页切回）
    if (!document.hidden && graphChart) {
        console.log('页面重新可见，检查图谱缓存...');

        // 静默检查缓存，不显示加载动画
        try {
            const statusResponse = await fetch(`${API_BASE}/chat/graph/status`, {
                headers: getAuthHeaders()
            });

            if (statusResponse.ok) {
                const statusData = await statusResponse.json();

                // 如果有缓存且当前图谱数据为空或过期，重新加载
                if (statusData.has_cache && (!graphData || statusData.cache_info?.is_cached)) {
                    console.log('检测到缓存，重新加载图谱...');

                    // 静默重新加载（不显示加载动画的变体）
                    const result = await loadGraphData(false);
                    if (result && result.success) {
                        setGraphData(result);
                        updateStatsUI(result);
                        renderGraphUI();
                        renderCommunitiesUI(result);
                        updateGraphStatus('success', `图谱已更新 (${result.statistics?.entity_count || 0} 实体)`);
                    }
                }
            }
        } catch (error) {
            console.warn('页面恢复时检查缓存失败:', error);
        }
    }
});

function bindEvents() {
    if (buildGraphBtn) {
        buildGraphBtn.addEventListener('click', () => loadGraphDataUI(false));
    }
    if (rebuildGraphBtn) {
        rebuildGraphBtn.addEventListener('click', () => loadGraphDataUI(true));
    }
    if (layoutType) {
        layoutType.addEventListener('change', () => {
            currentLayout = layoutType.value;
            updateGraphLayout();
        });
    }
    if (displayMode) {
        displayMode.addEventListener('change', () => {
            currentDisplayMode = displayMode.value;
            renderGraphUI();
        });
    }
    if (closeSidebarBtn) {
        closeSidebarBtn.addEventListener('click', () => {
            if (entitySidebar) entitySidebar.style.display = 'none';
        });
    }

    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
}


function initGraphUI() {
    if (!graphCanvas) {
        console.error('graphCanvas element not found');
        return;
    }

    graphChart = initGraph(graphCanvas);

    // 注册缩放事件
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            if (graphChart) resizeGraph(graphChart);
        }, 200);
    });
}
// app/web/js/graph/main.js - 修改 loadGraphDataUI

async function loadGraphDataUI(forceRebuild = false) {
    if (!graphLoading) return;

    graphLoading.style.display = 'flex';
    updateGraphStatus('loading', '正在构建知识图谱...');

    try {
        // 直接调用 loadGraphData
        const result = await loadGraphData(forceRebuild);

        if (result && result.success) {
            // 确保数据已设置
            setGraphData(result);

            console.log('图谱数据:', {
                entities: result.entities?.length,
                relations: result.relations?.length,
                communities: result.communities?.length
            });

            updateStatsUI(result);
            renderGraphUI();
            renderCommunitiesUI(result);

            updateGraphStatus('success', `图谱已加载 (${result.statistics?.entity_count || 0} 实体, ${result.statistics?.relation_count || 0} 关系)`);

            if (statsCards) statsCards.style.display = 'grid';
            if (communitiesPanel) communitiesPanel.style.display = 'block';

            showToast('知识图谱构建成功', 'success');
        } else {
            updateGraphStatus('error', result?.error || '图谱数据为空，请先上传文档');
            showEmptyGraphState();
            showToast(result?.error || '请先上传文档以构建知识图谱', 'warning');
        }

    } catch (error) {
        console.error('加载图谱失败:', error);
        updateGraphStatus('error', '加载失败: ' + error.message);
        showEmptyGraphState();
        showToast('构建知识图谱失败: ' + error.message, 'error');
    } finally {
        if (graphLoading) graphLoading.style.display = 'none';
    }
}
function updateStatsUI(data) {
    updateStats(data, 'entityCount', 'relationCount', 'communityCount', 'docCount');
}

function renderGraphUI() {
    if (!graphChart) {
        console.error('graphChart not initialized');
        return;
    }
    renderGraph(graphChart, currentLayout, currentDisplayMode, (node) => {
        showEntityDetail(node, graphData, entitySidebar, entityDetail, relatedEntities, (entityName) => {
            highlightEntityInGraph(entityName);
        });
    });
}

function renderCommunitiesUI(data) {
    renderCommunities(data, communitiesList, (communityId) => {
        highlightCommunityInGraph(communityId);
    });
}

function updateGraphLayout() {
    if (!graphChart) return;

    if (currentLayout === 'force') {
        // 更新力导向参数
    }
    if (graphChart) {
        const option = graphChart.getOption();
        option.series[0].layout = currentLayout;
        graphChart.setOption(option);
        setTimeout(() => graphChart.resize(), 100);
    }
}

function highlightCommunityInGraph(communityId) {
    highlightCommunity(graphChart, graphData, currentDisplayMode, displayMode, communityId);
}

function highlightEntityInGraph(entityName) {
    if (!graphChart) return;

    const option = graphChart.getOption();
    const series = option.series[0];

    if (series && series.data) {
        const newData = series.data.map(node => {
            if (node.id === entityName || node.name === entityName) {
                return {
                    ...node,
                    itemStyle: {
                        color: '#ff6b6b',
                        borderWidth: 4,
                        borderColor: '#fff',
                        shadowBlur: 15
                    },
                    symbolSize: (node.symbolSize || 15) * 1.2
                };
            }
            return node;
        });
        series.data = newData;
        graphChart.setOption(option);
    }
}
function updateGraphStatus(status, message) {
    if (!graphStatusDot || !graphStatusText) return;

    const dotColors = {
        idle: '#adb5bd',
        loading: '#ffc107',
        success: '#40c057',
        error: '#fa5252'
    };

    graphStatusDot.style.background = dotColors[status] || '#adb5bd';
    graphStatusText.textContent = message;
}

function getAuthHeaders() {
    const token = localStorage.getItem('rag_token');
    if (token && token !== 'null') {
        return { 'Authorization': `Bearer ${token}` };
    }
    return {};
}

async function verifyToken() {
    const token = localStorage.getItem('rag_token');
    if (!token) return false;

    try {
        const response = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: getAuthHeaders()
        });
        if (!response.ok) return false;
        const data = await response.json();
        return data.success === true;
    } catch (error) {
        return false;
    }
}

function displayCurrentUser() {
    const username = localStorage.getItem('rag_username');
    const userNameSpan = document.getElementById('userNameDisplay');
    if (userNameSpan) {
        userNameSpan.textContent = username || '用户';
    }
}

async function displayUserRole() {
    const roleSpan = document.getElementById('userRoleBadge');
    if (!roleSpan) return;

    try {
        const response = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: getAuthHeaders()
        });
        if (response.ok) {
            const data = await response.json();
            const role = data.role || 'normal';
            if (role === 'admin') {
                roleSpan.innerHTML = '👑 管理员';
                roleSpan.className = 'user-role-badge admin';
            } else if (role === 'owner') {
                roleSpan.innerHTML = '⭐ 所有者';
                roleSpan.className = 'user-role-badge owner';
            } else {
                roleSpan.innerHTML = '👤 普通用户';
                roleSpan.className = 'user-role-badge normal';
            }
        }
    } catch (error) {
        roleSpan.innerHTML = '👤 普通用户';
    }
}

function handleLogout() {
    if (confirm('确定要退出登录吗？')) {
        localStorage.removeItem('rag_token');
        localStorage.removeItem('rag_user_id');
        localStorage.removeItem('rag_username');
        window.location.href = '/login.html';
    }
}

function logout() {
    handleLogout();
}

export { graphChart, currentLayout, currentDisplayMode, updateGraphLayout };