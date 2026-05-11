// app/web/js/graph/collapsible.js - 完整代码

export function initCollapsiblePanels() {
    // 社区面板收起/展开
    initCommunityPanelCollapse();
    // 统计卡片收起/展开
    initStatsCardsCollapse();
}

function initCommunityPanelCollapse() {
    const toggleBtn = document.getElementById('toggleCommunitiesBtn');
    const communitiesList = document.getElementById('communitiesList');

    if (!toggleBtn || !communitiesList) {
        console.warn('社区面板元素未找到');
        return;
    }

    // 初始状态：展开
    let isCollapsed = false;

    // 移除旧监听器，避免重复
    const newToggleBtn = toggleBtn.cloneNode(true);
    toggleBtn.parentNode.replaceChild(newToggleBtn, toggleBtn);

    newToggleBtn.addEventListener('click', (e) => {
        e.preventDefault();
        isCollapsed = !isCollapsed;

        if (isCollapsed) {
            communitiesList.style.display = 'none';
            newToggleBtn.textContent = '展开';
        } else {
            communitiesList.style.display = 'grid';
            newToggleBtn.textContent = '收起';
        }

        // 触发窗口大小调整，让图谱容器重新计算大小
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 100);
    });
}

function initStatsCardsCollapse() {
    const statsCards = document.getElementById('statsCards');

    if (!statsCards) {
        console.warn('统计卡片元素未找到');
        return;
    }

    // 查找或创建收起按钮
    let collapseStatsBtn = document.getElementById('collapseStatsBtn');
    let statsHeader = document.querySelector('.stats-header');

    if (!statsHeader) {
        // 创建 stats-header
        statsHeader = document.createElement('div');
        statsHeader.className = 'stats-header';
        statsHeader.innerHTML = `
            <span class="stats-title">📊 统计信息</span>
            <button id="collapseStatsBtn" class="btn-collapse-stats">收起</button>
        `;
        statsCards.parentNode.insertBefore(statsHeader, statsCards);
        collapseStatsBtn = document.getElementById('collapseStatsBtn');
    }

    if (!collapseStatsBtn) {
        collapseStatsBtn = document.querySelector('#collapseStatsBtn');
    }

    if (!collapseStatsBtn) {
        console.warn('统计卡片收起按钮未找到');
        return;
    }

    // 初始状态：展开
    let isCollapsed = false;

    // 移除旧监听器
    const newCollapseBtn = collapseStatsBtn.cloneNode(true);
    collapseStatsBtn.parentNode.replaceChild(newCollapseBtn, collapseStatsBtn);

    newCollapseBtn.addEventListener('click', (e) => {
        e.preventDefault();
        isCollapsed = !isCollapsed;

        if (isCollapsed) {
            statsCards.style.display = 'none';
            newCollapseBtn.textContent = '展开';
        } else {
            statsCards.style.display = 'grid';
            newCollapseBtn.textContent = '收起';
        }

        // 触发窗口大小调整
        setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
        }, 100);
    });
}

export function isCommunityPanelCollapsed() {
    const communitiesList = document.getElementById('communitiesList');
    return communitiesList ? communitiesList.style.display === 'none' : false;
}

export function isStatsCardsCollapsed() {
    const statsCards = document.getElementById('statsCards');
    return statsCards ? statsCards.style.display === 'none' : false;
}

export function expandAllPanels() {
    const communitiesList = document.getElementById('communitiesList');
    const statsCards = document.getElementById('statsCards');
    const toggleBtn = document.getElementById('toggleCommunitiesBtn');
    const collapseStatsBtn = document.getElementById('collapseStatsBtn');

    if (communitiesList && communitiesList.style.display === 'none') {
        communitiesList.style.display = 'grid';
        if (toggleBtn) toggleBtn.textContent = '收起';
    }

    if (statsCards && statsCards.style.display === 'none') {
        statsCards.style.display = 'grid';
        if (collapseStatsBtn) collapseStatsBtn.textContent = '收起';
    }

    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 300);
}

export function collapseAllPanels() {
    const communitiesList = document.getElementById('communitiesList');
    const statsCards = document.getElementById('statsCards');
    const toggleBtn = document.getElementById('toggleCommunitiesBtn');
    const collapseStatsBtn = document.getElementById('collapseStatsBtn');

    if (communitiesList && communitiesList.style.display !== 'none') {
        communitiesList.style.display = 'none';
        if (toggleBtn) toggleBtn.textContent = '展开';
    }

    if (statsCards && statsCards.style.display !== 'none') {
        statsCards.style.display = 'none';
        if (collapseStatsBtn) collapseStatsBtn.textContent = '展开';
    }

    setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
    }, 300);
}