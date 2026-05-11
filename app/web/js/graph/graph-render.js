// app/web/js/graph/graph-render.js - 图谱渲染模块
// 图谱渲染逻辑

import { graphData, NODE_COLORS, getEntityCategory } from './graph-data.js';

// 构建社区视图节点
function buildCommunityNodes(communities) {
    return communities.slice(0, 30).map((community, idx) => ({
        id: `comm_${community.id}`,
        name: `社区 ${community.id}`,
        category: idx % 10,
        value: community.size || 5,
        symbolSize: Math.min(45, 20 + (community.size || 5)),
        itemStyle: {
            color: `hsl(${((community.id * 37) % 360)}, 70%, 55%)`,
            borderColor: '#fff',
            borderWidth: 2,
            shadowBlur: 10,
            shadowColor: 'rgba(0,0,0,0.1)'
        },
        label: {
            show: true,
            fontSize: 12,
            fontWeight: 'bold',
            color: '#212529',
            textShadowBlur: 4,
            textShadowColor: 'rgba(255,255,255,0.5)'
        },
        isCommunity: true,
        size: community.size,
        density: community.density,
        keywords: community.keywords,
        summary: community.summary
    }));
}

// 构建社区视图关系
function buildCommunityLinks(communities, entities, relations) {
    const relationCount = new Map();
    const entityToCommunity = new Map();

    if (entities) {
        entities.forEach(entity => {
            for (const comm of communities) {
                if (comm.keywords && comm.keywords.some(kw =>
                    entity.name && (entity.name.includes(kw) || kw.includes(entity.name)))) {
                    entityToCommunity.set(entity.name, comm.id);
                    break;
                }
            }
        });
    }

    if (relations) {
        relations.forEach(relation => {
            const sourceComm = entityToCommunity.get(relation.source);
            const targetComm = entityToCommunity.get(relation.target);
            if (sourceComm && targetComm && sourceComm !== targetComm) {
                const key = `${sourceComm}|${targetComm}`;
                relationCount.set(key, (relationCount.get(key) || 0) + (relation.weight || 1));
            }
        });
    }

    const links = [];
    let linkCount = 0;
    for (const [key, count] of relationCount) {
        if (linkCount >= 40) break;
        const [source, target] = key.split('|');
        links.push({
            source: `comm_${source}`,
            target: `comm_${target}`,
            weight: Math.min(count, 5),
            lineStyle: {
                width: Math.min(3, 1 + count / 3),
                curveness: 0.2,
                opacity: 0.6,
                color: '#4263eb'
            }
        });
        linkCount++;
    }

    return links;
}

// 构建实体视图节点
function buildEntityNodes(entities, relations) {
    // 计算实体度数
    const degreeCount = new Map();
    if (relations) {
        relations.forEach(rel => {
            degreeCount.set(rel.source, (degreeCount.get(rel.source) || 0) + 1);
            degreeCount.set(rel.target, (degreeCount.get(rel.target) || 0) + 1);
        });
    }

    // 按频率+度数排序
    const sortedEntities = [...(entities || [])].sort((a, b) => {
        const scoreA = (a.frequency || 0) + (degreeCount.get(a.name) || 0);
        const scoreB = (b.frequency || 0) + (degreeCount.get(b.name) || 0);
        return scoreB - scoreA;
    });

    const topEntities = sortedEntities.slice(0, 60);

    return topEntities.map(entity => {
        const colors = NODE_COLORS[entity.type] || NODE_COLORS.default;
        const degree = degreeCount.get(entity.name) || 0;
        const size = Math.min(35, 12 + (entity.frequency || 1) * 1.5 + degree * 0.5);

        return {
            id: entity.name,
            name: entity.name,
            category: getEntityCategory(entity.type),
            value: entity.frequency || 1,
            degree: degree,
            symbolSize: size,
            itemStyle: {
                color: colors.color,
                borderColor: colors.borderColor,
                borderWidth: 2,
                shadowBlur: 8,
                shadowColor: 'rgba(0,0,0,0.1)'
            },
            label: {
                show: true,
                fontSize: 11,
                fontWeight: size > 20 ? 'bold' : 'normal',
                color: '#212529',
                textShadowBlur: 2,
                textShadowColor: 'rgba(255,255,255,0.5)'
            },
            type: entity.type,
            frequency: entity.frequency,
            summary: entity.summary,
            isCommunity: false
        };
    });
}

// 构建实体视图关系
function buildEntityLinks(entities, relations) {
    const nodeSet = new Set(entities.map(n => n.id));
    const filteredRelations = (relations || []).filter(rel =>
        nodeSet.has(rel.source) && nodeSet.has(rel.target)
    ).slice(0, 150);

    return filteredRelations.map(relation => {
        const weight = relation.weight || 1;
        return {
            source: relation.source,
            target: relation.target,
            weight: weight,
            relationType: relation.relation_type,
            lineStyle: {
                width: Math.min(3, 1 + weight / 2),
                curveness: 0.2,
                opacity: 0.7,
                color: '#4263eb'
            }
        };
    });
}

// 构建类别
function buildCategories(isCommunityMode, communities) {
    if (isCommunityMode) {
        return (communities || []).slice(0, 10).map((c, i) => ({ name: `社区 ${c.id}` }));
    } else {
        return [
            { name: '👤 人物', itemStyle: { color: NODE_COLORS.PERSON.color } },
            { name: '🏢 组织', itemStyle: { color: NODE_COLORS.ORGANIZATION.color } },
            { name: '📍 地点', itemStyle: { color: NODE_COLORS.LOCATION.color } },
            { name: '💡 概念', itemStyle: { color: NODE_COLORS.CONCEPT.color } },
            { name: '📦 产品', itemStyle: { color: NODE_COLORS.PRODUCT.color } },
            { name: '📅 日期', itemStyle: { color: NODE_COLORS.DATE.color } },
            { name: '🔢 数值', itemStyle: { color: NODE_COLORS.NUMBER.color } },
            { name: '📌 其他', itemStyle: { color: NODE_COLORS.default.color } }
        ];
    }
}

// 渲染图谱
export function renderGraph(chart, layout, displayMode, onNodeClick) {
    if (!chart || !graphData) {
        console.warn('renderGraph: chart or graphData is null', { chart: !!chart, graphData: !!graphData });
        return;
    }

    console.log('renderGraph 调用:', {
        displayMode: displayMode,
        entitiesCount: graphData.entities?.length,
        relationsCount: graphData.relations?.length,
        communitiesCount: graphData.communities?.length
    });

    const isCommunityMode = displayMode === 'community';
    console.log(`Rendering graph: mode=${displayMode}`);

    let nodes = [];
    let links = [];
    let categories = [];

    if (isCommunityMode) {
        const communities = graphData.communities || [];
        nodes = buildCommunityNodes(communities);
        links = buildCommunityLinks(communities, graphData.entities, graphData.relations);
        categories = buildCategories(true, communities);
    } else {
        nodes = buildEntityNodes(graphData.entities, graphData.relations);
        links = buildEntityLinks(nodes, graphData.relations);
        categories = buildCategories(false);
    }

    console.log(`Final: ${nodes.length} nodes, ${links.length} links`);

    if (nodes.length === 0) {
        chart.setOption({
            title: {
                show: true,
                text: '暂无图谱数据\n请先上传文档',
                left: 'center',
                top: 'center',
                textStyle: { color: '#adb5bd', fontSize: 14, fontWeight: 'normal' }
            },
            backgroundColor: 'transparent'
        });
        return;
    }

    const option = {
        title: { show: false },
        tooltip: {
            trigger: 'item',
            formatter: (params) => {
                if (params.dataType === 'node') {
                    if (params.data.isCommunity) {
                        return `<div style="max-width: 250px;">
                            <strong style="color: #4263eb;">🏘️ ${params.name}</strong><br/>
                            <span style="color: #6c757d;">实体数: ${params.data.size || 0}</span><br/>
                            <span style="color: #6c757d;">密度: ${(params.data.density || 0).toFixed(2)}</span><br/>
                            <span style="color: #40c057;">关键词: ${(params.data.keywords || []).slice(0, 5).join(', ')}</span>
                        </div>`;
                    } else {
                        let typeIcon = '';
                        switch (params.data.type) {
                            case 'PERSON': typeIcon = '👤'; break;
                            case 'ORGANIZATION': typeIcon = '🏢'; break;
                            case 'LOCATION': typeIcon = '📍'; break;
                            case 'CONCEPT': typeIcon = '💡'; break;
                            case 'PRODUCT': typeIcon = '📦'; break;
                            default: typeIcon = '📌';
                        }
                        return `<div style="max-width: 250px;">
                            <strong style="color: #4263eb;">${typeIcon} ${params.name}</strong><br/>
                            <span style="color: #6c757d;">类型: ${params.data.type || '概念'}</span><br/>
                            <span style="color: #fd7e14;">频率: ${params.data.frequency || 1}</span><br/>
                            <span style="color: #339af0;">连接数: ${params.data.degree || 0}</span>
                            ${params.data.summary ? `<hr style="margin: 6px 0; border-color: #e9ecef;"/><span style="color: #495057; font-size: 11px;">${params.data.summary.substring(0, 100)}</span>` : ''}
                        </div>`;
                    }
                } else {
                    return `<div>
                        <strong style="color: #4263eb;">🔗 ${params.data.source} → ${params.data.target}</strong><br/>
                        <span style="color: #40c057;">关系: ${params.data.relationType || 'RELATED_TO'}</span><br/>
                        <span style="color: #fd7e14;">权重: ${params.data.weight || 1}</span>
                    </div>`;
                }
            },
            backgroundColor: '#ffffff',
            borderColor: '#4263eb',
            borderWidth: 1,
            textStyle: { color: '#212529', fontSize: 12 },
            extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 8px;'
        },
        legend: {
            data: categories,
            orient: 'vertical',
            left: 'left',
            top: 'top',
            textStyle: { color: '#6c757d' },
            backgroundColor: 'rgba(255,255,255,0.9)',
            borderRadius: 8,
            padding: [8, 12],
            itemWidth: 20,
            itemHeight: 12
        },
        series: [{
            type: 'graph',
            layout: layout,
            data: nodes,
            links: links,
            categories: categories,
            roam: true,
            draggable: true,
            focusNodeAdjacency: false,
            zoom: 0.8,
            label: {
                show: true,
                position: 'right',
                offset: [8, 0],
                formatter: (params) => {
                    let name = params.name;
                    if (name && name.length > 15) {
                        name = name.substring(0, 12) + '...';
                    }
                    return name;
                }
            },
            edgeLabel: { show: false },
            emphasis: {
                focus: 'adjacency',
                lineStyle: { width: 3, color: '#fd7e14' },
                label: { show: true, fontSize: 12, fontWeight: 'bold' },
                scale: 1.1
            },
            lineStyle: {
                color: 'source',
                curveness: 0.2,
                opacity: 0.6
            },
            force: {
                initIterations: 300,
                repulsion: 800,
                edgeLength: [80, 200],
                gravity: 0.08,
                friction: 0.1,
                layoutAnimation: true
            },
            roamZoom: true,
            roamPan: true,
            animation: true,
            animationDuration: 500,
            animationEasing: 'cubicOut'
        }],
        backgroundColor: 'transparent'
    };

    chart.setOption(option, true);

    setTimeout(() => {
        if (chart) chart.resize();
    }, 150);

    // 绑定点击事件
    chart.off('click');
    chart.on('click', (params) => {
        if (params.dataType === 'node' && params.data && onNodeClick) {
            onNodeClick(params.data);
        }
    });
}

// 渲染社区列表
export function renderCommunities(data, container, onCommunityClick) {
    if (!container) return;

    const communities = data.communities || [];

    if (communities.length === 0) {
        container.innerHTML = '<div style="text-align: center; padding: 20px; color: #adb5bd;">暂无社区数据</div>';
        return;
    }

    let html = '';
    communities.forEach(community => {
        const entitiesPreview = (community.keywords || []).slice(0, 6).join('、');
        html += `
            <div class="community-card" data-community-id="${community.id}">
                <div class="community-header">
                    <span class="community-title">🏘️ 社区 ${community.id}</span>
                    <span class="community-size">${community.size} 个实体</span>
                </div>
                <div class="community-entities">
                    关键词: ${entitiesPreview || '无'}
                </div>
                <div class="community-summary">
                    ${community.summary || `包含 ${community.size} 个相关实体，密度 ${(community.density || 0).toFixed(2)}`}
                </div>
            </div>
        `;
    });

    container.innerHTML = html;

    document.querySelectorAll('.community-card').forEach(card => {
        card.addEventListener('click', () => {
            const communityId = card.dataset.communityId;
            if (onCommunityClick) onCommunityClick(communityId);
        });
    });
}

// 高亮社区
export function highlightCommunity(chart, data, currentDisplayMode, displayModeSelect, communityId) {
    if (!chart) return;

    if (currentDisplayMode === 'community') {
        const option = chart.getOption();
        const series = option.series[0];

        if (series && series.data) {
            const newData = series.data.map(node => {
                if (node.id === `comm_${communityId}`) {
                    return {
                        ...node,
                        itemStyle: {
                            color: '#ff6b6b',
                            borderWidth: 4,
                            borderColor: '#fff',
                            shadowBlur: 20
                        },
                        symbolSize: (node.symbolSize || 20) * 1.3
                    };
                }
                return node;
            });
            series.data = newData;
            chart.setOption(option);
        }
    } else {
        if (displayModeSelect) displayModeSelect.value = 'community';
        // 需要重新渲染，由调用方处理
    }
}