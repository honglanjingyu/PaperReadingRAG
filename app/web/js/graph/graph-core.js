// app/web/js/graph/graph-core.js - ECharts 图表核心
// ECharts 图表初始化和管理

export function initGraph(canvasElement) {
    console.log('Initializing ECharts with Neo4j-style graph...');

    const container = canvasElement.closest('.graph-container');
    if (container && container.clientHeight < 100) {
        container.style.height = 'calc(100vh - 200px)';
    }

    const chart = echarts.init(canvasElement);

    // 右键菜单阻止默认行为
    canvasElement.addEventListener('contextmenu', (e) => {
        e.preventDefault();
    });

    return chart;
}

export function resizeGraph(chart) {
    if (chart) chart.resize();
}

export function setupGraphClickEvents(chart, onNodeClick, onNodeDoubleClick) {
    chart.off('click');
    chart.on('click', (params) => {
        if (params.dataType === 'node' && params.data && onNodeClick) {
            onNodeClick(params.data);
        }
    });

    chart.off('dblclick');
    chart.on('dblclick', (params) => {
        if (params.dataType === 'node' && params.data && onNodeDoubleClick) {
            onNodeDoubleClick(params.data.id);
        }
    });
}