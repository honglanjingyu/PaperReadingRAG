/* app/web/js/common.js */
/* 公共函数 */

// API基础路径
const API_BASE = '/api';

// 显示提示消息
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// HTML转义
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 获取文件图标
function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'pdf': '📕',
        'docx': '📘',
        'txt': '📄',
        'md': '📝',
        'markdown': '📝'
    };
    return icons[ext] || '📄';
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// 格式化时间
function formatDate(timestamp) {
    return new Date(timestamp * 1000).toLocaleString();
}

// 获取系统配置
async function getSystemConfig() {
    try {
        const response = await fetch(`${API_BASE}/config`);
        return await response.json();
    } catch (error) {
        console.error('获取配置失败:', error);
        return null;
    }
}

// 健康检查
async function healthCheck() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        return data.status === 'healthy';
    } catch (error) {
        return false;
    }
}

// 更新系统状态显示
async function updateSystemStatus() {
    const isHealthy = await healthCheck();
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');

    if (statusDot) {
        statusDot.style.background = isHealthy ? '#4caf50' : '#f44336';
    }
    if (statusText) {
        statusText.textContent = isHealthy ? '系统运行中' : '系统异常';
    }
}

// 页面加载时更新状态
document.addEventListener('DOMContentLoaded', () => {
    updateSystemStatus();
    // 每30秒更新一次状态
    setInterval(updateSystemStatus, 30000);
});