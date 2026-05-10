// app/web/js/upload/utils.js
// 工具函数模块

// 显示提示消息
export function showToast(message, type = 'success', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
}

// HTML转义
export function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 获取文件图标
export function getFileIcon(filename) {
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
export function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// 格式化时间
export function formatDate(timestamp) {
    if (!timestamp) return '未知';
    try {
        return new Date(timestamp * 1000).toLocaleString();
    } catch (e) {
        return '未知';
    }
}

// 获取等级徽章HTML
export function getLevelBadgeHtml(level) {
    if (level === 'admin') {
        return '<span class="level-badge admin">🔐 管理员</span>';
    } else if (level === 'owner') {
        return '<span class="level-badge owner">🔒 所有者</span>';
    }
    return '<span class="level-badge normal">📄 普通</span>';
}

// 退出登录
export function logout() {
    localStorage.removeItem('rag_token');
    localStorage.removeItem('rag_user_id');
    localStorage.removeItem('rag_username');
    window.location.href = '/login.html';
}

// 检查登录状态
export function isLoggedIn() {
    const token = localStorage.getItem('rag_token');
    return token && token !== 'null' && token !== 'undefined';
}

// 获取认证头
export function getAuthHeaders() {
    const token = localStorage.getItem('rag_token');
    if (token && token !== 'null' && token !== 'undefined') {
        return { 'Authorization': `Bearer ${token}` };
    }
    return {};
}

// 验证 Token 有效性
export async function verifyToken() {
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