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

// ========== 认证相关函数 ==========

// 获取认证 Token
function getAuthToken() {
    return localStorage.getItem('rag_token');
}

// 获取认证请求头
function getAuthHeaders() {
    const token = getAuthToken();
    if (token && token !== 'null' && token !== 'undefined') {
        return { 'Authorization': `Bearer ${token}` };
    }
    return {};
}

// 检查是否已登录
function isLoggedIn() {
    const token = getAuthToken();
    return token && token !== 'null' && token !== 'undefined';
}

// 退出登录
function logout() {
    localStorage.removeItem('rag_token');
    localStorage.removeItem('rag_user_id');
    localStorage.removeItem('rag_username');
    window.location.href = '/login.html';
}

// 带认证的 fetch 封装
async function authFetch(url, options = {}) {
    const headers = {
        'Content-Type': 'application/json',
        ...getAuthHeaders(),
        ...options.headers
    };

    const response = await fetch(url, { ...options, headers });

    if (response.status === 401) {
        logout();
        throw new Error('登录已过期，请重新登录');
    }

    return response;
}

// 验证 Token 有效性
async function verifyToken() {
    const token = getAuthToken();
    if (!token) return false;

    try {
        const response = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: getAuthHeaders()
        });
        const data = await response.json();
        return data.success === true;
    } catch (error) {
        return false;
    }
}

// 页面加载时更新状态
document.addEventListener('DOMContentLoaded', () => {
    updateSystemStatus();
    // 每30秒更新一次状态
    setInterval(updateSystemStatus, 30000);
});

/* app/web/js/common.js - 添加以下函数 */

// ========== 用户信息显示 ==========

// 显示当前登录用户名
function displayCurrentUser() {
    const username = localStorage.getItem('rag_username');
    const userNameSpan = document.getElementById('userNameDisplay');
    
    if (userNameSpan) {
        if (username && username !== 'null' && username !== 'undefined') {
            userNameSpan.textContent = username;
        } else {
            userNameSpan.textContent = '用户';
        }
    }
}

// 退出登录
function handleLogout() {
    if (confirm('确定要退出登录吗？')) {
        // 清除本地存储
        localStorage.removeItem('rag_token');
        localStorage.removeItem('rag_user_id');
        localStorage.removeItem('rag_username');
        localStorage.removeItem('rag_current_session_id');
        
        // 清除会话相关缓存
        if (window.sessionStorage) {
            window.sessionStorage.clear();
        }
        
        // 跳转到登录页
        window.location.href = '/login.html';
    }
}

// 修改 isLoggedIn 函数，增加有效性检查
function isLoggedIn() {
    const token = getAuthToken();
    if (!token || token === 'null' || token === 'undefined') return false;
    
    // 检查 token 格式（简单验证）
    try {
        const parts = token.split('.');
        if (parts.length !== 3) return false;
        return true;
    } catch (e) {
        return false;
    }
}

// 修改 verifyToken 函数，增加更详细的日志
async function verifyToken() {
    const token = getAuthToken();
    if (!token) return false;

    try {
        const response = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) return false;
        
        const data = await response.json();
        
        if (data.success) {
            // 确保用户名已存储
            if (data.username && !localStorage.getItem('rag_username')) {
                localStorage.setItem('rag_username', data.username);
            }
            return true;
        }
        return false;
    } catch (error) {
        console.error('Token 验证失败:', error);
        return false;
    }
}

// 页面加载时显示用户名（在所有页面中调用）
document.addEventListener('DOMContentLoaded', () => {
    updateSystemStatus();
    displayCurrentUser();  // 添加这一行
    
    // 绑定退出登录按钮（如果存在）
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', handleLogout);
    }
    
    setInterval(updateSystemStatus, 30000);
});