// app/web/js/upload/api.js
// API 请求模块

import { API_BASE } from './config.js';
import { getAuthHeaders, logout, showToast } from './utils.js';

// 获取用户等级
export async function getUserRole() {
    const token = localStorage.getItem('rag_token');
    if (!token) return 'normal';

    try {
        const response = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: getAuthHeaders()
        });
        if (response.ok) {
            const data = await response.json();
            return data.role || 'normal';
        }
        return 'normal';
    } catch (error) {
        console.error('获取用户等级失败:', error);
        return 'normal';
    }
}

// app/web/js/upload/api.js
// 修改 fetchFileList 函数

// 获取文档列表
export async function fetchFileList() {
    const headers = getAuthHeaders();
    // 添加时间戳防止缓存
    const timestamp = Date.now();
    const response = await fetch(`${API_BASE}/upload/list?t=${timestamp}`, {
        headers: headers,
        cache: 'no-store'  // 禁用缓存
    });

    if (response.status === 401) {
        logout();
        throw new Error('登录已过期');
    }

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    return await response.json();
}

// 获取处理状态
export async function fetchProcessStatus(processId) {
    const headers = getAuthHeaders();
    const response = await fetch(`${API_BASE}/upload/status/${processId}`, {
        headers: headers
    });

    if (response.status === 401) {
        logout();
        throw new Error('登录已过期');
    }

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    return await response.json();
}

// 删除单个文档
export async function deleteDocumentApi(filename) {
    const headers = getAuthHeaders();
    const response = await fetch(`${API_BASE}/upload/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
        headers: headers
    });

    if (response.status === 401) {
        logout();
        throw new Error('登录已过期');
    }

    return await response.json();
}

// 批量删除文档
export async function batchDeleteApi(filenames) {
    const headers = getAuthHeaders();
    const response = await fetch(`${API_BASE}/upload/delete-batch`, {
        method: 'POST',
        headers: {
            ...headers,
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ filenames })
    });

    if (response.status === 401) {
        logout();
        throw new Error('登录已过期');
    }

    return await response.json();
}