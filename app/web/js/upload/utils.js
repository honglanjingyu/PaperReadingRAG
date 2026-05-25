// app/web/js/upload/utils.js
// 工具函数模块 - 支持多模态

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

// 获取文件图标（扩展支持多媒体）
export function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        // 文档
        'pdf': '📕',
        'docx': '📘',
        'txt': '📄',
        'md': '📝',
        'markdown': '📝',
        // 图片
        'jpg': '🖼️',
        'jpeg': '🖼️',
        'png': '🖼️',
        'gif': '🎞️',
        'bmp': '🖼️',
        'webp': '🖼️',
        'tiff': '🖼️',
        // 音频
        'mp3': '🎵',
        'wav': '🎵',
        'flac': '🎵',
        'm4a': '🎵',
        'aac': '🎵',
        'ogg': '🎵',
        // 视频
        'mp4': '🎬',
        'avi': '🎬',
        'mov': '🎬',
        'mkv': '🎬',
        'flv': '🎬',
        'wmv': '🎬',
        'webm': '🎬'
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
        console.error('Token 验证失败:', error);
        return false;
    }
}

// ========== 多模态相关函数 ==========

// 获取媒体类型
export function getMediaType(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const imageExts = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff'];
    const audioExts = ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'];
    const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm'];
    const docExts = ['pdf', 'docx', 'txt', 'md', 'markdown'];

    if (imageExts.includes(ext)) return 'image';
    if (audioExts.includes(ext)) return 'audio';
    if (videoExts.includes(ext)) return 'video';
    if (docExts.includes(ext)) return 'document';
    return 'other';
}

// 获取媒体类型标签HTML
export function getMediaTypeBadge(mediaType) {
    if (mediaType === 'image') {
        return '<span class="media-badge image">🖼️ 图片</span>';
    } else if (mediaType === 'audio') {
        return '<span class="media-badge audio">🎵 音频</span>';
    } else if (mediaType === 'video') {
        return '<span class="media-badge video">🎬 视频</span>';
    }
    return '<span class="media-badge doc">📄 文档</span>';
}

// 获取媒体类型提示文字
export function getMediaTypeHint(mediaType) {
    if (mediaType === 'image') return '图片文件';
    if (mediaType === 'audio') return '音频文件（将自动进行语音转文字）';
    if (mediaType === 'video') return '视频文件（将自动提取音频和关键帧）';
    return '文档文件';
}

// 获取媒体类型的处理状态文字
export function getProcessingMessage(mediaType) {
    if (mediaType === 'image') return '正在处理中...';
    if (mediaType === 'audio') return '正在通过 ASR 转写音频...';
    if (mediaType === 'video') return '正在处理视频（提取音频+关键帧）...';
    return '正在处理文档...';
}