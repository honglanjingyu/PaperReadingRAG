// app/web/js/upload.js
// 上传页面主入口 - 支持多模态和批量上传

import { elements, initElements, state } from './upload/config.js';
import { showToast, isLoggedIn, verifyToken, logout, getMediaType, getProcessingMessage } from './upload/utils.js';
import { uploadFile } from './upload/upload-service.js';
import { loadFileList } from './upload/file-list.js';
import { batchDeleteDocuments, updateBatchDeleteButton } from './upload/delete.js';
import { getUserRole } from './upload/api.js';
import { initBatchUploadElements, addFilesToPending, resetBatchUpload, isUploadingFiles, getPendingFilesCount } from './upload/batch-upload.js';

// 显示当前用户名
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

// 显示用户等级
async function displayUserRole() {
    const roleSpan = document.getElementById('userRoleBadge');
    if (!roleSpan) return;

    try {
        const token = localStorage.getItem('rag_token');
        if (!token) {
            roleSpan.innerHTML = '👤 普通用户';
            roleSpan.className = 'user-role-badge normal';
            return;
        }

        const role = await getUserRole();

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
    } catch (error) {
        console.error('获取用户等级失败:', error);
        roleSpan.innerHTML = '👤 普通用户';
        roleSpan.className = 'user-role-badge normal';
    }
}

// 初始化上传事件监听
function initUploadEventListeners() {
    if (!elements.uploadArea || !elements.fileInput) {
        console.error('上传区域或文件输入元素未找到');
        return;
    }

    // 点击上传区域
    elements.uploadArea.addEventListener('click', () => {
        if (isUploadingFiles()) {
            showToast('请等待当前上传完成', 'warning');
            return;
        }
        elements.fileInput.click();
    });

    // 拖拽上传
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });

    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });

    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');

        if (isUploadingFiles()) {
            showToast('请等待当前上传完成', 'warning');
            return;
        }

        const files = Array.from(e.dataTransfer.files);
        if (files.length > 0) {
            if (files.length === 1) {
                // 单个文件使用原有逻辑
                uploadFile(files[0]);
            } else {
                // 多个文件使用批量上传
                addFilesToPending(files);
            }
        }
    });

    // 文件选择（支持多选）
    elements.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            const files = Array.from(e.target.files);
            if (files.length === 1) {
                uploadFile(files[0]);
            } else {
                addFilesToPending(files);
            }
        }
        elements.fileInput.value = '';
    });

    // 批量删除按钮
    if (elements.batchDeleteBtn) {
        elements.batchDeleteBtn.addEventListener('click', batchDeleteDocuments);
    }
}

// 页面初始化
document.addEventListener('DOMContentLoaded', async () => {
    // 初始化 DOM 元素
    initElements();

    // 初始化批量上传相关元素
    initBatchUploadElements();

    // 显示用户信息
    displayCurrentUser();
    await displayUserRole();

    // 退出登录按钮
    const logoutBtn = document.getElementById('logoutBtn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            if (confirm('确定要退出登录吗？')) {
                logout();
            }
        });
    }

    // 检查登录状态
    if (!isLoggedIn()) {
        window.location.href = '/login.html';
        return;
    }

    // 验证 Token
    try {
        const isValid = await verifyToken();
        if (!isValid) {
            logout();
            return;
        }
    } catch (error) {
        console.error('Token 验证失败:', error);
        logout();
        return;
    }

    // 初始化事件监听
    initUploadEventListeners();

    // 加载文件列表
    await loadFileList();

    // 重置批量上传状态
    resetBatchUpload();
});