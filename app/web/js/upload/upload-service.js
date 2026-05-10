// app/web/js/upload/upload-service.js - 添加批量上传支持
// 上传核心逻辑模块

import { elements, state } from './config.js';
import { getAuthHeaders, showToast, logout } from './utils.js';
import { fetchProcessStatus } from './api.js';
import { loadFileList } from './file-list.js';
import { addFilesToPending, getPendingFilesCount } from './batch-upload.js';

// 显示进度条
function showProgress() {
    if (elements.progressContainer) {
        elements.progressContainer.style.display = 'block';
        elements.progressFill.style.width = '0%';
        elements.progressText.textContent = '正在处理...';
    }
}

// 开始轮询状态
function startStatusPolling(processId, filename) {
    if (state.statusInterval) clearInterval(state.statusInterval);

    state.statusInterval = setInterval(async () => {
        try {
            const data = await fetchProcessStatus(processId);

            if (elements.progressFill) {
                elements.progressFill.style.width = `${data.progress}%`;
            }
            if (elements.progressText) {
                elements.progressText.textContent = data.message;
            }

            if (data.status === 'completed') {
                clearInterval(state.statusInterval);
                state.statusInterval = null;

                setTimeout(() => {
                    if (elements.progressContainer) {
                        elements.progressContainer.style.display = 'none';
                    }
                    loadFileList();
                    showToast(`${filename} 处理完成！生成了 ${data.result?.chunks_count || 0} 个分块`, 'success');
                }, 1000);

            } else if (data.status === 'failed') {
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                if (elements.progressContainer) {
                    elements.progressContainer.style.display = 'none';
                }
                showToast(`${filename} 处理失败: ${data.error || '未知错误'}`, 'error');
            }
        } catch (error) {
            console.error('获取状态失败:', error);
        }
    }, 1000);
}

// 上传单个文件（供原有逻辑使用）
export async function uploadFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const supportedExts = ['.pdf', '.docx', '.txt', '.md', '.markdown'];
    if (!supportedExts.includes(ext)) {
        showToast(`不支持的文件类型: ${ext}，支持: ${supportedExts.join(', ')}`, 'error');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showToast('文件大小不能超过 50MB', 'error');
        return;
    }

    // 如果是批量模式，添加到待上传队列
    const pendingCount = getPendingFilesCount();
    if (pendingCount > 0) {
        // 已经有待上传文件，添加到批量队列
        addFilesToPending([file]);
        return;
    }

    // 原有单文件上传逻辑
    const formData = new FormData();
    formData.append('file', file);

    if (elements.chunkSize) formData.append('chunk_size', elements.chunkSize.value);
    if (elements.fromPage) formData.append('from_page', elements.fromPage.value);
    if (elements.toPage) formData.append('to_page', elements.toPage.value);
    if (elements.enableVectorization) formData.append('enable_vectorization', elements.enableVectorization.checked);
    if (elements.enableStorage) formData.append('enable_storage', elements.enableStorage.checked);

    showToast(`正在上传 ${file.name}...`, 'success');

    try {
        const headers = getAuthHeaders();
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            headers: headers,
            body: formData
        });

        if (response.status === 401) {
            showToast('登录已过期，请重新登录', 'error');
            logout();
            return;
        }

        const data = await response.json();

        if (data.success) {
            state.currentProcessId = data.process_id;
            showProgress();
            startStatusPolling(data.process_id, file.name);
        } else {
            showToast(data.detail || '上传失败', 'error');
        }
    } catch (error) {
        console.error('上传错误:', error);
        showToast('上传失败: ' + error.message, 'error');
    }
}

export { addFilesToPending } from './batch-upload.js';