// app/web/js/upload/upload-service.js
// 上传核心逻辑模块

import { elements, state } from './config.js';
import { getAuthHeaders, showToast, logout } from './utils.js';
import { loadFileList } from './file-list.js';
import { addFilesToPending, getPendingFilesCount } from './batch-upload.js';

const API_BASE = '/api';

// 显示进度条
function showProgress() {
    if (elements.progressContainer) {
        elements.progressContainer.style.display = 'block';
        if (elements.progressFill) elements.progressFill.style.width = '0%';
        if (elements.progressText) elements.progressText.textContent = '正在处理...';
    }
}

function hideProgress() {
    if (elements.progressContainer) {
        elements.progressContainer.style.display = 'none';
        if (elements.progressFill) elements.progressFill.style.width = '0%';
        if (elements.progressText) elements.progressText.textContent = '';
    }
}

// 开始轮询状态
function startStatusPolling(taskId, filename) {
    console.log(`开始轮询任务状态: taskId=${taskId}, filename=${filename}`);

    if (state.statusInterval) {
        clearInterval(state.statusInterval);
        state.statusInterval = null;
    }

    let pollCount = 0;
    const maxPolls = 180; // 最多轮询 180 次，每次 2 秒 = 6 分钟
    let isCompleted = false;

    state.statusInterval = setInterval(async () => {
        if (isCompleted) return;

        pollCount++;

        try {
            // 使用正确的任务状态接口
            const statusUrl = `${API_BASE}/upload/task/${taskId}`;
            console.log(`轮询状态 [${filename}] (${pollCount}): ${statusUrl}`);

            const headers = getAuthHeaders();
            const response = await fetch(statusUrl, {
                headers: headers,
                cache: 'no-store'
            });

            if (response.status === 401) {
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                hideProgress();
                showToast('登录已过期，请重新登录', 'error');
                logout();
                return;
            }

            if (!response.ok) {
                console.warn(`状态接口返回 ${response.status}: ${filename}`);
                if (pollCount >= maxPolls) {
                    clearInterval(state.statusInterval);
                    state.statusInterval = null;
                    hideProgress();
                    showToast(`${filename} 状态查询超时`, 'error');
                    isCompleted = true;
                }
                return;
            }

            const statusData = await response.json();
            console.log(`${filename} 状态数据:`, statusData);

            // 更新进度显示
            if (elements.progressFill && statusData.progress !== undefined) {
                elements.progressFill.style.width = `${statusData.progress || 50}%`;
            }
            if (elements.progressText && statusData.message) {
                elements.progressText.textContent = statusData.message;
            }

            // 检查完成状态
            if (statusData.status === 'completed') {
                console.log(`✅ ${filename} 处理完成`);
                clearInterval(state.statusInterval);
                state.statusInterval = null;

                setTimeout(() => {
                    hideProgress();
                    // 刷新文件列表
                    loadFileList().catch(e => console.error('刷新文件列表失败:', e));
                    showToast(`${filename} 处理完成！`, 'success');
                }, 500);
                isCompleted = true;

            } else if (statusData.status === 'failed') {
                console.log(`❌ ${filename} 处理失败:`, statusData.error);
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                hideProgress();
                showToast(`${filename} 处理失败: ${statusData.error || '未知错误'}`, 'error');
                isCompleted = true;

            } else if (pollCount >= maxPolls) {
                console.log(`⏰ ${filename} 轮询超时`);
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                hideProgress();
                showToast(`${filename} 处理超时`, 'error');
                isCompleted = true;
            }

        } catch (error) {
            console.error(`获取 ${filename} 状态失败:`, error);
            if (pollCount >= maxPolls) {
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                hideProgress();
                showToast(`${filename} 状态查询失败: ${error.message}`, 'error');
                isCompleted = true;
            }
        }
    }, 2000); // 每 2 秒轮询一次
}

// 上传单个文件
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
        addFilesToPending([file]);
        return;
    }

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

        // 使用异步上传接口
        const response = await fetch(`${API_BASE}/upload/async`, {
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
        console.log('上传响应:', data);

        if (data.success && data.task_id) {
            state.currentProcessId = data.task_id;
            showProgress();
            startStatusPolling(data.task_id, file.name);
        } else {
            showToast(data.detail || data.message || '上传失败', 'error');
        }
    } catch (error) {
        console.error('上传错误:', error);
        showToast('上传失败: ' + error.message, 'error');
    }
}

export { addFilesToPending } from './batch-upload.js';