// app/web/js/upload/upload-service.js
// 上传核心逻辑模块 - 支持多模态，修复轮询逻辑

import { elements, state } from './config.js';
import { getAuthHeaders, showToast, logout, getMediaType, getMediaTypeHint, getProcessingMessage } from './utils.js';
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

// 开始轮询状态（支持多模态）- 修复版：正确停止轮询
function startStatusPolling(taskId, filename, mediaType = null) {
    console.log(`开始轮询任务状态: taskId=${taskId}, filename=${filename}, mediaType=${mediaType}`);

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

            // 根据媒体类型显示友好的进度消息
            if (elements.progressText) {
                let progressMsg = statusData.message;
                if (!progressMsg || progressMsg === '正在处理...') {
                    if (statusData.media_type) {
                        progressMsg = getProcessingMessage(statusData.media_type);
                    } else if (mediaType) {
                        progressMsg = getProcessingMessage(mediaType);
                    }
                }
                elements.progressText.textContent = progressMsg;
            }

            // ========== 关键修复：检查 completed 状态 ==========
            const currentStatus = String(statusData.status).toLowerCase();

            if (currentStatus === 'completed') {
                console.log(`✅ ${filename} 处理完成 (status=${currentStatus})`);

                // 清除轮询
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                isCompleted = true;

                // 延迟隐藏进度条并刷新列表
                setTimeout(() => {
                    hideProgress();
                    // 刷新文件列表
                    loadFileList().catch(e => console.error('刷新文件列表失败:', e));

                    // 显示成功消息
                    let successMsg = `${filename} 处理完成！`;
                    if (statusData.media_type === 'image') {
                        successMsg = `${filename} 识别完成！`;
                    } else if (statusData.media_type === 'audio') {
                        successMsg = `${filename} 语音转文字完成！`;
                    } else if (statusData.media_type === 'video') {
                        successMsg = `${filename} 视频处理完成！`;
                    }
                    showToast(successMsg, 'success');
                }, 500);
                return;

            } else if (currentStatus === 'failed') {
                console.log(`❌ ${filename} 处理失败:`, statusData.error);
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                hideProgress();
                showToast(`${filename} 处理失败: ${statusData.error || '未知错误'}`, 'error');
                isCompleted = true;
                return;

            } else if (currentStatus === 'cancelled') {
                console.log(`🛑 ${filename} 已取消`);
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                hideProgress();
                showToast(`${filename} 已取消`, 'warning');
                isCompleted = true;
                return;

            } else if (pollCount >= maxPolls) {
                console.log(`⏰ ${filename} 轮询超时`);
                clearInterval(state.statusInterval);
                state.statusInterval = null;
                hideProgress();
                showToast(`${filename} 处理超时`, 'error');
                isCompleted = true;
                return;
            }

            // 如果状态是 processing，继续轮询
            console.log(`⏳ ${filename} 处理中: status=${currentStatus}, progress=${statusData.progress}`);

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

// 上传单个文件（支持多模态）- 修改为使用 /upload/async 接口
export async function uploadFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const mediaType = getMediaType(file.name);

    // 扩展支持的文件类型
    const supportedExts = [
        // 文档
        '.pdf', '.docx', '.txt', '.md', '.markdown',
        // 图片
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff',
        // 音频
        '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg',
        // 视频
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'
    ];

    if (!supportedExts.includes(ext)) {
        showToast(`不支持的文件类型: ${ext}`, 'error');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showToast('文件大小不能超过 50MB', 'error');
        return;
    }

    const typeHint = getMediaTypeHint(mediaType);

    // 批量上传模式判断
    const pendingCount = getPendingFilesCount();
    if (pendingCount > 0) {
        showToast(`${file.name} 已添加到队列 (${typeHint})`, 'success');
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

    showToast(`正在上传 ${file.name} (${typeHint})...`, 'success');

    try {
        const headers = getAuthHeaders();

        // 使用异步上传接口 /upload/async
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

            // 根据返回的媒体类型或本地检测的媒体类型显示处理提示
            const actualMediaType = data.media_type || mediaType;

            if (elements.progressText) {
                const processMsg = getProcessingMessage(actualMediaType);
                elements.progressText.textContent = processMsg;
            }

            // 开始轮询任务状态
            startStatusPolling(data.task_id, file.name, actualMediaType);
        } else {
            showToast(data.detail || data.message || '上传失败', 'error');
        }
    } catch (error) {
        console.error('上传错误:', error);
        showToast('上传失败: ' + error.message, 'error');
    }
}

// 取消上传任务
export async function cancelUploadTask(taskId, filename) {
    if (!taskId) return false;

    try {
        const headers = getAuthHeaders();
        const response = await fetch(`${API_BASE}/upload/task/${taskId}/cancel`, {
            method: 'POST',
            headers: headers
        });

        const data = await response.json();
        if (data.success) {
            showToast(`${filename || '任务'} 已取消`, 'warning');
            return true;
        } else {
            showToast(`取消失败: ${data.message}`, 'error');
            return false;
        }
    } catch (error) {
        console.error('取消任务失败:', error);
        showToast(`取消失败: ${error.message}`, 'error');
        return false;
    }
}

export { addFilesToPending } from './batch-upload.js';