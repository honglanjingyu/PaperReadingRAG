// app/web/js/upload/batch-upload.js
// 批量上传核心模块 - 支持多模态

import { elements, state, updateSelectedFiles } from './config.js';
import { getAuthHeaders, showToast, escapeHtml, formatFileSize, logout, getMediaType, getMediaTypeHint, getMediaTypeBadge } from './utils.js';
import { loadFileList } from './file-list.js';
import { API_BASE } from './config.js';

// 待上传文件队列
let pendingFiles = [];
let isUploading = false;
let currentUploadTasks = [];

// 上传进度记录
let uploadResults = {
    completed: [],
    failed: []
};

// DOM 元素
let batchUploadToolbar, pendingFilesContainer, pendingFilesList;
let batchFileCountSpan, batchTotalSizeSpan;
let globalProgressContainer, globalProgressFill, globalProgressStatus, globalProgressStats;
let uploadProgressList, uploadProgressItems;
let batchConfirmModal, confirmFileList, confirmFileCountSpan;
let clearBatchBtn, uploadBatchBtn;

// 初始化批量上传相关 DOM 元素
export function initBatchUploadElements() {
    batchUploadToolbar = document.getElementById('batchUploadToolbar');
    pendingFilesContainer = document.getElementById('pendingFilesContainer');
    pendingFilesList = document.getElementById('pendingFilesList');
    batchFileCountSpan = document.getElementById('batchFileCount');
    batchTotalSizeSpan = document.getElementById('batchTotalSize');

    globalProgressContainer = document.getElementById('globalProgressContainer');
    globalProgressFill = document.getElementById('globalProgressFill');
    globalProgressStatus = document.getElementById('globalProgressStatus');
    globalProgressStats = document.getElementById('globalProgressStats');

    uploadProgressList = document.getElementById('uploadProgressList');
    uploadProgressItems = document.getElementById('uploadProgressItems');

    batchConfirmModal = document.getElementById('batchConfirmModal');
    confirmFileList = document.getElementById('confirmFileList');
    confirmFileCountSpan = document.getElementById('confirmFileCount');

    clearBatchBtn = document.getElementById('clearBatchBtn');
    uploadBatchBtn = document.getElementById('uploadBatchBtn');

    // 绑定事件
    if (clearBatchBtn) {
        clearBatchBtn.addEventListener('click', clearPendingFiles);
    }
    if (uploadBatchBtn) {
        uploadBatchBtn.addEventListener('click', showConfirmModal);
    }

    // 模态框事件
    const modalClose = document.querySelector('#batchConfirmModal .modal-close');
    const cancelBtn = document.querySelector('#batchConfirmModal .btn-cancel');
    const confirmBtn = document.querySelector('#batchConfirmModal .btn-confirm');

    if (modalClose) {
        modalClose.addEventListener('click', () => hideModal());
    }
    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => hideModal());
    }
    if (confirmBtn) {
        confirmBtn.addEventListener('click', () => {
            hideModal();
            startBatchUpload();
        });
    }

    // 点击模态框背景关闭
    if (batchConfirmModal) {
        batchConfirmModal.addEventListener('click', (e) => {
            if (e.target === batchConfirmModal) hideModal();
        });
    }

    // 折叠/展开进度列表
    const collapseBtn = document.getElementById('collapseProgressBtn');
    if (collapseBtn) {
        collapseBtn.addEventListener('click', () => {
            if (uploadProgressItems) {
                const isCollapsed = uploadProgressItems.style.display === 'none';
                uploadProgressItems.style.display = isCollapsed ? 'block' : 'none';
                collapseBtn.textContent = isCollapsed ? '收起' : '展开';
            }
        });
    }
}

// 获取文件图标
function getFileIconByName(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'pdf': '📕', 'docx': '📘', 'txt': '📄', 'md': '📝', 'markdown': '📝',
        'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️', 'gif': '🎞️', 'bmp': '🖼️', 'webp': '🖼️', 'tiff': '🖼️',
        'mp3': '🎵', 'wav': '🎵', 'flac': '🎵', 'm4a': '🎵', 'aac': '🎵', 'ogg': '🎵',
        'mp4': '🎬', 'avi': '🎬', 'mov': '🎬', 'mkv': '🎬', 'flv': '🎬', 'wmv': '🎬', 'webm': '🎬'
    };
    return icons[ext] || '📄';
}

// 添加文件到待上传队列
export function addFilesToPending(files) {
    const validFiles = [];

    const supportedExts = [
        '.pdf', '.docx', '.txt', '.md', '.markdown',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff',
        '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg',
        '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'
    ];

    for (const file of files) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!supportedExts.includes(ext)) {
            showToast(`跳过不支持的文件: ${file.name}`, 'warning');
            continue;
        }

        if (file.size > 50 * 1024 * 1024) {
            showToast(`跳过超大文件: ${file.name} (最大 50MB)`, 'warning');
            continue;
        }

        const exists = pendingFiles.some(f => f.name === file.name);
        if (exists) {
            showToast(`文件已在队列中: ${file.name}`, 'warning');
            continue;
        }

        const mediaType = getMediaType(file.name);
        const typeHint = getMediaTypeHint(mediaType);

        validFiles.push({
            name: file.name,
            size: file.size,
            file: file,
            status: 'pending',
            mediaType: mediaType,
            icon: getFileIconByName(file.name),
            hint: typeHint
        });
    }

    if (validFiles.length === 0) return;

    pendingFiles.push(...validFiles);
    updatePendingFilesUI();
    showBatchUploadToolbar();

    showToast(`已添加 ${validFiles.length} 个文件到上传队列`, 'success');
}

// 显示批量上传工具栏
function showBatchUploadToolbar() {
    if (batchUploadToolbar) batchUploadToolbar.style.display = 'flex';
    if (pendingFilesContainer) pendingFilesContainer.style.display = 'block';
}

// 隐藏批量上传工具栏
function hideBatchUploadToolbar() {
    if (pendingFiles.length === 0) {
        if (batchUploadToolbar) batchUploadToolbar.style.display = 'none';
        if (pendingFilesContainer) pendingFilesContainer.style.display = 'none';
    }
}

// 更新待上传文件列表 UI
function updatePendingFilesUI() {
    if (!pendingFilesList) return;

    if (pendingFiles.length === 0) {
        pendingFilesList.innerHTML = '<div style="text-align: center; padding: 20px; color: #999;">暂无待上传文件</div>';
        hideBatchUploadToolbar();
        return;
    }

    const totalSize = pendingFiles.reduce((sum, f) => sum + f.size, 0);
    if (batchFileCountSpan) batchFileCountSpan.textContent = `已选择 ${pendingFiles.length} 个文件`;
    if (batchTotalSizeSpan) batchTotalSizeSpan.textContent = formatFileSize(totalSize);

    let html = '';
    for (const file of pendingFiles) {
        html += `
            <div class="pending-file-item" data-filename="${escapeHtml(file.name)}">
                <div class="pending-file-info">
                    <span class="pending-file-icon">${file.icon}</span>
                    <span class="pending-file-name">${escapeHtml(file.name)}</span>
                    <span class="pending-file-size">(${formatFileSize(file.size)})</span>
                    <span class="pending-file-type">${getMediaTypeBadge(file.mediaType)}</span>
                </div>
                <span class="pending-file-remove" data-filename="${escapeHtml(file.name)}">✕</span>
            </div>
        `;
    }

    pendingFilesList.innerHTML = html;

    pendingFilesList.querySelectorAll('.pending-file-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const filename = btn.dataset.filename;
            removeFileFromPending(filename);
        });
    });

    pendingFilesList.querySelectorAll('.pending-file-item').forEach(item => {
        item.addEventListener('click', (e) => {
            if (e.target.classList.contains('pending-file-remove')) return;
            const filename = item.dataset.filename;
            removeFileFromPending(filename);
        });
    });
}

// 从待上传队列移除文件
function removeFileFromPending(filename) {
    pendingFiles = pendingFiles.filter(f => f.name !== filename);
    updatePendingFilesUI();
}

// 清空待上传队列
function clearPendingFiles() {
    if (pendingFiles.length === 0) return;

    if (isUploading) {
        showToast('请等待上传完成后再清空', 'warning');
        return;
    }

    if (confirm(`确定要清空 ${pendingFiles.length} 个待上传文件吗？`)) {
        pendingFiles = [];
        updatePendingFilesUI();
        hideBatchUploadToolbar();
    }
}

// 显示确认模态框
function showConfirmModal() {
    if (pendingFiles.length === 0) {
        showToast('请先选择要上传的文件', 'warning');
        return;
    }

    if (!batchConfirmModal) return;

    if (confirmFileCountSpan) confirmFileCountSpan.textContent = pendingFiles.length;

    let html = '';
    for (const file of pendingFiles) {
        html += `
            <div class="confirm-file-item">
                <span>${file.icon} ${escapeHtml(file.name)}</span>
                <span>${formatFileSize(file.size)}</span>
                <span style="font-size: 10px; color: #6c757d;">${file.hint}</span>
            </div>
        `;
    }
    if (confirmFileList) confirmFileList.innerHTML = html;

    batchConfirmModal.style.display = 'flex';
}

// 隐藏模态框
function hideModal() {
    if (batchConfirmModal) batchConfirmModal.style.display = 'none';
}

// 上传单个文件
async function uploadSingleFile(fileItem, options) {
    const file = fileItem.file;
    const fileName = fileItem.name;
    const mediaType = fileItem.mediaType;

    return new Promise((resolve) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('chunk_size', options.chunk_size);
        formData.append('from_page', options.from_page);
        formData.append('to_page', options.to_page);
        formData.append('enable_vectorization', options.enable_vectorization);
        formData.append('enable_storage', options.enable_storage);

        const xhr = new XMLHttpRequest();

        xhr.onload = () => {
            if (xhr.status === 401) {
                resolve({ success: false, error: '登录已过期', filename: fileName });
                return;
            }

            if (xhr.status < 200 || xhr.status >= 300) {
                resolve({ success: false, error: `HTTP ${xhr.status}`, filename: fileName });
                return;
            }

            try {
                const data = JSON.parse(xhr.responseText);
                if (data.success) {
                    resolve({
                        success: true,
                        task_id: data.task_id,
                        filename: fileName,
                        media_type: data.media_type || mediaType
                    });
                } else {
                    resolve({ success: false, error: data.detail || data.message || '上传失败', filename: fileName });
                }
            } catch (e) {
                resolve({ success: false, error: '解析响应失败', filename: fileName });
            }
        };

        xhr.onerror = () => {
            resolve({ success: false, error: '网络错误', filename: fileName });
        };

        xhr.ontimeout = () => {
            resolve({ success: false, error: '请求超时', filename: fileName });
        };

        const headers = getAuthHeaders();
        xhr.open('POST', '/api/upload/async');
        if (headers.Authorization) {
            xhr.setRequestHeader('Authorization', headers.Authorization);
        }
        xhr.timeout = 120000;
        xhr.send(formData);
    });
}

// 轮询单个文件状态
async function pollFileStatus(fileInfo) {
    const fileName = fileInfo.filename;
    const taskId = fileInfo.task_id;

    return new Promise((resolve) => {
        let pollCount = 0;
        let isCompleted = false;
        const maxPolls = 90;

        const interval = setInterval(async () => {
            if (isCompleted) return;

            pollCount++;

            try {
                const response = await fetch(`${API_BASE}/upload/task/${taskId}`, {
                    headers: getAuthHeaders(),
                    cache: 'no-store'
                });

                if (!response.ok) {
                    if (pollCount >= maxPolls) {
                        clearInterval(interval);
                        updateProgressItem(fileName, 'failed', 100, '✗ 状态查询超时');
                        uploadResults.failed.push({ name: fileName, error: '状态查询超时' });
                        isCompleted = true;
                        resolve();
                    }
                    return;
                }

                const statusData = await response.json();

                if (statusData.status === 'completed') {
                    clearInterval(interval);
                    updateProgressItem(fileName, 'completed', 100, '✓ 处理完成');
                    uploadResults.completed.push({ name: fileName, result: statusData.result });
                    isCompleted = true;
                    resolve();

                } else if (statusData.status === 'failed') {
                    clearInterval(interval);
                    const errorMsg = statusData.error || statusData.message || '处理失败';
                    updateProgressItem(fileName, 'failed', 100, `✗ ${errorMsg}`);
                    uploadResults.failed.push({ name: fileName, error: errorMsg });
                    isCompleted = true;
                    resolve();

                } else if (statusData.status === 'processing') {
                    const progressPercent = statusData.progress || 50;
                    let statusMsg = '⚙️ 处理中...';
                    if (statusData.media_type === 'image') statusMsg = '🖼️ 识别中...';
                    else if (statusData.media_type === 'audio') statusMsg = '🎵 ASR转写中...';
                    else if (statusData.media_type === 'video') statusMsg = '🎬 视频处理中...';
                    updateProgressItem(fileName, 'processing', progressPercent, statusMsg);
                    pollCount = 0;

                } else if (pollCount >= maxPolls) {
                    clearInterval(interval);
                    updateProgressItem(fileName, 'failed', 100, '✗ 处理超时');
                    uploadResults.failed.push({ name: fileName, error: '处理超时' });
                    isCompleted = true;
                    resolve();
                }

            } catch (error) {
                console.error(`获取 ${fileName} 状态失败:`, error);
                if (pollCount >= maxPolls) {
                    clearInterval(interval);
                    updateProgressItem(fileName, 'failed', 100, `✗ ${error.message}`);
                    uploadResults.failed.push({ name: fileName, error: error.message });
                    isCompleted = true;
                    resolve();
                }
            }
        }, 2000);

        setTimeout(() => {
            if (!isCompleted) {
                clearInterval(interval);
                updateProgressItem(fileName, 'failed', 100, '✗ 处理超时');
                uploadResults.failed.push({ name: fileName, error: '处理超时' });
                isCompleted = true;
                resolve();
            }
        }, 180000);
    });
}

// 启动批量上传
export async function startBatchUpload() {
    if (pendingFiles.length === 0) return;
    if (isUploading) {
        showToast('已有上传任务正在进行', 'warning');
        return;
    }

    isUploading = true;
    uploadResults = { completed: [], failed: [] };

    showProgressUI();

    const chunkSize = document.getElementById('chunkSize')?.value || 256;
    const fromPage = document.getElementById('fromPage')?.value || 0;
    const toPage = document.getElementById('toPage')?.value || 100000;
    const enableVectorization = document.getElementById('enableVectorization')?.checked ?? true;
    const enableStorage = document.getElementById('enableStorage')?.checked ?? true;

    const filesToUpload = [...pendingFiles];
    const total = filesToUpload.length;

    initProgressItems(filesToUpload);
    updateGlobalProgress(0, total, '正在上传文件...');

    const options = {
        chunk_size: parseInt(chunkSize),
        from_page: parseInt(fromPage),
        to_page: parseInt(toPage),
        enable_vectorization: enableVectorization,
        enable_storage: enableStorage
    };

    // 上传所有文件
    const uploadPromises = filesToUpload.map(async (fileItem) => {
        updateProgressItem(fileItem.name, 'uploading', 0, '📤 上传中 0%');

        const uploadResult = await uploadSingleFile(fileItem, options);

        if (!uploadResult.success) {
            updateProgressItem(fileItem.name, 'failed', 100, `✗ ${uploadResult.error}`);
            uploadResults.failed.push({ name: fileItem.name, error: uploadResult.error });
            updateGlobalProgress(uploadResults.completed.length + uploadResults.failed.length, total);
            return null;
        }

        updateProgressItem(fileItem.name, 'processing', 30, '✓ 上传成功，处理中...');
        updateGlobalProgress(uploadResults.completed.length + uploadResults.failed.length, total);

        return {
            filename: uploadResult.filename,
            task_id: uploadResult.task_id,
            media_type: uploadResult.media_type
        };
    });

    const uploadResults_array = await Promise.all(uploadPromises);
    const successfulUploads = uploadResults_array.filter(r => r !== null);

    // 轮询所有文件状态
    if (successfulUploads.length > 0) {
        const pollPromises = successfulUploads.map(fileInfo => pollFileStatus(fileInfo));
        await Promise.all(pollPromises);
    }

    // 完成
    isUploading = false;

    const succeededCount = uploadResults.completed.length;
    const failedCount = uploadResults.failed.length;

    if (failedCount > 0) {
        let errorMsg = `${succeededCount} 个文件处理成功，${failedCount} 个失败`;
        const failedFiles = uploadResults.failed.map(f => `  • ${f.name}: ${f.error}`).join('\n');
        showToast(`${errorMsg}\n\n失败的文件:\n${failedFiles}`, 'error', 8000);
    } else if (succeededCount > 0) {
        showToast(`✅ 成功处理 ${succeededCount} 个文件`, 'success');
    }

    pendingFiles = [];
    updatePendingFilesUI();
    hideBatchUploadToolbar();

    await loadFileList();

    setTimeout(() => {
        hideProgressUI();
    }, 3000);
}

// 显示进度 UI
function showProgressUI() {
    if (globalProgressContainer) {
        globalProgressContainer.style.display = 'block';
        if (globalProgressFill) globalProgressFill.style.width = '0%';
    }
    if (uploadProgressList) {
        uploadProgressList.style.display = 'block';
        if (uploadProgressItems) {
            uploadProgressItems.style.display = 'block';
        }
    }
}

// 隐藏进度 UI
function hideProgressUI() {
    if (globalProgressContainer) globalProgressContainer.style.display = 'none';
    if (uploadProgressList) uploadProgressList.style.display = 'none';
}

// 初始化进度项
function initProgressItems(files) {
    if (!uploadProgressItems) return;

    let html = '';
    for (const file of files) {
        const fileName = file.name;
        const fileId = escapeHtml(fileName).replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '_');
        const mediaType = file.mediaType;
        let typeIcon = '';
        if (mediaType === 'image') typeIcon = '🖼️';
        else if (mediaType === 'audio') typeIcon = '🎵';
        else if (mediaType === 'video') typeIcon = '🎬';
        else typeIcon = '📄';

        html += `
            <div class="upload-progress-item pending" id="upload-item-${fileId}" data-filename="${escapeHtml(fileName)}">
                <div class="file-info">
                    <span class="file-name" title="${escapeHtml(fileName)}">${typeIcon} ${escapeHtml(fileName.length > 30 ? fileName.substring(0, 27) + '...' : fileName)}</span>
                    <span class="file-status" id="upload-status-${fileId}">⏳ 等待中</span>
                </div>
                <div class="progress-bar-small">
                    <div class="progress-fill-small" id="upload-progress-${fileId}" style="width: 0%"></div>
                </div>
            </div>
        `;
    }

    uploadProgressItems.innerHTML = html;
}

// 更新进度项
function updateProgressItem(filename, status, percent, message = '') {
    const fileId = escapeHtml(filename).replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '_');
    const statusSpan = document.getElementById(`upload-status-${fileId}`);
    const progressFill = document.getElementById(`upload-progress-${fileId}`);
    const itemDiv = document.getElementById(`upload-item-${fileId}`);

    if (itemDiv) {
        itemDiv.classList.remove('pending', 'uploading', 'processing', 'completed', 'failed');
        itemDiv.classList.add(status);
    }

    if (statusSpan) {
        let statusText = message || getStatusText(status, percent);
        statusSpan.textContent = statusText;
    }

    if (progressFill) {
        let widthPercent = percent;
        if (status === 'completed') widthPercent = 100;
        if (status === 'failed') widthPercent = 100;
        if (status === 'pending') widthPercent = 0;

        progressFill.style.width = `${widthPercent}%`;

        if (status === 'completed') {
            progressFill.style.background = '#28a745';
        } else if (status === 'failed') {
            progressFill.style.background = '#dc3545';
        } else if (status === 'processing') {
            progressFill.style.background = '#ffc107';
        } else if (status === 'uploading') {
            progressFill.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
        } else {
            progressFill.style.background = '#e0e0e0';
        }
    }
}

function getStatusText(status, percent) {
    if (status === 'pending') return '⏳ 等待中';
    if (status === 'uploading') return `📤 上传中 ${percent}%`;
    if (status === 'processing') return `⚙️ 处理中 ${percent}%`;
    if (status === 'completed') return '✓ 处理完成';
    if (status === 'failed') return '✗ 处理失败';
    return '⏳ 等待中';
}

// 更新全局进度
function updateGlobalProgress(completed, total, statusMessage = null) {
    const percent = total > 0 ? Math.round((completed / total) * 100) : 0;

    if (globalProgressFill) {
        globalProgressFill.style.width = `${percent}%`;
    }

    if (globalProgressStats) {
        globalProgressStats.textContent = `已完成: ${completed} / ${total}`;
    }

    if (globalProgressStatus) {
        if (statusMessage) {
            globalProgressStatus.textContent = statusMessage;
        } else if (completed === total && total > 0) {
            globalProgressStatus.textContent = '上传完成！';
        } else if (completed > 0) {
            globalProgressStatus.textContent = `正在上传 ${completed}/${total}`;
        } else {
            globalProgressStatus.textContent = '准备中...';
        }
    }
}

// 重置批量上传状态
export function resetBatchUpload() {
    pendingFiles = [];
    isUploading = false;
    updatePendingFilesUI();
    hideBatchUploadToolbar();
    hideProgressUI();
}

// 获取待上传文件数量
export function getPendingFilesCount() {
    return pendingFiles.length;
}

// 检查是否有正在上传的任务
export function isUploadingFiles() {
    return isUploading;
}