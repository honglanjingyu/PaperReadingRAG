// app/web/js/upload/batch-upload.js
// 批量上传核心模块 - 串行上传版

import { elements, state, updateSelectedFiles } from './config.js';
import { getAuthHeaders, showToast, escapeHtml, formatFileSize, logout } from './utils.js';
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

// 添加文件到待上传队列
export function addFilesToPending(files) {
    const validFiles = [];
    const supportedExts = ['.pdf', '.docx', '.txt', '.md', '.markdown'];

    for (const file of files) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!supportedExts.includes(ext)) {
            showToast(`跳过不支持的文件: ${file.name} (支持: ${supportedExts.join(', ')})`, 'warning');
            continue;
        }

        if (file.size > 50 * 1024 * 1024) {
            showToast(`跳过超大文件: ${file.name} (最大 50MB)`, 'warning');
            continue;
        }

        // 检查是否已在队列中
        const exists = pendingFiles.some(f => f.name === file.name);
        if (exists) {
            showToast(`文件已在队列中: ${file.name}`, 'warning');
            continue;
        }

        validFiles.push({
            name: file.name,
            size: file.size,
            file: file,
            status: 'pending'
        });
    }

    if (validFiles.length === 0) return;

    pendingFiles.push(...validFiles);
    updatePendingFilesUI();
    showBatchUploadToolbar();
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
                    <span class="pending-file-icon">${getFileIconByName(file.name)}</span>
                    <span class="pending-file-name">${escapeHtml(file.name)}</span>
                    <span class="pending-file-size">(${formatFileSize(file.size)})</span>
                </div>
                <span class="pending-file-remove" data-filename="${escapeHtml(file.name)}">✕</span>
            </div>
        `;
    }

    pendingFilesList.innerHTML = html;

    // 绑定移除事件
    pendingFilesList.querySelectorAll('.pending-file-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const filename = btn.dataset.filename;
            removeFileFromPending(filename);
        });
    });

    // 点击整个项也可以移除
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
                <span>📄 ${escapeHtml(file.name)}</span>
                <span>${formatFileSize(file.size)}</span>
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

// 获取文件图标
function getFileIconByName(filename) {
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

export async function startBatchUpload() {
    if (pendingFiles.length === 0) return;
    if (isUploading) {
        showToast('已有上传任务正在进行', 'warning');
        return;
    }

    isUploading = true;
    uploadResults = { completed: [], failed: [] };

    // 显示进度界面
    showProgressUI();

    // 获取配置参数
    const chunkSize = document.getElementById('chunkSize')?.value || 256;
    const fromPage = document.getElementById('fromPage')?.value || 0;
    const toPage = document.getElementById('toPage')?.value || 100000;
    const enableVectorization = document.getElementById('enableVectorization')?.checked ?? true;
    const enableStorage = document.getElementById('enableStorage')?.checked ?? true;

    const filesToUpload = [...pendingFiles];
    const total = filesToUpload.length;

    // 初始化进度项
    initProgressItems(filesToUpload);

    updateGlobalProgress(0, total, '正在上传文件...');

    // ========== 用于追踪每个文件的状态 ==========
    const uploadedFiles = [];  // 上传成功的文件
    const uploadedCountMap = { completed: 0, failed: 0 };

    // 存储每个文件的轮询控制器（用于可选的中止功能）
    const pollControllers = [];

    // ========== 为每个文件创建独立的上传和轮询任务 ==========
    const uploadTasks = filesToUpload.map(async (fileItem) => {
        const file = fileItem.file;
        const fileName = fileItem.name;

        updateProgressItem(fileName, 'uploading', 0, '📤 上传中 0%');

        try {
            // 1. 上传文件
            const result = await uploadSingleFile(file, {
                chunk_size: parseInt(chunkSize),
                from_page: parseInt(fromPage),
                to_page: parseInt(toPage),
                enable_vectorization: enableVectorization,
                enable_storage: enableStorage
            });

            if (!result.success) {
                // 上传失败
                uploadedCountMap.failed++;
                updateProgressItem(fileName, 'failed', 100, `✗ ${result.error}`);
                uploadResults.failed.push({
                    name: fileName,
                    error: result.error
                });
                updateGlobalProgress(
                    uploadedCountMap.completed + uploadedCountMap.failed,
                    total,
                    `上传完成: ${uploadedCountMap.completed}/${total}`
                );
                return;
            }

            // 2. 上传成功，立即开始轮询处理状态
            uploadedCountMap.completed++;
            updateProgressItem(fileName, 'completed', 100, '✓ 上传成功，处理中...');

            // 记录上传成功的文件
            uploadedFiles.push({
                name: fileName,
                process_id: result.process_id
            });

            updateGlobalProgress(
                uploadedCountMap.completed + uploadedCountMap.failed,
                total,
                `上传完成: ${uploadedCountMap.completed}/${total}`
            );

            // 3. 立即开始轮询这个文件的处理状态（不等待其他文件）
            await pollSingleFileStatus({
                name: fileName,
                process_id: result.process_id
            });

        } catch (error) {
            uploadedCountMap.failed++;
            updateProgressItem(fileName, 'failed', 100, `✗ ${error.message}`);
            uploadResults.failed.push({
                name: fileName,
                error: error.message
            });
            updateGlobalProgress(
                uploadedCountMap.completed + uploadedCountMap.failed,
                total,
                `上传完成: ${uploadedCountMap.completed}/${total}`
            );
        }
    });

    // 等待所有上传和轮询任务完成
    await Promise.all(uploadTasks);

    // ========== 全部完成 ==========
    isUploading = false;

    const succeededCount = uploadResults.completed.length;
    const finalFailedCount = uploadResults.failed.length;

    if (finalFailedCount > 0) {
        let errorMsg = `${succeededCount} 个文件处理成功，${finalFailedCount} 个失败`;
        const failedFiles = uploadResults.failed.map(f => `  • ${f.name}: ${f.error}`).join('\n');
        showToast(`${errorMsg}\n\n失败的文件:\n${failedFiles}`, 'error', 8000);
    } else if (succeededCount > 0) {
        showToast(`✅ 成功处理 ${succeededCount} 个文件`, 'success');
    }

    // 清空待上传队列
    pendingFiles = [];
    updatePendingFilesUI();
    hideBatchUploadToolbar();

    // 刷新文件列表
    await loadFileList();

    // 延迟隐藏进度详情
    setTimeout(() => {
        hideProgressUI();
    }, 3000);
}
// app/web/js/upload/batch-upload.js

async function pollSingleFileStatus(file) {
    const fileName = file.name;
    const processId = file.process_id;

    return new Promise((resolve) => {
        let pollCount = 0;
        let isCompleted = false;

        const interval = setInterval(async () => {
            if (isCompleted) return;

            try {
                // 修复：使用正确的接口路径
                const statusUrl = `/api/upload/task/${processId}`;
                console.log(`轮询状态 [${fileName}]: ${statusUrl}`);

                const response = await fetch(statusUrl, {
                    headers: getAuthHeaders()
                });

                console.log(`状态响应状态码 [${fileName}]:`, response.status);

                if (!response.ok) {
                    console.warn(`状态接口返回 ${response.status}: ${fileName}`);
                    pollCount++;
                    if (pollCount >= 15) { // 30秒超时
                        clearInterval(interval);
                        updateProgressItem(fileName, 'failed', 100, '✗ 状态查询超时');
                        uploadResults.failed.push({
                            name: fileName,
                            error: '状态查询超时'
                        });
                        isCompleted = true;
                        resolve();
                    }
                    return;
                }

                const statusData = await response.json();
                console.log(`${fileName} 状态数据:`, statusData);

                // 修复：检查 statusData.status 字段
                if (statusData.status === 'completed') {
                    clearInterval(interval);
                    updateProgressItem(fileName, 'completed', 100, '✓ 处理完成');
                    uploadResults.completed.push({
                        name: fileName,
                        result: statusData.result
                    });
                    isCompleted = true;
                    resolve();

                } else if (statusData.status === 'failed') {
                    clearInterval(interval);
                    const errorMsg = statusData.error || statusData.message || '处理失败';
                    updateProgressItem(fileName, 'failed', 100, `✗ ${errorMsg}`);
                    uploadResults.failed.push({
                        name: fileName,
                        error: errorMsg
                    });
                    isCompleted = true;
                    resolve();

                } else if (statusData.status === 'processing') {
                    // 处理中，更新进度
                    const progressPercent = statusData.progress || 50;
                    updateProgressItem(fileName, 'processing', progressPercent, `⚙️ ${statusData.message || '处理中...'}`);
                    pollCount = 0; // 重置计数
                } else {
                    // 未知状态
                    pollCount++;
                    if (pollCount >= 15) {
                        clearInterval(interval);
                        updateProgressItem(fileName, 'failed', 100, '✗ 状态查询超时');
                        uploadResults.failed.push({
                            name: fileName,
                            error: '状态查询超时'
                        });
                        isCompleted = true;
                        resolve();
                    }
                }

            } catch (error) {
                console.error(`获取 ${fileName} 状态失败:`, error);
                pollCount++;
                if (pollCount >= 15) {
                    clearInterval(interval);
                    updateProgressItem(fileName, 'failed', 100, `✗ ${error.message}`);
                    uploadResults.failed.push({
                        name: fileName,
                        error: error.message
                    });
                    isCompleted = true;
                    resolve();
                }
            }
        }, 2000); // 每2秒轮询

        // 设置总超时（5分钟）
        setTimeout(() => {
            if (!isCompleted) {
                console.log(`${fileName} 轮询超时`);
                clearInterval(interval);
                updateProgressItem(fileName, 'failed', 100, '✗ 处理超时');
                uploadResults.failed.push({
                    name: fileName,
                    error: '处理超时'
                });
                isCompleted = true;
                resolve();
            }
        }, 300000);
    });
}
async function pollProcessingStatus(uploadedFiles) {
    const total = uploadedFiles.length;
    let completed = 0;
    let failed = 0;

    // 记录每个文件的状态
    const fileStatus = {};
    for (const file of uploadedFiles) {
        fileStatus[file.name] = {
            process_id: file.process_id,
            status: 'processing',
            pollCount: 0,
            lastStatus: null
        };
    }

    return new Promise((resolve) => {
        const checkInterval = setInterval(async () => {
            let allDone = true;

            for (const file of uploadedFiles) {
                const status = fileStatus[file.name];
                if (status.status !== 'completed' && status.status !== 'failed') {
                    allDone = false;

                    try {
                        // 使用正确的状态接口
                        const statusUrl = `/api/upload/task/${status.process_id}`;
                        console.log(`轮询状态: ${file.name} -> ${statusUrl}`);

                        const response = await fetch(statusUrl, {
                            headers: getAuthHeaders()
                        });

                        if (!response.ok) {
                            console.warn(`状态接口返回 ${response.status}: ${file.name}`);
                            continue;
                        }

                        const statusData = await response.json();
                        console.log(`${file.name} 状态数据:`, statusData);

                        if (statusData.status === 'completed') {
                            if (status.status !== 'completed') {
                                status.status = 'completed';
                                completed++;
                                updateProgressItem(file.name, 'completed', 100, '✓ 处理完成');
                                uploadResults.completed.push({
                                    name: file.name,
                                    result: statusData.result
                                });
                                console.log(`✅ ${file.name} 处理完成`);
                            }
                        } else if (statusData.status === 'failed') {
                            if (status.status !== 'failed') {
                                status.status = 'failed';
                                failed++;
                                const errorMsg = statusData.error || statusData.message || '处理失败';
                                updateProgressItem(file.name, 'failed', 100, `✗ ${errorMsg}`);
                                uploadResults.failed.push({
                                    name: file.name,
                                    error: errorMsg
                                });
                                console.log(`❌ ${file.name} 处理失败: ${errorMsg}`);
                            }
                        } else {
                            // 处理中，更新进度显示
                            status.pollCount++;
                            let progressPercent = 50;
                            if (statusData.progress && statusData.progress > 0) {
                                progressPercent = Math.min(95, 50 + Math.floor(statusData.progress * 0.45));
                            } else {
                                progressPercent = Math.min(95, 50 + Math.floor(status.pollCount * 0.5));
                            }
                            updateProgressItem(file.name, 'processing', progressPercent, `⚙️ ${statusData.message || '处理中...'}`);
                            console.log(`🔄 ${file.name} 处理中: ${statusData.message || ''} (${status.pollCount}次轮询)`);
                        }
                    } catch (error) {
                        console.error(`获取 ${file.name} 状态失败:`, error);
                    }
                }
            }

            // 更新全局进度
            updateGlobalProgress(completed + failed, total, `处理进度: ${completed + failed}/${total}`);

            // 日志输出当前进度
            console.log(`轮询进度: 完成=${completed}, 失败=${failed}, 总计=${total}`);

            // 检查是否全部完成
            if (allDone) {
                console.log('所有文件处理完成，停止轮询');
                clearInterval(checkInterval);
                resolve();
            }
        }, 2000); // 每2秒轮询一次

        // 设置总超时（5分钟）
        setTimeout(() => {
            console.log('轮询超时，强制停止');
            clearInterval(checkInterval);
            resolve();
        }, 300000);
    });
}
async function getTaskStatus(taskId) {
    // 使用 API_BASE
    const response = await fetch(`${API_BASE}/upload/task/${taskId}`, {
        headers: getAuthHeaders()
    });
    if (!response.ok) return null;
    return await response.json();
}
// 新增：上传单个文件的函数
async function uploadSingleFile(file, options) {
    return new Promise((resolve) => {
        const formData = new FormData();
        formData.append('file', file);

        // 添加配置参数
        formData.append('chunk_size', options.chunk_size);
        formData.append('from_page', options.from_page);
        formData.append('to_page', options.to_page);
        formData.append('enable_vectorization', options.enable_vectorization);
        formData.append('enable_storage', options.enable_storage);

        const xhr = new XMLHttpRequest();

        xhr.onload = () => {
            if (xhr.status === 401) {
                resolve({ success: false, error: '登录已过期' });
                return;
            }

            if (xhr.status < 200 || xhr.status >= 300) {
                resolve({ success: false, error: `HTTP ${xhr.status}` });
                return;
            }

            try {
                const data = JSON.parse(xhr.responseText);
                if (data.success) {
                    resolve({
                        success: true,
                        process_id: data.task_id,
                        filename: file.name
                    });
                } else {
                    resolve({ success: false, error: data.detail || data.message || '上传失败' });
                }
            } catch (e) {
                resolve({ success: false, error: '解析响应失败' });
            }
        };

        xhr.onerror = () => {
            resolve({ success: false, error: '网络错误' });
        };

        xhr.ontimeout = () => {
            resolve({ success: false, error: '请求超时' });
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

// 修改 initProgressItems 函数（用于批量上传）
function initProgressItems(files) {
    if (!uploadProgressItems) return;

    let html = '';
    for (const file of files) {
        const fileName = file.name;
        const fileId = escapeHtml(fileName).replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '_');
        html += `
            <div class="upload-progress-item pending" id="upload-item-${fileId}" data-filename="${escapeHtml(fileName)}">
                <div class="file-info">
                    <span class="file-name" title="${escapeHtml(fileName)}">${escapeHtml(fileName.length > 30 ? fileName.substring(0, 27) + '...' : fileName)}</span>
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
        let statusText = '';
        let displayPercent = percent;

        switch (status) {
            case 'pending':
                statusText = message || '⏳ 等待中';
                displayPercent = 0;
                break;
            case 'uploading':
                statusText = message || `📤 上传中 ${percent}%`;
                break;
            case 'processing':
                statusText = message || `⚙️ 处理中 ${percent}%`;
                break;
            case 'completed':
                statusText = message || '✓ 上传成功';
                displayPercent = 100;
                break;
            case 'failed':
                statusText = message || '✗ 上传失败';
                displayPercent = 100;
                break;
            default:
                statusText = message || '⏳ 等待中';
                displayPercent = 0;
        }
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

// 显示上传完成结果
function showUploadComplete() {
    const succeeded = uploadResults.completed.length;
    const failed = uploadResults.failed.length;

    if (failed > 0) {
        let errorMsg = `${succeeded} 个文件上传成功，${failed} 个失败`;
        const failedFiles = uploadResults.failed.map(f => `  • ${f.name}: ${f.error}`).join('\n');
        showToast(`${errorMsg}\n\n失败的文件:\n${failedFiles}`, 'error', 8000);
    } else if (succeeded > 0) {
        showToast(`✅ 成功上传 ${succeeded} 个文件`, 'success');
    }

    // ========== 新增：3秒后自动隐藏进度详情 ==========
    setTimeout(() => {
        hideProgressUI();
        // 同时重置全局进度显示
        if (globalProgressContainer) {
            globalProgressContainer.style.display = 'none';
        }
        if (uploadProgressList) {
            uploadProgressList.style.display = 'none';
        }
        console.log('上传进度详情已自动隐藏');
    }, 1000);  // 1秒后自动隐藏
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