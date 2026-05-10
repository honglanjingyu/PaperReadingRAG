// app/web/js/upload/batch-upload.js
// 批量上传核心模块 - 串行上传版

import { elements, state, updateSelectedFiles } from './config.js';
import { getAuthHeaders, showToast, escapeHtml, formatFileSize, logout } from './utils.js';
import { loadFileList } from './file-list.js';

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

// ========== 批量上传核心逻辑 ==========

// 开始批量上传（串行模式）
export async function startBatchUpload() {
    if (pendingFiles.length === 0) return;
    if (isUploading) {
        showToast('已有上传任务正在进行', 'warning');
        return;
    }

    isUploading = true;
    uploadResults = { completed: [], failed: [] };
    currentUploadTasks = [];

    // 显示进度界面
    showProgressUI();

    // 初始化进度项（所有文件初始状态为等待中）
    initProgressItems();

    const filesToUpload = [...pendingFiles];
    const total = filesToUpload.length;
    let completedCount = 0;

    // 更新全局进度
    updateGlobalProgress(0, total, '准备上传...');

    // ========== 串行上传，逐个处理 ==========
    for (let i = 0; i < filesToUpload.length; i++) {
        const file = filesToUpload[i];

        // 更新全局状态
        updateGlobalProgress(completedCount, total, `正在上传第 ${i + 1}/${total} 个文件: ${file.name.substring(0, 30)}`);

        // 上传单个文件（等待完成）
        const result = await uploadSingleFileSerial(file, i);

        if (result.success) {
            completedCount++;
            uploadResults.completed.push(result);
        } else {
            uploadResults.failed.push({ name: result.name, error: result.error });
        }

        // 更新全局进度
        updateGlobalProgress(completedCount, total);
    }

    // 更新最终结果
    updateGlobalProgress(total, total, '上传完成！');

    // 显示完成结果
    showUploadComplete();

    // 清空待上传队列
    pendingFiles = [];
    updatePendingFilesUI();
    hideBatchUploadToolbar();

    // 刷新文件列表
    await loadFileList();

    // 多次刷新以确保数据同步
    if (uploadResults.completed.length > 0) {
        setTimeout(async () => {
            await loadFileList();
            console.log('批量上传完成，刷新文件列表');
        }, 1000);
        setTimeout(async () => {
            await loadFileList();
            console.log('批量上传完成，二次刷新文件列表');
        }, 3000);
    }

    isUploading = false;
}

// 串行上传单个文件（带模拟进度）
async function uploadSingleFileSerial(fileItem, index) {
    const { file, name } = fileItem;

    return new Promise((resolve) => {
        const xhr = new XMLHttpRequest();
        let uploadComplete = false;
        let simulatedProgress = 0;
        let progressInterval = null;
        let processId = null;
        let statusInterval = null;

        // 模拟缓慢增加的进度（0% -> 90%）
        const startSimulatedProgress = () => {
            simulatedProgress = 0;
            progressInterval = setInterval(() => {
                if (!uploadComplete && simulatedProgress < 90) {
                    // 缓慢增加，每秒增加 2-4%
                    const increment = Math.random() * 2 + 2;
                    simulatedProgress = Math.min(90, simulatedProgress + increment);
                    updateProgressItem(name, 'uploading', Math.floor(simulatedProgress), `📤 上传中 ${Math.floor(simulatedProgress)}%`);
                }
            }, 300);
        };

        const stopSimulatedProgress = () => {
            if (progressInterval) {
                clearInterval(progressInterval);
                progressInterval = null;
            }
        };

        // 真实上传进度
        xhr.upload.addEventListener('progress', (e) => {
            if (e.lengthComputable) {
                const realPercent = Math.min(Math.round((e.loaded / e.total) * 100), 90);
                // 如果真实进度大于模拟进度，使用真实进度
                if (realPercent > simulatedProgress) {
                    simulatedProgress = realPercent;
                    updateProgressItem(name, 'uploading', realPercent, `📤 上传中 ${realPercent}%`);
                }
            }
        });

        xhr.onload = () => {
            stopSimulatedProgress();

            if (xhr.status === 401) {
                updateProgressItem(name, 'failed', 100, '✗ 登录已过期');
                resolve({ success: false, name, error: '登录已过期' });
                return;
            }

            if (xhr.status === 413) {
                updateProgressItem(name, 'failed', 100, '✗ 文件过大');
                resolve({ success: false, name, error: '文件过大' });
                return;
            }

            if (xhr.status < 200 || xhr.status >= 300) {
                updateProgressItem(name, 'failed', 100, `✗ HTTP ${xhr.status}`);
                resolve({ success: false, name, error: `HTTP ${xhr.status}` });
                return;
            }

            try {
                const data = JSON.parse(xhr.responseText);
                if (data.success) {
                    processId = data.process_id;
                    updateProgressItem(name, 'processing', 95, '⚙️ 后端处理中...');

                    // 开始轮询后端处理状态
                    let pollCount = 0;
                    statusInterval = setInterval(async () => {
                        pollCount++;
                        try {
                            const headers = getAuthHeaders();
                            const statusResponse = await fetch(`/api/upload/status/${processId}`, {
                                headers: headers
                            });

                            if (statusResponse.ok) {
                                const statusData = await statusResponse.json();

                                if (statusData.status === 'completed') {
                                    // 处理完成
                                    clearInterval(statusInterval);
                                    updateProgressItem(name, 'completed', 100, '✓ 处理完成');
                                    resolve({ success: true, name, data: statusData });
                                } else if (statusData.status === 'failed') {
                                    clearInterval(statusInterval);
                                    updateProgressItem(name, 'failed', 100, `✗ ${statusData.error || '处理失败'}`);
                                    resolve({ success: false, name, error: statusData.error || '处理失败' });
                                } else {
                                    // 更新进度（根据轮询次数和时间估计）
                                    let progressPercent = 95;
                                    if (statusData.progress && statusData.progress > 0) {
                                        // 使用后端返回的进度
                                        progressPercent = Math.min(99, 40 + Math.floor(statusData.progress * 0.59));
                                    } else {
                                        // 根据轮询次数估算（最多轮询60次约120秒）
                                        progressPercent = Math.min(99, 95 + Math.floor(pollCount * 0.1));
                                    }
                                    updateProgressItem(name, 'processing', progressPercent, `⚙️ ${statusData.message || '处理中...'}`);
                                }
                            }
                        } catch (e) {
                            console.error('轮询状态失败:', e);
                            // 轮询失败次数过多，标记为失败
                            if (pollCount > 30) {
                                clearInterval(statusInterval);
                                updateProgressItem(name, 'failed', 100, '✗ 状态查询超时');
                                resolve({ success: false, name, error: '状态查询超时' });
                            }
                        }
                    }, 2000);

                    // 设置总超时（120秒）
                    setTimeout(() => {
                        if (statusInterval) {
                            clearInterval(statusInterval);
                            if (!uploadResults.completed.some(r => r.name === name)) {
                                updateProgressItem(name, 'failed', 100, '✗ 后端处理超时');
                                resolve({ success: false, name, error: '后端处理超时' });
                            }
                        }
                    }, 120000);

                } else {
                    updateProgressItem(name, 'failed', 100, `✗ ${data.detail || data.message || '上传失败'}`);
                    resolve({ success: false, name, error: data.detail || '上传失败' });
                }
            } catch (e) {
                updateProgressItem(name, 'failed', 100, '✗ 解析响应失败');
                resolve({ success: false, name, error: '解析响应失败' });
            }
        };

        xhr.onerror = () => {
            stopSimulatedProgress();
            updateProgressItem(name, 'failed', 100, '✗ 网络错误');
            resolve({ success: false, name, error: '网络错误' });
        };

        xhr.ontimeout = () => {
            stopSimulatedProgress();
            updateProgressItem(name, 'failed', 100, '✗ 请求超时');
            resolve({ success: false, name, error: '请求超时' });
        };

        // 启动模拟进度
        startSimulatedProgress();

        // 构建 FormData
        const formData = new FormData();
        formData.append('file', file);

        // 添加配置参数
        const chunkSizeElem = document.getElementById('chunkSize');
        const fromPageElem = document.getElementById('fromPage');
        const toPageElem = document.getElementById('toPage');
        const enableVectorizationElem = document.getElementById('enableVectorization');
        const enableStorageElem = document.getElementById('enableStorage');

        if (chunkSizeElem) formData.append('chunk_size', chunkSizeElem.value);
        if (fromPageElem) formData.append('from_page', fromPageElem.value);
        if (toPageElem) formData.append('to_page', toPageElem.value);
        if (enableVectorizationElem) formData.append('enable_vectorization', enableVectorizationElem.checked);
        if (enableStorageElem) formData.append('enable_storage', enableStorageElem.checked);

        // 发送请求
        const headers = getAuthHeaders();
        xhr.open('POST', '/api/upload');
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

// 初始化进度项（所有文件初始状态为等待中）
function initProgressItems() {
    if (!uploadProgressItems) return;

    let html = '';
    for (let i = 0; i < pendingFiles.length; i++) {
        const file = pendingFiles[i];
        const fileId = escapeHtml(file.name).replace(/[^a-zA-Z0-9\u4e00-\u9fa5]/g, '_');
        html += `
            <div class="upload-progress-item pending" id="upload-item-${fileId}" data-filename="${escapeHtml(file.name)}">
                <div class="file-info">
                    <span class="file-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name.length > 30 ? file.name.substring(0, 27) + '...' : file.name)}</span>
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