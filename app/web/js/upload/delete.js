// app/web/js/upload/delete.js
// 删除相关逻辑模块

import { elements, state, updateSelectedFiles, getSelectedFiles } from './config.js';
import { showToast, logout, escapeHtml } from './utils.js';
import { deleteDocumentApi, batchDeleteApi, getUserRole } from './api.js';
import { loadFileList } from './file-list.js';

// 更新批量删除按钮状态和选中计数
export function updateBatchDeleteButton() {
    const selectedFiles = getSelectedFiles();

    if (elements.selectedCountSpan) {
        elements.selectedCountSpan.textContent = selectedFiles.size;
    }

    if (elements.batchDeleteBtn) {
        if (selectedFiles.size > 0) {
            elements.batchDeleteBtn.style.display = 'inline-block';
        } else {
            elements.batchDeleteBtn.style.display = 'none';
        }
    }

    // 更新全选复选框状态
    if (elements.selectAllCheckbox) {
        const checkboxes = document.querySelectorAll('.file-checkbox-input');
        if (checkboxes.length > 0) {
            const allChecked = Array.from(checkboxes).every(cb => cb.checked);
            const someChecked = Array.from(checkboxes).some(cb => cb.checked);
            elements.selectAllCheckbox.checked = allChecked;
            elements.selectAllCheckbox.indeterminate = someChecked && !allChecked;
        } else {
            elements.selectAllCheckbox.checked = false;
            elements.selectAllCheckbox.indeterminate = false;
        }
    }
}

// 删除单个文档
export async function deleteDocument(filename) {
    if (!filename) return;

    if (!confirm(`确定要删除 "${filename}" 吗？此操作不可恢复。`)) return;

    try {
        const data = await deleteDocumentApi(filename);

        if (data.success) {
            showToast(`已删除 ${filename}`, 'success');
            const selectedFiles = getSelectedFiles();
            selectedFiles.delete(filename);
            updateSelectedFiles(selectedFiles);
            updateBatchDeleteButton();
            setTimeout(() => {
                loadFileList();
            }, 500);
        } else {
            showToast(data.detail || '删除失败', 'error');
        }
    } catch (error) {
        console.error('删除失败:', error);
        showToast('删除失败: ' + error.message, 'error');
    }
}

// 进度对话框
let progressDialog = null;
let progressInterval = null;

function showDeleteProgress(total) {
    if (progressDialog) progressDialog.remove();

    progressDialog = document.createElement('div');
    progressDialog.className = 'delete-progress';
    progressDialog.innerHTML = `
        <div style="margin-bottom: 12px;">🗑️ 正在删除文档...</div>
        <div id="deleteProgressText" style="font-size: 12px; color: #ccc; margin-bottom: 12px;">准备中...</div>
        <div class="progress-bar" style="background: rgba(255,255,255,0.2);">
            <div id="deleteProgressFill" class="progress-fill" style="width: 0%; background: #4caf50;"></div>
        </div>
        <div id="deleteProgressPercent" style="font-size: 11px; margin-top: 8px; color: #aaa;">0%</div>
    `;
    document.body.appendChild(progressDialog);

    let percent = 0;
    if (progressInterval) clearInterval(progressInterval);
    progressInterval = setInterval(() => {
        if (percent < 90 && progressDialog) {
            percent += 3;
            const fill = progressDialog.querySelector('#deleteProgressFill');
            const percentSpan = progressDialog.querySelector('#deleteProgressPercent');
            if (fill) fill.style.width = `${percent}%`;
            if (percentSpan) percentSpan.textContent = `${percent}%`;
        }
    }, 200);
}

function hideDeleteProgress() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    if (progressDialog) {
        progressDialog.remove();
        progressDialog = null;
    }
}

// 批量删除文档
export async function batchDeleteDocuments() {
    const selectedFiles = getSelectedFiles();
    const filesToDelete = Array.from(selectedFiles);

    if (filesToDelete.length === 0) {
        showToast('请先勾选要删除的文档', 'warning');
        return;
    }

    const userLevel = await getUserRole();

    let confirmMsg = `确定要删除以下 ${filesToDelete.length} 个文档吗？\n\n${filesToDelete.join('\n')}\n\n此操作将同时删除：\n• 本地文件\n• Milvus 向量数据库\n• Elasticsearch 索引\n• Redis 对话历史\n• 相关缓存\n\n此操作不可恢复！`;

    if (userLevel === 'normal') {
        confirmMsg = `⚠️ 您当前是普通用户，只能删除自己上传的文档。\n\n确定要删除以下 ${filesToDelete.length} 个文档吗？\n${filesToDelete.join('\n')}`;
    } else if (userLevel === 'admin') {
        confirmMsg = `🔐 管理员权限：可以删除普通用户和您自己上传的文档。\n\n确定要删除以下 ${filesToDelete.length} 个文档吗？\n${filesToDelete.join('\n')}`;
    } else if (userLevel === 'owner') {
        confirmMsg = `👑 所有者权限：可以删除所有文档。\n\n确定要删除以下 ${filesToDelete.length} 个文档吗？\n${filesToDelete.join('\n')}`;
    }

    if (!confirm(confirmMsg)) return;

    showDeleteProgress(filesToDelete.length);

    try {
        const data = await batchDeleteApi(filesToDelete);

        hideDeleteProgress();

        if (data.success || data.success_count > 0) {
            const successCount = data.success_count;
            const failCount = data.fail_count;
            const permissionDenied = data.permission_denied || [];
            const notFound = data.not_found || [];

            let resultMessage = `批量删除完成\n\n✅ 成功: ${successCount} 个\n❌ 失败: ${failCount} 个`;

            if (permissionDenied.length > 0) {
                resultMessage += `\n\n🚫 无权限 (${permissionDenied.length}个):\n${permissionDenied.join('\n')}`;
            }

            if (notFound.length > 0) {
                resultMessage += `\n\n📭 不存在 (${notFound.length}个):\n${notFound.join('\n')}`;
            }

            showToast(resultMessage, successCount > 0 ? 'success' : 'error', 8000);

            updateSelectedFiles(new Set());
            updateBatchDeleteButton();
            await loadFileList();
        } else {
            showToast(data.message || '批量删除失败', 'error');
        }

    } catch (error) {
        hideDeleteProgress();
        console.error('批量删除失败:', error);
        showToast('批量删除失败: ' + error.message, 'error');
    }
}