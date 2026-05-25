// app/web/js/upload/file-list.js
// 文件列表渲染和管理模块 - 支持多模态

import { elements, state, updateSelectedFiles, getSelectedFiles } from './config.js';
import { escapeHtml, getFileIcon, formatFileSize, formatDate, getLevelBadgeHtml, getMediaType, getMediaTypeBadge } from './utils.js';
import { fetchFileList } from './api.js';
import { deleteDocument, updateBatchDeleteButton } from './delete.js';

// 单个复选框切换
export function onFileCheckboxChange(checkbox, filename) {
    const selectedFiles = getSelectedFiles();
    const fileItem = checkbox.closest('.file-item');

    if (checkbox.checked) {
        selectedFiles.add(filename);
        if (fileItem) fileItem.classList.add('selected');
    } else {
        selectedFiles.delete(filename);
        if (fileItem) fileItem.classList.remove('selected');
    }

    updateSelectedFiles(selectedFiles);
    updateBatchDeleteButton();
}

// 全选/取消全选
export function toggleSelectAll() {
    if (!elements.selectAllCheckbox) return;

    const isChecked = elements.selectAllCheckbox.checked;
    const checkboxes = document.querySelectorAll('.file-checkbox-input');
    const selectedFiles = getSelectedFiles();

    checkboxes.forEach(checkbox => {
        checkbox.checked = isChecked;
        const filename = checkbox.dataset.filename;
        if (isChecked) {
            selectedFiles.add(filename);
            const fileItem = checkbox.closest('.file-item');
            if (fileItem) fileItem.classList.add('selected');
        } else {
            selectedFiles.delete(filename);
            const fileItem = checkbox.closest('.file-item');
            if (fileItem) fileItem.classList.remove('selected');
        }
    });

    updateSelectedFiles(selectedFiles);

    if (elements.selectAllCheckbox) {
        elements.selectAllCheckbox.indeterminate = false;
    }

    updateBatchDeleteButton();
}

// 加载文件列表
export async function loadFileList() {
    console.log('loadFileList 被调用');

    if (!elements.fileListDiv) {
        console.error('fileListDiv 元素未找到');
        return;
    }

    try {
        const data = await fetchFileList();

        if (!data.success) {
            console.error('获取文件列表失败:', data);
            elements.fileListDiv.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">❌ 加载失败，请刷新重试</div>';
            return;
        }

        if (data.total === 0 || !data.documents || data.documents.length === 0) {
            elements.fileListDiv.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">📭 暂无文档，请上传</div>';
            if (elements.batchDeleteBtn) elements.batchDeleteBtn.style.display = 'none';
            if (elements.selectAllCheckbox) {
                elements.selectAllCheckbox.checked = false;
                elements.selectAllCheckbox.indeterminate = false;
            }
            return;
        }

        // 清空选中的文件
        updateSelectedFiles(new Set());

        let html = '';
        for (const doc of data.documents) {
            const createdDate = formatDate(doc.created);
            const escapedFilename = escapeHtml(doc.filename).replace(/'/g, "\\'");
            const mediaType = doc.media_type || getMediaType(doc.filename);
            const mediaBadge = getMediaTypeBadge(mediaType);

            // 显示 OCR/ASR 置信度（如果有）
            let confidenceInfo = '';
            if (doc.ocr_confidence && doc.ocr_confidence > 0) {
                confidenceInfo = `<span class="file-confidence">🔍 OCR: ${(doc.ocr_confidence * 100).toFixed(0)}%</span>`;
            } else if (doc.transcript_confidence && doc.transcript_confidence > 0) {
                confidenceInfo = `<span class="file-confidence">🎤 ASR: ${(doc.transcript_confidence * 100).toFixed(0)}%</span>`;
            }

            // 显示提取的文字长度
            let extractedInfo = '';
            if (doc.extracted_text_length && doc.extracted_text_length > 0) {
                extractedInfo = `<span class="file-extracted">📝 提取: ${doc.extracted_text_length}字符</span>`;
            }

            // 显示时长（音频/视频）
            let durationInfo = '';
            if (doc.duration_seconds && doc.duration_seconds > 0) {
                const minutes = Math.floor(doc.duration_seconds / 60);
                const seconds = Math.floor(doc.duration_seconds % 60);
                durationInfo = `<span class="file-duration">⏱️ ${minutes}:${seconds.toString().padStart(2, '0')}</span>`;
            }

            html += `
                <div class="file-item" data-filename="${escapeHtml(doc.filename)}">
                    <div class="file-checkbox">
                        <input type="checkbox" class="file-checkbox-input" data-filename="${escapeHtml(doc.filename)}">
                    </div>
                    <div class="file-info">
                        <span class="file-icon">${getFileIcon(doc.filename)}</span>
                        <div class="file-details">
                            <div class="file-name">
                                ${escapeHtml(doc.filename)}
                                ${mediaBadge}
                            </div>
                            <div class="file-meta">
                                <span class="file-size">📦 ${formatFileSize(doc.size)}</span>
                                <span class="file-date">📅 ${createdDate}</span>
                                ${confidenceInfo}
                                ${extractedInfo}
                                ${durationInfo}
                            </div>
                            <div class="file-level">${getLevelBadgeHtml(doc.user_level)}</div>
                        </div>
                    </div>
                    <button class="delete-btn" data-filename="${escapedFilename}" title="删除文档">
                        🗑️
                    </button>
                </div>
            `;
        }

        elements.fileListDiv.innerHTML = html;

        // 绑定复选框事件
        const checkboxes = document.querySelectorAll('.file-checkbox-input');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation();
                const filename = checkbox.dataset.filename;
                onFileCheckboxChange(checkbox, filename);
            });
        });

        // 绑定删除按钮事件
        const deleteBtns = document.querySelectorAll('.delete-btn');
        deleteBtns.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const filename = btn.dataset.filename;
                if (filename) {
                    deleteDocument(filename);
                }
            });
        });

        // 绑定全选事件
        if (elements.selectAllCheckbox) {
            const newSelectAll = elements.selectAllCheckbox.cloneNode(true);
            elements.selectAllCheckbox.parentNode.replaceChild(newSelectAll, elements.selectAllCheckbox);
            elements.selectAllCheckbox = newSelectAll;
            elements.selectAllCheckbox.addEventListener('change', toggleSelectAll);
        }

        updateBatchDeleteButton();

    } catch (error) {
        console.error('加载文件列表失败:', error);
        elements.fileListDiv.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">❌ 加载失败: ' + escapeHtml(error.message) + '</div>';
    }
}