/* app/web/js/upload.js */
/* 上传页面逻辑 */

// DOM 元素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const fileListDiv = document.getElementById('fileList');

let currentProcessId = null;
let statusInterval = null;

// 上传文件
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('chunk_size', document.getElementById('chunkSize').value);
    formData.append('from_page', document.getElementById('fromPage').value);
    formData.append('to_page', document.getElementById('toPage').value);
    formData.append('enable_vectorization', document.getElementById('enableVectorization').checked);
    formData.append('enable_storage', document.getElementById('enableStorage').checked);

    showToast(`正在上传 ${file.name}...`, 'success');

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            currentProcessId = data.process_id;
            showProgress();
            startStatusPolling(data.process_id, file.name);
        } else {
            showToast(data.detail || '上传失败', 'error');
        }
    } catch (error) {
        showToast('上传失败: ' + error.message, 'error');
    }
}

// 显示进度条
function showProgress() {
    progressContainer.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = '正在处理...';
}

// 开始轮询状态
function startStatusPolling(processId, filename) {
    if (statusInterval) clearInterval(statusInterval);

    statusInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/upload/status/${processId}`);
            const data = await response.json();

            progressFill.style.width = `${data.progress}%`;
            progressText.textContent = data.message;

            if (data.status === 'completed') {
                clearInterval(statusInterval);
                setTimeout(() => {
                    progressContainer.style.display = 'none';
                    loadFileList();
                    showToast(`${filename} 处理完成！生成了 ${data.result?.chunks_count || 0} 个分块`, 'success');
                }, 1000);
            } else if (data.status === 'failed') {
                clearInterval(statusInterval);
                progressContainer.style.display = 'none';
                showToast(`${filename} 处理失败: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('获取状态失败:', error);
        }
    }, 1000);
}

// 加载文件列表
async function loadFileList() {
    try {
        const response = await fetch(`${API_BASE}/upload/list`);
        const data = await response.json();

        if (!data.success || data.total === 0) {
            fileListDiv.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">暂无文档，请上传</div>';
            return;
        }

        let html = '';
        for (const doc of data.documents) {
            html += `
                <div class="file-item">
                    <div class="file-info">
                        <span class="file-icon">${getFileIcon(doc.filename)}</span>
                        <div>
                            <div class="file-name">${escapeHtml(doc.filename)}</div>
                            <div class="file-size">${formatFileSize(doc.size)} · ${formatDate(doc.created)}</div>
                        </div>
                    </div>
                    <div>
                        <button class="delete-btn" onclick="deleteDocument('${escapeHtml(doc.filename)}')">🗑️</button>
                    </div>
                </div>
            `;
        }
        fileListDiv.innerHTML = html;

    } catch (error) {
        console.error('加载文件列表失败:', error);
        fileListDiv.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">加载失败</div>';
    }
}

// 删除文档
async function deleteDocument(filename) {
    if (!confirm(`确定要删除 ${filename} 吗？`)) return;

    try {
        const response = await fetch(`${API_BASE}/upload/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        const data = await response.json();

        if (data.success) {
            showToast(`已删除 ${filename}`, 'success');
            loadFileList();
        } else {
            showToast(data.detail || '删除失败', 'error');
        }
    } catch (error) {
        showToast('删除失败: ' + error.message, 'error');
    }
}

// 事件监听
function initEventListeners() {
    uploadArea.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) uploadFile(file);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files[0]) uploadFile(e.target.files[0]);
    });
}

// 页面初始化
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadFileList();
});