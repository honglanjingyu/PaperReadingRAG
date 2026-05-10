/* app/web/js/upload.js */
/* 上传页面逻辑 - 完整版 */

// DOM 元素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const fileListDiv = document.getElementById('fileList');

let currentProcessId = null;
let statusInterval = null;

// 获取认证头
function getAuthHeaders() {
    const token = localStorage.getItem('rag_token');
    if (token && token !== 'null' && token !== 'undefined') {
        return { 'Authorization': `Bearer ${token}` };
    }
    return {};
}

// 检查登录状态
function isLoggedIn() {
    const token = localStorage.getItem('rag_token');
    return token && token !== 'null' && token !== 'undefined';
}

// 退出登录
function logout() {
    localStorage.removeItem('rag_token');
    localStorage.removeItem('rag_user_id');
    localStorage.removeItem('rag_username');
    window.location.href = '/login.html';
}

// 显示提示消息
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// HTML转义
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 获取文件图标
function getFileIcon(filename) {
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

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// 格式化时间
function formatDate(timestamp) {
    if (!timestamp) return '未知';
    try {
        return new Date(timestamp * 1000).toLocaleString();
    } catch (e) {
        return '未知';
    }
}

// 获取等级徽章样式
function getLevelBadge(level) {
    if (level === 'admin') {
        return '<span class="level-badge admin">🔐 管理员</span>';
    } else if (level === 'owner') {
        return '<span class="level-badge owner">🔒 所有者</span>';
    }
    return '<span class="level-badge normal">📄 普通</span>';
}

// 显示进度条
function showProgress() {
    if (progressContainer) {
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = '正在处理...';
    }
}

// 开始轮询状态
function startStatusPolling(processId, filename) {
    if (statusInterval) clearInterval(statusInterval);

    statusInterval = setInterval(async () => {
        try {
            const headers = getAuthHeaders();
            const response = await fetch(`${API_BASE}/upload/status/${processId}`, {
                headers: headers
            });

            if (response.status === 401) {
                logout();
                return;
            }

            if (!response.ok) {
                console.error('状态查询失败:', response.status);
                return;
            }

            const data = await response.json();
            console.log('处理状态:', data.status, data.progress + '%');

            if (progressFill) {
                progressFill.style.width = `${data.progress}%`;
            }
            if (progressText) {
                progressText.textContent = data.message;
            }

            if (data.status === 'completed') {
                clearInterval(statusInterval);
                statusInterval = null;
                console.log('文档处理完成，刷新文件列表...');

                setTimeout(() => {
                    if (progressContainer) {
                        progressContainer.style.display = 'none';
                    }
                    loadFileList();  // 刷新文件列表
                    showToast(`${filename} 处理完成！生成了 ${data.result?.chunks_count || 0} 个分块`, 'success');
                }, 1000);

            } else if (data.status === 'failed') {
                clearInterval(statusInterval);
                statusInterval = null;
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
                showToast(`${filename} 处理失败: ${data.error || '未知错误'}`, 'error');
            }
        } catch (error) {
            console.error('获取状态失败:', error);
        }
    }, 1000);
}

// 上传文件
async function uploadFile(file) {
    // 验证文件类型
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    const supportedExts = ['.pdf', '.docx', '.txt', '.md', '.markdown'];
    if (!supportedExts.includes(ext)) {
        showToast(`不支持的文件类型: ${ext}，支持: ${supportedExts.join(', ')}`, 'error');
        return;
    }

    // 验证文件大小 (50MB)
    if (file.size > 50 * 1024 * 1024) {
        showToast('文件大小不能超过 50MB', 'error');
        return;
    }

    console.log(`上传文件: ${file.name}, 大小: ${formatFileSize(file.size)}`);

    const formData = new FormData();
    formData.append('file', file);

    const chunkSizeInput = document.getElementById('chunkSize');
    const fromPageInput = document.getElementById('fromPage');
    const toPageInput = document.getElementById('toPage');
    const enableVectorizationCheckbox = document.getElementById('enableVectorization');
    const enableStorageCheckbox = document.getElementById('enableStorage');

    if (chunkSizeInput) formData.append('chunk_size', chunkSizeInput.value);
    if (fromPageInput) formData.append('from_page', fromPageInput.value);
    if (toPageInput) formData.append('to_page', toPageInput.value);
    if (enableVectorizationCheckbox) formData.append('enable_vectorization', enableVectorizationCheckbox.checked);
    if (enableStorageCheckbox) formData.append('enable_storage', enableStorageCheckbox.checked);

    showToast(`正在上传 ${file.name}...`, 'success');

    try {
        const headers = getAuthHeaders();
        console.log('上传请求头:', headers);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            headers: headers,
            body: formData
        });

        console.log('上传响应状态:', response.status);

        if (response.status === 401) {
            showToast('登录已过期，请重新登录', 'error');
            logout();
            return;
        }

        const data = await response.json();
        console.log('上传响应数据:', data);

        if (data.success) {
            currentProcessId = data.process_id;
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

// 加载文件列表
async function loadFileList() {
    console.log('loadFileList 被调用');

    if (!fileListDiv) {
        console.error('fileListDiv 元素未找到');
        return;
    }

    try {
        const headers = getAuthHeaders();
        console.log('请求头:', headers);

        const response = await fetch(`${API_BASE}/upload/list`, {
            headers: headers
        });

        console.log('响应状态:', response.status);

        if (response.status === 401) {
            console.log('未授权，跳转登录');
            logout();
            return;
        }

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        console.log('服务器返回数据:', data);

        if (!data.success) {
            console.error('获取文件列表失败:', data);
            fileListDiv.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">加载失败，请刷新重试</div>';
            return;
        }

        console.log(`找到 ${data.total} 个文档`);

        if (data.total === 0 || !data.documents || data.documents.length === 0) {
            console.log('没有文档，显示提示信息');
            fileListDiv.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">📭 暂无文档，请上传</div>';
            return;
        }

        // 调试：打印每个文档的信息
        console.log('文档列表:');
        data.documents.forEach(doc => {
            console.log(`  - ${doc.filename}, 等级: ${doc.user_level}, 大小: ${formatFileSize(doc.size)}`);
        });

        let html = '';
    for (const doc of data.documents) {
        const createdDate = doc.created ? formatDate(doc.created) : '未知';
        html += `
            <div class="file-item" data-filename="${escapeHtml(doc.filename)}">
                <div class="file-info">
                    <span class="file-icon">${getFileIcon(doc.filename)}</span>
                    <div>
                        <div class="file-name">${escapeHtml(doc.filename)}</div>
                        <div class="file-size">${formatFileSize(doc.size)} · ${createdDate}</div>
                        <div class="file-level">${getLevelBadge(doc.user_level)}</div>
                    </div>
                </div>
                <div>
                    <button class="delete-btn" onclick="deleteDocument('${escapeHtml(doc.filename).replace(/'/g, "\\'")}')" title="删除文档">
                        🗑️
                    </button>
                </div>
            </div>
        `;
    }
        fileListDiv.innerHTML = html;
        console.log('文件列表渲染完成');

    } catch (error) {
        console.error('加载文件列表失败:', error);
        fileListDiv.innerHTML = '<div style="text-align: center; color: #999; padding: 32px;">❌ 加载失败: ' + escapeHtml(error.message) + '</div>';
    }
}

// 删除文档
async function deleteDocument(filename) {
    if (!filename) return;

    if (!confirm(`确定要删除 "${filename}" 吗？此操作不可恢复。`)) return;

    console.log(`删除文档: ${filename}`);

    try {
        const headers = getAuthHeaders();
        const response = await fetch(`${API_BASE}/upload/${encodeURIComponent(filename)}`, {
            method: 'DELETE',
            headers: headers
        });

        console.log('删除响应状态:', response.status);

        if (response.status === 401) {
            showToast('登录已过期，请重新登录', 'error');
            logout();
            return;
        }

        const data = await response.json();

        if (data.success) {
            showToast(`已删除 ${filename}`, 'success');
            // 延迟刷新，让用户看到删除效果
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

// 初始化事件监听
function initEventListeners() {
    if (!uploadArea || !fileInput) {
        console.error('上传区域或文件输入元素未找到');
        return;
    }

    // 点击上传区域
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // 拖拽上传
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
        if (file) {
            uploadFile(file);
        }
    });

    // 文件选择
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            uploadFile(e.target.files[0]);
        }
        // 清空 input，允许重复上传同一个文件
        fileInput.value = '';
    });

    console.log('事件监听初始化完成');
}

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
            roleSpan.textContent = '普通用户';
            roleSpan.className = 'user-role-badge normal';
            return;
        }

        const response = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            const data = await response.json();
            const role = data.role || 'normal';

            if (role === 'admin') {
                roleSpan.textContent = '👑 管理员';
                roleSpan.className = 'user-role-badge admin';
            } else if (role === 'owner') {
                roleSpan.textContent = '⭐ 所有者';
                roleSpan.className = 'user-role-badge owner';
            } else {
                roleSpan.textContent = '👤 普通用户';
                roleSpan.className = 'user-role-badge normal';
            }
        } else {
            roleSpan.textContent = '普通用户';
            roleSpan.className = 'user-role-badge normal';
        }
    } catch (error) {
        console.error('获取用户等级失败:', error);
        roleSpan.textContent = '普通用户';
        roleSpan.className = 'user-role-badge normal';
    }
}

// 页面初始化
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Upload 页面 DOMContentLoaded');

    // 显示用户名
    displayCurrentUser();
    await displayUserRole();

    // 绑定退出登录按钮
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
        console.log('未登录，跳转到登录页');
        window.location.href = '/login.html';
        return;
    }

    // 验证 token 有效性
    try {
        const token = localStorage.getItem('rag_token');
        const response = await fetch('/api/auth/verify', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        const data = await response.json();
        if (!data.success) {
            console.log('Token 无效，跳转到登录页');
            logout();
            return;
        }
        console.log('Token 验证通过，用户等级:', data.role);
    } catch (error) {
        console.error('Token 验证失败:', error);
        logout();
        return;
    }

    // 初始化事件监听
    initEventListeners();

    // 加载文件列表
    await loadFileList();

    console.log('Upload 页面初始化完成');
});

// 将 deleteDocument 挂载到 window 对象，供 onclick 调用
window.deleteDocument = deleteDocument;