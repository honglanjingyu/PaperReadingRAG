// app/web/js/chat/utils.js
// 确保 formatContent 函数正确导出

// HTML转义
export function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// JS字符串转义
export function escapeJs(text) {
    return text.replace(/'/g, "\\'").replace(/"/g, '\\"');
}

// 滚动到底部
export function scrollToBottom(container) {
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

// 格式化内容（支持Markdown简单解析）
export function formatContent(content) {
    if (!content) return '';

    let html = escapeHtml(content);

    // 代码块
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // 行内代码
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // 粗体
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // 斜体
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // 标题（# 开头）
    html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.*?)$/gm, '<h1>$1</h1>');

    // 无序列表
    html = html.replace(/^- (.*?)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*?<\/li>)/gs, '<ul>$1</ul>');

    // 有序列表
    html = html.replace(/^\d+\. (.*?)$/gm, '<li>$1</li>');

    // 引用块
    html = html.replace(/^> (.*?)$/gm, '<blockquote>$1</blockquote>');

    // 换行转br（但不影响块级元素内的换行）
    html = html.replace(/\n/g, '<br>');

    // 清理多余的 br 标签
    html = html.replace(/<\/h\d><br>/g, '</h\d>');
    html = html.replace(/<\/ul><br>/g, '</ul>');
    html = html.replace(/<\/blockquote><br>/g, '</blockquote>');

    return html;
}

// 显示提示消息
export function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// 显示会话提示
export function showSessionToast(message, type = 'info') {
    const existingToasts = document.querySelectorAll('.session-toast');
    existingToasts.forEach(toast => toast.remove());

    const toast = document.createElement('div');
    toast.className = `session-toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}