// app/web/js/upload/config.js - 添加新元素
// 配置和DOM元素模块

export const API_BASE = '/api';

// DOM 元素引用
export const elements = {
    uploadArea: null,
    fileInput: null,
    progressContainer: null,
    progressFill: null,
    progressText: null,
    fileListDiv: null,
    chunkSize: null,
    fromPage: null,
    toPage: null,
    enableVectorization: null,
    enableStorage: null,
    batchDeleteBtn: null,
    selectAllCheckbox: null,
    selectedCountSpan: null,
    // 新增批量上传相关元素
    concurrentUpload: null,
    concurrencyLimit: null,
    globalProgressContainer: null,
    uploadProgressList: null
};

// 全局状态
export const state = {
    currentProcessId: null,
    statusInterval: null,
    selectedFiles: new Set(),
    progressDialog: null,
    progressInterval: null
};

// 初始化 DOM 元素
export function initElements() {
    elements.uploadArea = document.getElementById('uploadArea');
    elements.fileInput = document.getElementById('fileInput');
    elements.progressContainer = document.getElementById('progressContainer');
    elements.progressFill = document.getElementById('progressFill');
    elements.progressText = document.getElementById('progressText');
    elements.fileListDiv = document.getElementById('fileList');
    elements.chunkSize = document.getElementById('chunkSize');
    elements.fromPage = document.getElementById('fromPage');
    elements.toPage = document.getElementById('toPage');
    elements.enableVectorization = document.getElementById('enableVectorization');
    elements.enableStorage = document.getElementById('enableStorage');
    elements.batchDeleteBtn = document.getElementById('batchDeleteBtn');
    elements.selectAllCheckbox = document.getElementById('selectAllCheckbox');
    elements.selectedCountSpan = document.getElementById('selectedCount');

    // 新增
    elements.concurrentUpload = document.getElementById('concurrentUpload');
    elements.concurrencyLimit = document.getElementById('concurrencyLimit');
    elements.globalProgressContainer = document.getElementById('globalProgressContainer');
    elements.uploadProgressList = document.getElementById('uploadProgressList');
}

// 更新选中文件集合
export function updateSelectedFiles(filesSet) {
    state.selectedFiles = filesSet;
}

// 获取选中文件列表
export function getSelectedFiles() {
    return state.selectedFiles;
}