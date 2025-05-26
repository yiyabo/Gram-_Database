/**
 * 抗革兰氏阴性菌预测系统 - 主JavaScript文件
 * 实现用户界面交互和数据处理功能
 */

// 在DOM加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素
    const fileForm = document.getElementById('fileForm');
    const textForm = document.getElementById('textForm');
    const fastaFile = document.getElementById('fastaFile');
    const fastaText = document.getElementById('fastaText');
    const dropZone = document.getElementById('dropZone');
    const loadExample = document.getElementById('loadExample');
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    const resetForms = document.getElementById('resetForms');
    const exportCSV = document.getElementById('exportCSV');
    const exportFASTA = document.getElementById('exportFASTA');
    
    // 存储预测结果
    let predictionResults = [];
    
    // 文件上传处理
    if (fastaFile) {
        fastaFile.addEventListener('change', function() {
            const fileName = this.files[0] ? this.files[0].name : '';
            const fileNameElement = dropZone.querySelector('.selected-file-name');
            if (fileNameElement) {
                fileNameElement.textContent = fileName;
            }
        });
    }
    
    // 拖放功能
    if (dropZone) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropZone.classList.add('dragover');
        }
        
        function unhighlight() {
            dropZone.classList.remove('dragover');
        }
        
        dropZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fastaFile.files = files;
                const fileName = files[0].name;
                const fileNameElement = dropZone.querySelector('.selected-file-name');
                if (fileNameElement) {
                    fileNameElement.textContent = fileName;
                }
            }
        }
    }
    
    // 加载示例数据
    if (loadExample) {
        loadExample.addEventListener('click', function() {
            fetch('/example')
                .then(response => response.json())
                .then(data => {
                    if (fastaText) {
                        fastaText.value = data.fasta;
                    }
                })
                .catch(error => {
                    console.error('加载示例数据失败:', error);
                    showError('加载示例数据失败，请重试');
                });
        });
    }
    
    // 文件表单提交
    if (fileForm) {
        fileForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (!fastaFile.files.length) {
                showError('请选择FASTA文件');
                return;
            }
            
            const formData = new FormData();
            formData.append('fasta_file', fastaFile.files[0]);
            
            submitPrediction(formData);
        });
    }
    
    // 文本表单提交
    if (textForm) {
        textForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            if (!fastaText.value.trim()) {
                showError('请输入FASTA序列数据');
                return;
            }
            
            const formData = new FormData();
            formData.append('fasta_text', fastaText.value);
            
            submitPrediction(formData);
        });
    }
    
    // 提交预测请求
    function submitPrediction(formData) {
        // 显示加载中
        showLoading();
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // 存储结果
            predictionResults = data.results;
            
            // 显示结果
            displayResults(data);
        })
        .catch(error => {
            console.error('预测请求失败:', error);
            showError('预测请求失败，请重试');
        });
    }
    
    // 显示预测结果
    function displayResults(data) {
        // 隐藏加载中
        hideLoading();
        
        // 显示结果区域
        resultsSection.classList.remove('d-none');
        
        // 更新统计数据
        document.getElementById('totalSequences').textContent = data.stats.total;
        document.getElementById('positiveCount').textContent = data.stats.positive;
        document.getElementById('positivePercentage').textContent = 
            Math.round((data.stats.positive / data.stats.total) * 100) + '%';
        
        // 更新结果表格
        const tableBody = document.getElementById('resultsTableBody');
        tableBody.innerHTML = '';
        
        data.results.forEach(result => {
            const row = document.createElement('tr');
            
            // 序列ID
            const idCell = document.createElement('td');
            idCell.textContent = result.id;
            row.appendChild(idCell);
            
            // 序列
            const seqCell = document.createElement('td');
            seqCell.classList.add('sequence-text');
            seqCell.textContent = result.sequence;
            seqCell.title = result.sequence; // 鼠标悬停显示完整序列
            row.appendChild(seqCell);
            
            // 预测概率
            const probCell = document.createElement('td');
            probCell.textContent = (result.probability * 100).toFixed(2) + '%';
            row.appendChild(probCell);
            
            // 预测结果
            const resultCell = document.createElement('td');
            const resultClass = result.prediction === 1 ? 'prediction-positive' : 'prediction-negative';
            resultCell.innerHTML = `<span class="${resultClass}">${result.label}</span>`;
            row.appendChild(resultCell);
            
            tableBody.appendChild(row);
        });
        
        // 绘制图表
        drawResultChart(data);
    }
    
    // 绘制结果图表
    function drawResultChart(data) {
        const ctx = document.getElementById('resultChart').getContext('2d');
        
        // 计算概率分布
        const probBins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        const probCounts = Array(probBins.length - 1).fill(0);
        
        data.results.forEach(result => {
            const prob = result.probability;
            for (let i = 0; i < probBins.length - 1; i++) {
                if (prob >= probBins[i] && prob < probBins[i + 1]) {
                    probCounts[i]++;
                    break;
                }
            }
            // 处理概率为1的情况
            if (prob === 1) {
                probCounts[probCounts.length - 1]++;
            }
        });
        
        // 生成标签
        const labels = [];
        for (let i = 0; i < probBins.length - 1; i++) {
            labels.push(`${(probBins[i] * 100).toFixed(0)}-${(probBins[i+1] * 100).toFixed(0)}%`);
        }
        
        // 销毁旧图表
        if (window.resultChart instanceof Chart) {
            window.resultChart.destroy();
        }
        
        // 创建新图表
        window.resultChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '序列数量',
                    data: probCounts,
                    backgroundColor: function(context) {
                        const index = context.dataIndex;
                        return index < 5 ? 'rgba(231, 76, 60, 0.7)' : 'rgba(46, 204, 113, 0.7)';
                    },
                    borderColor: function(context) {
                        const index = context.dataIndex;
                        return index < 5 ? 'rgba(192, 57, 43, 1)' : 'rgba(39, 174, 96, 1)';
                    },
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                return '预测概率: ' + tooltipItems[0].label;
                            },
                            label: function(context) {
                                return '序列数量: ' + context.raw;
                            }
                        }
                    },
                    title: {
                        display: true,
                        text: '预测概率分布',
                        font: {
                            size: 16
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '序列数量'
                        },
                        ticks: {
                            precision: 0
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '预测概率'
                        }
                    }
                }
            }
        });
    }
    
    // 导出CSV
    if (exportCSV) {
        exportCSV.addEventListener('click', function() {
            if (!predictionResults.length) {
                showError('没有可导出的结果');
                return;
            }
            
            fetch('/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    format: 'csv',
                    results: predictionResults
                })
            })
            .then(response => {
                if (response.ok) {
                    return response.blob();
                }
                throw new Error('导出失败');
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `gram_prediction_${new Date().toISOString().slice(0, 10)}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('导出CSV失败:', error);
                showError('导出CSV失败，请重试');
            });
        });
    }
    
    // 导出FASTA
    if (exportFASTA) {
        exportFASTA.addEventListener('click', function() {
            if (!predictionResults.length) {
                showError('没有可导出的结果');
                return;
            }
            
            // 检查是否有正例
            const positiveResults = predictionResults.filter(r => r.prediction === 1);
            if (!positiveResults.length) {
                showError('没有阳性结果可导出');
                return;
            }
            
            fetch('/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    format: 'fasta',
                    results: predictionResults
                })
            })
            .then(response => {
                if (response.ok) {
                    return response.blob();
                }
                throw new Error('导出失败');
            })
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = `gram_positive_${new Date().toISOString().slice(0, 10)}.fasta`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('导出FASTA失败:', error);
                showError('导出FASTA失败，请重试');
            });
        });
    }
    
    // 重置表单
    if (resetForms) {
        resetForms.addEventListener('click', function() {
            resetAll();
        });
    }
    
    // 显示加载中
    function showLoading() {
        loadingSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        errorSection.classList.add('d-none');
    }
    
    // 隐藏加载中
    function hideLoading() {
        loadingSection.classList.add('d-none');
    }
    
    // 显示错误
    function showError(message) {
        errorMessage.textContent = message;
        errorSection.classList.remove('d-none');
        loadingSection.classList.add('d-none');
    }
    
    // 重置所有
    function resetAll() {
        // 重置表单
        if (fileForm) fileForm.reset();
        if (textForm) textForm.reset();
        
        // 重置文件名显示
        const fileNameElement = dropZone ? dropZone.querySelector('.selected-file-name') : null;
        if (fileNameElement) {
            fileNameElement.textContent = '';
        }
        
        // 隐藏结果和错误
        resultsSection.classList.add('d-none');
        errorSection.classList.add('d-none');
        loadingSection.classList.add('d-none');
        
        // 清空结果数据
        predictionResults = [];
    }
});
