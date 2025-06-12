document.addEventListener('DOMContentLoaded', () => {
    const appContainer = document.getElementById('app-container');
    const navLinks = document.querySelectorAll('[data-nav-link]');

    const routes = {
        '/': '/content/predict',
        '/predict': '/content/predict',
        '/generate': '/content/generate',
        '/database': '/content/database',
        '/about': '/content/about'
    };

    const loadContent = async (path) => {
        const contentPath = routes[path] || '/content/predict'; // Default to predict page
        try {
            appContainer.style.opacity = '0'; // Start fade out
            const response = await fetch(contentPath);
            if (!response.ok) throw new Error(`Failed to fetch content for ${path}`);
            
            const contentHtml = await response.text();
            
            setTimeout(() => {
                appContainer.innerHTML = contentHtml;
                appContainer.style.opacity = '1'; // Fade in
                // After loading content, initialize its specific JS logic
                initializePageScripts(path);
            }, 300); // Match transition duration

        } catch (error) {
            console.error('Error loading content:', error);
            appContainer.innerHTML = `<div class="alert alert-danger">Failed to load page content. Please try again.</div>`;
            appContainer.style.opacity = '1';
        }
    };

    const initializePageScripts = (path) => {
        if (path === '/' || path === '/predict') {
            initPredictPage();
        } else if (path === '/generate') {
            initGeneratePage();
        } else if (path === '/database') {
            initDatabasePage();
        }
        // Initialize common elements like toasts or particles if needed
        initParticles();
    };

    const handleNavClick = (e) => {
        e.preventDefault();
        const path = e.currentTarget.getAttribute('href');
        if (path !== window.location.pathname) {
            window.history.pushState({}, '', path);
            loadContent(path);
            updateActiveNavLink(path);
        }
    };

    const updateActiveNavLink = (path) => {
        navLinks.forEach(link => {
            if (link.getAttribute('href') === path) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    };

    navLinks.forEach(link => {
        link.addEventListener('click', handleNavClick);
    });

    window.addEventListener('popstate', () => {
        const path = window.location.pathname;
        loadContent(path);
        updateActiveNavLink(path);
    });

    // Initial page load
    const initialPath = window.location.pathname;
    loadContent(initialPath);
    updateActiveNavLink(initialPath);

    // Add transition style to app container
    appContainer.style.transition = 'opacity 0.3s ease-in-out';
});

    const initParticles = () => {
        if (document.getElementById('tsparticles-background') && typeof tsParticles !== 'undefined') {
            tsParticles.load("tsparticles-background", {
                fpsLimit: 60, background: { color: "transparent" },
                particles: {
                    number: { value: 80, density: { enable: true, area: 800 } },
                    color: { value: ["#ffffff", "#b3e5fc", "#81d4fa"] },
                    shape: { type: "circle" },
                    opacity: { value: { min: 0.1, max: 0.5 } },
                    size: { value: { min: 1, max: 3 } },
                    links: { enable: true, distance: 150, color: "#81d4fa", opacity: 0.2, width: 1 },
                    move: { enable: true, speed: 1, direction: "none", random: true, straight: false, outModes: { default: "out" } }
                },
                interactivity: {
                    events: { onHover: { enable: true, mode: "repulse" }, onClick: { enable: true, mode: "push" } },
                    modes: { repulse: { distance: 100, duration: 0.4 }, push: { quantity: 4 } }
                },
                detectRetina: true
            });
        }
    };

    const showToast = (message, type = 'info', title = '') => {
        const toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) return;
        const toastId = 'toast-' + Math.random().toString(36).substring(2, 9);
        const iconHtml = {
            success: '<i class="bi bi-check-circle-fill text-success me-2"></i>',
            danger: '<i class="bi bi-x-octagon-fill text-danger me-2"></i>',
            warning: '<i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>',
            info: '<i class="bi bi-info-circle-fill text-info me-2"></i>'
        }[type] || '<i class="bi bi-info-circle-fill text-info me-2"></i>';
        const toastHeaderClass = `text-${type}`;
        const toastTitle = title || type.charAt(0).toUpperCase() + type.slice(1);

        const toastHtml = `
            <div class="toast" role="alert" aria-live="assertive" aria-atomic="true" id="${toastId}" data-bs-delay="5000">
                <div class="toast-header">
                    ${iconHtml}
                    <strong class="me-auto ${toastHeaderClass}">${toastTitle}</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">${message}</div>
            </div>`;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);
        const bsToast = new bootstrap.Toast(toastElement);
        bsToast.show();
        toastElement.addEventListener('hidden.bs.toast', () => toastElement.remove());
    };

    // --- PREDICTION PAGE LOGIC ---
    const initPredictPage = () => {
        const submissionSection = document.getElementById('submission-section');
        const resultsSection = document.getElementById('resultsSection');
        const loadingSection = document.getElementById('loadingSection');
        const errorSection = document.getElementById('errorSection');
        const errorMessageEl = document.getElementById('errorMessage');
        
        const fileForm = document.getElementById('fileForm');
        const textForm = document.getElementById('textForm');
        const fastaFileEl = document.getElementById('fastaFile');
        const fastaTextEl = document.getElementById('fastaText');
        const dropZone = document.getElementById('dropZone');
        const selectedFileNameEl = dropZone?.querySelector('.selected-file-name');
        const loadExampleBtn = document.getElementById('loadExample');
        
        const newPredictionBtn = document.getElementById('newPredictionBtn');
        const resetViewBtn = document.getElementById('resetViewBtn');

        let resultsTable = null;
        let resultChart = null;

        const switchView = (view) => {
            submissionSection.classList.toggle('d-none', view === 'results');
            resultsSection.classList.toggle('d-none', view !== 'results');
            loadingSection.classList.add('d-none');
            errorSection.classList.add('d-none');
        };

        const handlePredictionSubmit = async (formData) => {
            loadingSection.classList.remove('d-none');
            submissionSection.classList.add('d-none');
            errorSection.classList.add('d-none');

            try {
                const response = await fetch('/api/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if (!response.ok || !data.success) {
                    throw new Error(data.error || 'Prediction failed. Please check your input.');
                }
                renderResults(data);
                switchView('results');
            } catch (error) {
                errorMessageEl.textContent = error.message;
                errorSection.classList.remove('d-none');
                loadingSection.classList.add('d-none');
            }
        };

        let currentResultsData = [];
        
        const renderResults = (data) => {
            // Store current results for export functionality
            currentResultsData = data.results;
            
            // Stats
            const statsContainer = document.getElementById('stats-container');
            statsContainer.innerHTML = `
                <div class="col-md-3 col-6 mb-3 mb-md-0"><h3>${data.stats.total}</h3><p>Total</p></div>
                <div class="col-md-3 col-6 mb-3 mb-md-0"><h3>${data.stats.positive}</h3><p>Positive</p></div>
                <div class="col-md-3 col-6"><h3>${data.stats.negative}</h3><p>Negative</p></div>
                <div class="col-md-3 col-6"><h3>${data.stats.positive_percentage}%</h3><p>Positive %</p></div>
            `;

            // Chart
            const chartEl = document.getElementById('resultChart');
            if (chartEl && typeof Chart !== 'undefined') {
                if (resultChart) resultChart.destroy();
                resultChart = new Chart(chartEl.getContext('2d'), {
                    type: 'doughnut',
                    data: {
                        labels: ['Positive', 'Negative', 'Failed'],
                        datasets: [{
                            data: [data.stats.positive, data.stats.negative, data.stats.failed],
                            backgroundColor: ['rgba(40, 167, 69, 0.8)', 'rgba(108, 117, 125, 0.8)', 'rgba(220, 53, 69, 0.7)'],
                            borderColor: ['#fff'],
                            borderWidth: 2,
                        }]
                    },
                    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
                });
            }

            renderFeatureBoxPlot(data.results);
            renderScatterPlot(data.results);
            renderAACompositionChart(data.results);
            renderRadarChart(data.results);
            renderProbabilityHistogram(data.results);
            renderSlidingWindowChart(data.sliding_window_data);
            renderDimensionalityReductionCharts(data.dimensionality_reduction_data);

            // DataTable
            if ($.fn.dataTable.isDataTable('#resultsDataTable')) {
                resultsTable.destroy();
            }
            resultsTable = $('#resultsDataTable').DataTable({
                data: data.results,
                columns: [
                    { title: "ID", data: "id" },
                    { title: "Sequence", data: "sequence", render: (d) => `<div class="sequence-text">${d.substring(0, 20)}...</div>` },
                    { title: "Probability", data: "probability", render: (d) => d.toFixed(4) },
                    { title: "Prediction", data: "label", render: (d, type, row) => `<span class="badge ${row.prediction === 1 ? 'bg-success' : 'bg-secondary'}">${d}</span>` }
                ],
                responsive: true,
                "lengthMenu": [[10, 25, 50, -1], [10, 25, 50, "All"]],
                "pageLength": 10
            });
        };

        // Event Listeners
        fileForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            if (fastaFileEl.files.length === 0) return showToast('Please select a file.', 'warning');
            const formData = new FormData();
            formData.append('fasta_file', fastaFileEl.files[0]);
            handlePredictionSubmit(formData);
        });

        textForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            if (fastaTextEl.value.trim() === '') return showToast('Please enter sequence data.', 'warning');
            const formData = new FormData();
            formData.append('fasta_text', fastaTextEl.value.trim());
            handlePredictionSubmit(formData);
        });

        loadExampleBtn?.addEventListener('click', async () => {
            try {
                const response = await fetch('/example');
                const data = await response.json();
                if (data.fasta) {
                    fastaTextEl.value = data.fasta;
                    new bootstrap.Tab(document.getElementById('text-tab')).show();
                }
            } catch (error) { showToast('Failed to load example data.', 'danger'); }
        });

        fastaFileEl?.addEventListener('change', () => {
            if (fastaFileEl.files.length > 0) selectedFileNameEl.textContent = fastaFileEl.files[0].name;
        });

        dropZone?.addEventListener('dragover', (e) => e.preventDefault());
        dropZone?.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) {
                fastaFileEl.files = e.dataTransfer.files;
                selectedFileNameEl.textContent = e.dataTransfer.files[0].name;
            }
        });

        newPredictionBtn?.addEventListener('click', () => switchView('submit'));
        resetViewBtn?.addEventListener('click', () => switchView('submit'));

        // Export functionality
        const exportCSVBtn = document.getElementById('exportCSV');
        const exportFASTABtn = document.getElementById('exportFASTA');

        exportCSVBtn?.addEventListener('click', async () => {
            if (currentResultsData.length === 0) {
                showToast('No data to export.', 'warning');
                return;
            }

            try {
                const response = await fetch('/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ format: 'csv', results: currentResultsData })
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `gram_prediction_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.csv`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    showToast('CSV exported successfully!', 'success');
                } else {
                    throw new Error('Export failed');
                }
            } catch (error) {
                showToast('Export failed: ' + error.message, 'danger');
            }
        });

        exportFASTABtn?.addEventListener('click', async () => {
            if (currentResultsData.length === 0) {
                showToast('No data to export.', 'warning');
                return;
            }

            try {
                const response = await fetch('/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ format: 'fasta', results: currentResultsData })
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `gram_positive_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.fasta`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    showToast('FASTA exported successfully!', 'success');
                } else {
                    throw new Error('Export failed');
                }
            } catch (error) {
                showToast('Export failed: ' + error.message, 'danger');
            }
        });

        const renderFeatureBoxPlot = (results) => {
            const canvas = document.getElementById('featureBoxPlot');
            if (!canvas || !results) return;

            try {
                // 分离positive和negative数据
                const positiveData = results.filter(r => r.prediction === 1);
                const negativeData = results.filter(r => r.prediction === 0);
                
                if (positiveData.length === 0 && negativeData.length === 0) {
                    console.warn('No valid data for feature comparison');
                    return;
                }

                // 主要特征和中文标签
                const mainFeatures = ['Charge', 'Hydrophobicity', 'Hydrophobic_Moment', 'Instability_Index', 'Isoelectric_Point', 'Aliphatic_Index'];
                const featureLabels = {
                    'Charge': 'Charge',
                    'Hydrophobicity': 'Hydrophobicity',
                    'Hydrophobic_Moment': 'Hydrophobic Moment',
                    'Instability_Index': 'Instability Index',
                    'Isoelectric_Point': 'Isoelectric Point',
                    'Aliphatic_Index': 'Aliphatic Index'
                };

                // 计算每个特征的平均值和标准差
                const calculateStats = (data, feature) => {
                    const values = data
                        .map(item => item.features && item.features[feature])
                        .filter(val => val !== undefined && val !== null && !isNaN(val) && isFinite(val));
                    
                    if (values.length === 0) return { mean: 0, std: 0, values: [] };
                    
                    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
                    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
                    const std = Math.sqrt(variance);
                    
                    return { mean, std, values };
                };

                // 计算所有数据的全局统计，用于标准化
                const getAllFeatureValues = (feature) => {
                    const allValues = [...positiveData, ...negativeData]
                        .map(item => item.features && item.features[feature])
                        .filter(val => val !== undefined && val !== null && !isNaN(val) && isFinite(val));
                    return allValues;
                };

                const labels = mainFeatures.map(f => featureLabels[f]);
                const positiveStats = mainFeatures.map(f => calculateStats(positiveData, f));
                const negativeStats = mainFeatures.map(f => calculateStats(negativeData, f));
                
                // 对每个特征进行Z-score标准化
                const normalizedPositiveMeans = [];
                const normalizedNegativeMeans = [];
                const normalizedPositiveStds = [];
                const normalizedNegativeStds = [];
                const effectSizes = []; // Cohen's d 效应量

                mainFeatures.forEach((feature, i) => {
                    const allValues = getAllFeatureValues(feature);
                    
                    if (allValues.length === 0) {
                        normalizedPositiveMeans.push(0);
                        normalizedNegativeMeans.push(0);
                        normalizedPositiveStds.push(0);
                        normalizedNegativeStds.push(0);
                        effectSizes.push(0);
                        return;
                    }
                    
                    // 计算全局均值和标准差用于标准化
                    const globalMean = allValues.reduce((sum, val) => sum + val, 0) / allValues.length;
                    const globalVariance = allValues.reduce((sum, val) => sum + Math.pow(val - globalMean, 2), 0) / allValues.length;
                    const globalStd = Math.sqrt(globalVariance);
                    
                    // 避免除零错误
                    if (globalStd === 0) {
                        normalizedPositiveMeans.push(0);
                        normalizedNegativeMeans.push(0);
                        normalizedPositiveStds.push(0);
                        normalizedNegativeStds.push(0);
                        effectSizes.push(0);
                        return;
                    }
                    
                    // Z-score标准化
                    const posNormalizedMean = (positiveStats[i].mean - globalMean) / globalStd;
                    const negNormalizedMean = (negativeStats[i].mean - globalMean) / globalStd;
                    const posNormalizedStd = positiveStats[i].std / globalStd;
                    const negNormalizedStd = negativeStats[i].std / globalStd;
                    
                    normalizedPositiveMeans.push(posNormalizedMean);
                    normalizedNegativeMeans.push(negNormalizedMean);
                    normalizedPositiveStds.push(posNormalizedStd);
                    normalizedNegativeStds.push(negNormalizedStd);
                    
                    // 计算Cohen's d效应量 (衡量两组差异的实际意义)
                    const pooledStd = Math.sqrt((Math.pow(positiveStats[i].std, 2) + Math.pow(negativeStats[i].std, 2)) / 2);
                    const cohensD = pooledStd > 0 ? (positiveStats[i].mean - negativeStats[i].mean) / pooledStd : 0;
                    effectSizes.push(cohensD);
                });

                // 使用标准化后的数据
                const positiveMeans = normalizedPositiveMeans;
                const negativeMeans = normalizedNegativeMeans;
                const positiveStds = normalizedPositiveStds;
                const negativeStds = normalizedNegativeStds;

                // 销毁现有图表
                let existingChart = Chart.getChart(canvas);
                if (existingChart) {
                    existingChart.destroy();
                }

                // 创建渐变背景
                const ctx = canvas.getContext('2d');
                const positiveGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
                positiveGradient.addColorStop(0, 'rgba(40, 167, 69, 0.8)');
                positiveGradient.addColorStop(1, 'rgba(40, 167, 69, 0.4)');
                
                const negativeGradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
                negativeGradient.addColorStop(0, 'rgba(108, 117, 125, 0.8)');
                negativeGradient.addColorStop(1, 'rgba(108, 117, 125, 0.4)');

                // 创建分组柱状图
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: `Anti-Gram-negative (n=${positiveData.length})`,
                            data: positiveMeans,
                            backgroundColor: positiveGradient,
                            borderColor: 'rgba(40, 167, 69, 1)',
                            borderWidth: 2,
                            errorBars: {
                                '+': positiveStds,
                                '-': positiveStds
                            }
                        }, {
                            label: `Non-Anti-Gram-negative (n=${negativeData.length})`,
                            data: negativeMeans,
                            backgroundColor: negativeGradient,
                            borderColor: 'rgba(108, 117, 125, 1)',
                            borderWidth: 2,
                            errorBars: {
                                '+': negativeStds,
                                '-': negativeStds
                            }
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: {
                            duration: 2000,
                            easing: 'easeInOutQuart'
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: 'Feature Comparison Analysis (Z-score Normalized)',
                                color: 'white',
                                font: {
                                    size: 18,
                                    weight: 'bold'
                                },
                                padding: {
                                    top: 15,
                                    bottom: 25
                                }
                            },
                            legend: {
                                labels: {
                                    color: 'white',
                                    font: {
                                        size: 13,
                                        weight: 'bold'
                                    },
                                    padding: 20,
                                    usePointStyle: true,
                                    pointStyle: 'rect'
                                }
                            },
                            tooltip: {
                                backgroundColor: 'rgba(0, 0, 0, 0.9)',
                                titleColor: 'white',
                                bodyColor: 'white',
                                borderColor: 'rgba(255, 255, 255, 0.3)',
                                borderWidth: 1,
                                cornerRadius: 10,
                                displayColors: true,
                                callbacks: {
                                    label: function(context) {
                                        const datasetIndex = context.datasetIndex;
                                        const featureIndex = context.dataIndex;
                                        const normalizedMean = context.parsed.y;
                                        const normalizedStd = datasetIndex === 0 ? positiveStds[featureIndex] : negativeStds[featureIndex];
                                        const cohensD = effectSizes[featureIndex];
                                        const originalMean = datasetIndex === 0 ? positiveStats[featureIndex].mean : negativeStats[featureIndex].mean;
                                        
                                        // 效应量解释
                                        let effectText = '';
                                        const absEffect = Math.abs(cohensD);
                                        if (absEffect < 0.2) effectText = '(very small difference)';
                                        else if (absEffect < 0.5) effectText = '(small difference)';
                                        else if (absEffect < 0.8) effectText = '(medium difference)';
                                        else effectText = '(large difference)';
                                        
                                        return [
                                            `${context.dataset.label}`,
                                            `Normalized value: ${normalizedMean.toFixed(3)}`,
                                            `Original mean: ${originalMean.toFixed(3)}`,
                                            `Effect size: ${cohensD.toFixed(3)} ${effectText}`
                                        ];
                                    },
                                    title: function(tooltipItems) {
                                        return `Feature: ${tooltipItems[0].label}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            y: {
                                title: {
                                    display: true,
                                    text: 'Z-score Normalized Value',
                                    color: 'white',
                                    font: {
                                        size: 14,
                                        weight: 'bold'
                                    }
                                },
                                ticks: {
                                    color: 'white',
                                    font: {
                                        size: 12
                                    }
                                },
                                grid: {
                                    color: 'rgba(255,255,255,0.15)',
                                    lineWidth: 1
                                },
                                border: {
                                    color: 'rgba(255,255,255,0.3)'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Feature Type',
                                    color: 'white',
                                    font: {
                                        size: 14,
                                        weight: 'bold'
                                    }
                                },
                                ticks: {
                                    color: 'white',
                                    maxRotation: 45,
                                    minRotation: 45,
                                    font: {
                                        size: 12,
                                        weight: 'bold'
                                    }
                                },
                                grid: {
                                    color: 'rgba(255,255,255,0.1)',
                                    lineWidth: 1
                                },
                                border: {
                                    color: 'rgba(255,255,255,0.3)'
                                }
                            }
                        },
                        interaction: {
                            mode: 'index',
                            intersect: false
                        }
                    }
                });

                // 在图表下方添加效应量分析
                setTimeout(() => {
                    const significantFeatures = effectSizes
                        .map((effect, i) => ({ 
                            feature: labels[i], 
                            cohensD: effect,
                            positiveMean: normalizedPositiveMeans[i],
                            negativeMean: normalizedNegativeMeans[i],
                            originalPoseMean: positiveStats[i].mean,
                            originalNegMean: negativeStats[i].mean
                        }))
                        .filter(item => Math.abs(item.cohensD) > 0.2) // 只显示有意义的差异
                        .sort((a, b) => Math.abs(b.cohensD) - Math.abs(a.cohensD));
                    
                    if (significantFeatures.length > 0) {
                        console.log('Features with significant effect size (Cohen\'s d > 0.2):', significantFeatures);
                        console.log('Effect size interpretation: <0.2=very small, 0.2-0.5=small, 0.5-0.8=medium, >0.8=large');
                    }
                }, 1000);

            } catch (error) {
                console.error('Feature comparison chart rendering error:', error);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('Feature comparison chart rendering failed', canvas.width / 2, canvas.height / 2);
            }
        };

        // 替代方案：当箱线图插件不可用时使用
        const renderFeatureBarChartAlternative = (boxPlotData, canvas) => {
            try {
                const mainFeatures = ['Length', 'Charge', 'Hydrophobicity', 'Hydrophobic_Moment', 'Instability_Index', 'Isoelectric_Point', 'Aliphatic_Index'];
                const featureLabels = {
                    'Length': 'Sequence Length',
                    'Charge': 'Charge', 
                    'Hydrophobicity': 'Hydrophobicity',
                    'Hydrophobic_Moment': 'Hydrophobic Moment',
                    'Instability_Index': 'Instability Index',
                    'Isoelectric_Point': 'Isoelectric Point',
                    'Aliphatic_Index': 'Aliphatic Index'
                };
                
                const filteredData = boxPlotData.filter(item => mainFeatures.includes(item.feature));
                
                if (filteredData.length === 0) return;

                const labels = filteredData.map(item => featureLabels[item.feature] || item.feature);
                const medianData = filteredData.map(item => item.stats.median);
                const q1Data = filteredData.map(item => item.stats.q1);
                const q3Data = filteredData.map(item => item.stats.q3);

                let existingChart = Chart.getChart(canvas);
                if (existingChart) existingChart.destroy();

                new Chart(canvas.getContext('2d'), {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: 'Median',
                            data: medianData,
                            backgroundColor: 'rgba(54, 162, 235, 0.7)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }, {
                            label: 'Q1-Q3 Range',
                            data: q3Data.map((q3, i) => q3 - q1Data[i]),
                            backgroundColor: 'rgba(255, 206, 86, 0.5)',
                            borderColor: 'rgba(255, 206, 86, 1)',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Feature Value Distribution (Alternative View)',
                                color: 'white',
                                font: { size: 16 }
                            },
                            legend: {
                                labels: { color: 'white' }
                            }
                        },
                        scales: {
                            y: {
                                title: { display: true, text: 'Feature Value', color: 'white' },
                                ticks: { color: 'white' },
                                grid: { color: 'rgba(255,255,255,0.1)' }
                            },
                            x: {
                                ticks: { color: 'white', maxRotation: 45 },
                                grid: { color: 'rgba(255,255,255,0.1)' }
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Alternative chart rendering failed:', error);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('Feature distribution chart rendering failed', canvas.width / 2, canvas.height / 2);
            }
        };

        const renderFeatureBarChart = (results) => {
            const canvas = document.getElementById('featureBarChart');
            if (!canvas) return;

            const positiveData = results.filter(r => r.prediction === 1);
            const negativeData = results.filter(r => r.prediction === 0);
            const features = ['Charge', 'Hydrophobicity', 'Hydrophobic_Moment'];

            const calcAverage = (data, key) => {
                if (!data || data.length === 0) return 0;
                const validValues = data
                    .map(item => item.features && item.features[key])
                    .filter(val => val !== undefined && val !== null && !isNaN(val) && isFinite(val));
                
                if (validValues.length === 0) return 0;
                const sum = validValues.reduce((acc, curr) => acc + curr, 0);
                return sum / validValues.length;
            };

            const positiveAvgs = features.map(f => calcAverage(positiveData, f));
            const negativeAvgs = features.map(f => calcAverage(negativeData, f));

            let existingChart = Chart.getChart(canvas);
            if (existingChart) {
                existingChart.destroy();
            }

            new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: features,
                    datasets: [
                        {
                            label: 'Anti-Gram-negative',
                            data: positiveAvgs,
                            backgroundColor: 'rgba(40, 167, 69, 0.7)',
                            borderColor: 'rgba(40, 167, 69, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'Non-Anti-Gram-negative',
                            data: negativeAvgs,
                            backgroundColor: 'rgba(108, 117, 125, 0.7)',
                            borderColor: 'rgba(108, 117, 125, 1)',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: { color: 'white' }
                        }
                    },
                    scales: {
                        y: {
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        },
                        x: {
                            ticks: { color: 'white' },
                            grid: { color: 'rgba(255,255,255,0.1)' }
                        }
                    }
                }
            });
        };

        const renderScatterPlot = (results) => {
            const canvas = document.getElementById('featureScatterPlot');
            if (!canvas) return;

            const positiveData = results
                .filter(r => r.prediction === 1)
                .map(r => ({
                    x: r.features.Hydrophobicity,
                    y: r.features.Charge
                }))
                .filter(point =>
                    point.x !== undefined && point.y !== undefined &&
                    !isNaN(point.x) && !isNaN(point.y) &&
                    isFinite(point.x) && isFinite(point.y)
                );
            
            const negativeData = results
                .filter(r => r.prediction === 0)
                .map(r => ({
                    x: r.features.Hydrophobicity,
                    y: r.features.Charge
                }))
                .filter(point =>
                    point.x !== undefined && point.y !== undefined &&
                    !isNaN(point.x) && !isNaN(point.y) &&
                    isFinite(point.x) && isFinite(point.y)
                );

            let existingChart = Chart.getChart(canvas);
            if (existingChart) existingChart.destroy();

            new Chart(canvas.getContext('2d'), {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Anti-Gram-negative',
                        data: positiveData,
                        backgroundColor: 'rgba(40, 167, 69, 0.7)'
                    }, {
                        label: 'Non-Anti-Gram-negative',
                        data: negativeData,
                        backgroundColor: 'rgba(108, 117, 125, 0.7)'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: 'white' } },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) { label += ': '; }
                                    label += `(Hydrophobicity: ${context.parsed.x.toFixed(2)}, Charge: ${context.parsed.y.toFixed(2)})`;
                                    return label;
                                }
                            }
                        }
                    },
                    scales: {
                        y: { title: { display: true, text: 'Charge', color: 'white' }, ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        x: { title: { display: true, text: 'Hydrophobicity', color: 'white' }, ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
        };

        const renderAACompositionChart = (results) => {
            const canvas = document.getElementById('aaCompositionChart');
            if (!canvas) return;

            const aminoAcids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];
            const positiveData = results.filter(r => r.prediction === 1);
            const negativeData = results.filter(r => r.prediction === 0);

            const calcAAFrequencies = (data) => {
                if (!data || data.length === 0) return aminoAcids.map(() => 0);
                const totalLength = data.reduce((sum, r) => sum + (r.sequence ? r.sequence.length : 0), 0);
                if (totalLength === 0) return aminoAcids.map(() => 0);

                const aaCounts = aminoAcids.map(aa =>
                    data.reduce((sum, r) => sum + ((r.features[`AA_${aa}`] || 0) * (r.sequence ? r.sequence.length : 0)), 0)
                );
                return aaCounts.map(count => (count / totalLength) * 100);
            };

            const positiveFreqs = calcAAFrequencies(positiveData);
            const negativeFreqs = calcAAFrequencies(negativeData);

            let existingChart = Chart.getChart(canvas);
            if (existingChart) existingChart.destroy();

            new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: aminoAcids,
                    datasets: [{
                        label: 'Anti-Gram-negative Freq (%)',
                        data: positiveFreqs,
                        backgroundColor: 'rgba(40, 167, 69, 0.7)',
                    }, {
                        label: 'Non-Anti-Gram-negative Freq (%)',
                        data: negativeFreqs,
                        backgroundColor: 'rgba(108, 117, 125, 0.7)',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: 'white' } } },
                    scales: {
                        y: { title: { display: true, text: 'Frequency (%)', color: 'white' }, ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        x: { ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
        };

        const renderRadarChart = (results) => {
            const canvas = document.getElementById('featureRadarChart');
            if (!canvas) return;

            try {
                const features = ['Charge', 'Hydrophobicity', 'Hydrophobic_Moment', 'Instability_Index', 'Aliphatic_Index'];
                const positiveData = results.filter(r => r.prediction === 1);
                const negativeData = results.filter(r => r.prediction === 0);

                // 安全的平均值计算函数，增加数据验证
                const getAverages = (data) => {
                    if (!data || data.length === 0) return features.map(() => 0);
                    return features.map(key => {
                        const validValues = data
                            .map(item => item.features && item.features[key])
                            .filter(val => val !== undefined && val !== null && !isNaN(val) && isFinite(val));
                        
                        if (validValues.length === 0) return 0;
                        const sum = validValues.reduce((acc, curr) => acc + curr, 0);
                        return sum / validValues.length;
                    });
                };

                const posAvgs = getAverages(positiveData);
                const negAvgs = getAverages(negativeData);

                // 验证计算出的平均值
                const isValidArray = (arr) => arr.every(val =>
                    val !== undefined && val !== null && !isNaN(val) && isFinite(val)
                );

                if (!isValidArray(posAvgs) || !isValidArray(negAvgs)) {
                    console.warn('Radar chart: Invalid data values detected, skipping rendering');
                    return;
                }

                // 使用相对比较标准化，突出positive和negative之间的差异
                const normalize = (posValues, negValues, allResultsData) => {
                    const posNormalized = [];
                    const negNormalized = [];
                    
                    for (let i = 0; i < features.length; i++) {
                        const featureName = features[i];
                        const posVal = posValues[i];
                        const negVal = negValues[i];
                        
                        // 确保值是有效的数字
                        if (!isFinite(posVal) || !isFinite(negVal)) {
                            posNormalized.push(0.5);
                            negNormalized.push(0.5);
                            continue;
                        }
                        
                        // 计算两组之间的差异比例
                        const maxVal = Math.max(Math.abs(posVal), Math.abs(negVal));
                        const diff = Math.abs(posVal - negVal);
                        
                        // 如果两个值都接近0或者差异很小，使用中心值
                        if (maxVal < 1e-10 || diff < 1e-10) {
                            posNormalized.push(0.5);
                            negNormalized.push(0.5);
                            continue;
                        }
                        
                        // 使用相对差异进行标准化，放大差异显示
                        const center = 0.5;
                        const scale = 0.3; // 控制偏离中心的最大幅度
                        
                        // 计算相对位置：哪个值更大就向外偏移
                        if (posVal > negVal) {
                            const ratio = Math.min(diff / Math.abs(posVal), 1.0); // 限制最大比例
                            posNormalized.push(center + scale * ratio);
                            negNormalized.push(center - scale * ratio);
                        } else {
                            const ratio = Math.min(diff / Math.abs(negVal), 1.0);
                            posNormalized.push(center - scale * ratio);
                            negNormalized.push(center + scale * ratio);
                        }
                    }
                    
                    return { pos: posNormalized, neg: negNormalized };
                };

                const normalizedData = normalize(posAvgs, negAvgs, results);
                const normalizedPosData = normalizedData.pos;
                const normalizedNegData = normalizedData.neg;

                // 最终验证标准化数据
                if (!isValidArray(normalizedPosData) || !isValidArray(normalizedNegData)) {
                    console.warn('Radar chart: Normalized data invalid, skipping rendering');
                    return;
                }

                let existingChart = Chart.getChart(canvas);
                if (existingChart) existingChart.destroy();

                new Chart(canvas.getContext('2d'), {
                    type: 'radar',
                    data: {
                        labels: features,
                        datasets: [{
                            label: `Anti-Gram-negative (n=${positiveData.length})`,
                            data: normalizedPosData,
                            backgroundColor: 'rgba(40, 167, 69, 0.3)',
                            borderColor: 'rgba(40, 167, 69, 1)',
                            pointBackgroundColor: 'rgba(40, 167, 69, 1)',
                            borderWidth: 2,
                            pointRadius: 5,
                            pointHoverRadius: 7
                        }, {
                            label: `Non-Anti-Gram-negative (n=${negativeData.length})`,
                            data: normalizedNegData,
                            backgroundColor: 'rgba(108, 117, 125, 0.3)',
                            borderColor: 'rgba(108, 117, 125, 1)',
                            pointBackgroundColor: 'rgba(108, 117, 125, 1)',
                            borderWidth: 2,
                            pointRadius: 5,
                            pointHoverRadius: 7
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                labels: { color: 'white' }
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const featureIndex = context.dataIndex;
                                        const isPositive = context.datasetIndex === 0;
                                        const originalValue = isPositive ? posAvgs[featureIndex] : negAvgs[featureIndex];
                                        const normalizedValue = context.parsed.r;
                                        
                                        return `${context.dataset.label}: ${originalValue.toFixed(3)} (normalized: ${normalizedValue.toFixed(3)})`;
                                    },
                                    title: function(tooltipItems) {
                                        return `Feature: ${tooltipItems[0].label}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            r: {
                                angleLines: { color: 'rgba(255,255,255,0.2)' },
                                grid: { color: 'rgba(255,255,255,0.2)' },
                                pointLabels: {
                                    font: { size: 12 },
                                    color: 'white'
                                },
                                ticks: {
                                    display: false,
                                    beginAtZero: true,
                                    min: 0,
                                    max: 1
                                },
                                min: 0,
                                max: 1
                            }
                        }
                    }
                });

            } catch (error) {
                console.error('Radar chart rendering error:', error);
                // 如果出现错误，在canvas上显示错误信息
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '16px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('Radar chart rendering failed', canvas.width / 2, canvas.height / 2);
            }
        };

        const renderProbabilityHistogram = (results) => {
            const canvas = document.getElementById('probabilityHistogram');
            if (!canvas) return;

            const probabilities = results.map(r => r.probability);
            const bins = Array(10).fill(0);
            probabilities.forEach(p => {
                if (p >= 0) {
                    const binIndex = Math.min(Math.floor(p * 10), 9);
                    bins[binIndex]++;
                }
            });
            const labels = bins.map((_, i) => `${(i * 0.1).toFixed(1)}-${((i + 1) * 0.1).toFixed(1)}`);

            let existingChart = Chart.getChart(canvas);
            if (existingChart) existingChart.destroy();

            new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Number of Sequences',
                        data: bins,
                        backgroundColor: 'rgba(0, 123, 255, 0.7)',
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { title: { display: true, text: 'Count', color: 'white' }, ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } },
                        x: { title: { display: true, text: 'Prediction Probability', color: 'white' }, ticks: { color: 'white' }, grid: { color: 'rgba(255,255,255,0.1)' } }
                    }
                }
            });
        };

        // 存储滑动窗口数据以便下拉菜单使用
        let currentSlidingWindowData = null;

        const renderSlidingWindowChart = (slidingWindowData) => {
            const canvas = document.getElementById('slidingWindowChart');
            const sequenceSelector = document.getElementById('sequenceSelector');
            
            if (!canvas || !slidingWindowData) return;

            // 存储数据
            currentSlidingWindowData = slidingWindowData;

            try {
                // 初始化下拉菜单
                initializeSequenceSelector(slidingWindowData);

                // 如果没有选中的序列，显示默认的第一个正样本和负样本
                renderSelectedSequenceChart();

            } catch (error) {
                console.error('Sliding window chart initialization error:', error);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('Sliding window chart initialization failed', canvas.width / 2, canvas.height / 2);
            }
        };

        const initializeSequenceSelector = (slidingWindowData) => {
            const sequenceSelector = document.getElementById('sequenceSelector');
            if (!sequenceSelector) return;

            // 清空现有选项
            sequenceSelector.innerHTML = '<option value="">Please select a sequence...</option>';

            // 添加正样本选项
            if (slidingWindowData.positive_samples && slidingWindowData.positive_samples.length > 0) {
                const positiveGroup = document.createElement('optgroup');
                positiveGroup.label = 'Positive Samples (Predicted as Anti-Gram-negative)';
                slidingWindowData.positive_samples.forEach((sample, index) => {
                    const option = document.createElement('option');
                    option.value = `positive_${index}`;
                    option.textContent = `${sample.id}: ${sample.sequence.substring(0, 15)}... (Anti-Gram-negative)`;
                    positiveGroup.appendChild(option);
                });
                sequenceSelector.appendChild(positiveGroup);
            }

            // 添加负样本选项
            if (slidingWindowData.negative_samples && slidingWindowData.negative_samples.length > 0) {
                const negativeGroup = document.createElement('optgroup');
                negativeGroup.label = 'Negative Samples (Predicted as Non-Anti-Gram-negative)';
                slidingWindowData.negative_samples.forEach((sample, index) => {
                    const option = document.createElement('option');
                    option.value = `negative_${index}`;
                    option.textContent = `${sample.id}: ${sample.sequence.substring(0, 15)}... (Non-Anti-Gram-negative)`;
                    negativeGroup.appendChild(option);
                });
                sequenceSelector.appendChild(negativeGroup);
            }

            // 默认选择第一个正样本（如果存在）
            if (slidingWindowData.positive_samples && slidingWindowData.positive_samples.length > 0) {
                sequenceSelector.value = 'positive_0';
            } else if (slidingWindowData.negative_samples && slidingWindowData.negative_samples.length > 0) {
                sequenceSelector.value = 'negative_0';
            }

            // 添加事件监听器
            sequenceSelector.removeEventListener('change', handleSequenceSelection);
            sequenceSelector.addEventListener('change', handleSequenceSelection);
        };

        const handleSequenceSelection = () => {
            renderSelectedSequenceChart();
        };

        const renderSelectedSequenceChart = () => {
            const canvas = document.getElementById('slidingWindowChart');
            const sequenceSelector = document.getElementById('sequenceSelector');
            
            if (!canvas || !currentSlidingWindowData || !sequenceSelector) return;

            try {
                const selectedValue = sequenceSelector.value;
                let selectedSample = null;
                let sampleType = '';

                if (selectedValue.startsWith('positive_')) {
                    const index = parseInt(selectedValue.replace('positive_', ''));
                    selectedSample = currentSlidingWindowData.positive_samples[index];
                    sampleType = 'Anti-Gram-negative Sample';
                } else if (selectedValue.startsWith('negative_')) {
                    const index = parseInt(selectedValue.replace('negative_', ''));
                    selectedSample = currentSlidingWindowData.negative_samples[index];
                    sampleType = 'Non-Anti-Gram-negative Sample';
                }

                if (!selectedSample) {
                    // 如果没有选择，显示提示信息
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = 'white';
                    ctx.font = '16px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('Please select a sequence from the dropdown menu', canvas.width / 2, canvas.height / 2);
                    return;
                }

                // 准备图表数据
                const datasets = [];
                const features = ['hydrophobicity', 'charge', 'hydrophobic_moment'];
                const featureNames = {
                    'hydrophobicity': 'Hydrophobicity',
                    'charge': 'Charge',
                    'hydrophobic_moment': 'Hydrophobic Moment'
                };
                const colors = {
                    'hydrophobicity': 'rgba(54, 162, 235, 0.8)',
                    'charge': 'rgba(255, 99, 132, 0.8)',
                    'hydrophobic_moment': 'rgba(75, 192, 192, 0.8)'
                };

                // 为每个特征创建数据集
                features.forEach((feature, index) => {
                    if (selectedSample && selectedSample.windows.length > 0) {
                        datasets.push({
                            label: `${featureNames[feature]}`,
                            data: selectedSample.windows.map(window => ({
                                x: window.position,
                                y: window[feature]
                            })),
                            borderColor: colors[feature],
                            backgroundColor: colors[feature],
                            borderWidth: 2,
                            pointRadius: 4,
                            tension: 0.3,
                            yAxisID: 'y'
                        });
                    }
                });

                // 销毁现有图表
                let existingChart = Chart.getChart(canvas);
                if (existingChart) existingChart.destroy();

                // 创建线性图
                new Chart(canvas.getContext('2d'), {
                    type: 'line',
                    data: {
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            mode: 'index',
                            intersect: false,
                        },
                        plugins: {
                            title: {
                                display: true,
                                text: `${selectedSample.id} (${sampleType}) - Sequence: ${selectedSample.sequence}`,
                                color: 'white',
                                font: {
                                    size: 14
                                }
                            },
                            legend: {
                                labels: { color: 'white' }
                            },
                            tooltip: {
                                callbacks: {
                                    title: function(tooltipItems) {
                                        return `Position: ${tooltipItems[0].parsed.x}`;
                                    },
                                    label: function(context) {
                                        const value = context.parsed.y.toFixed(3);
                                        return `${context.dataset.label}: ${value}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                type: 'linear',
                                position: 'bottom',
                                title: {
                                    display: true,
                                    text: 'Sequence Position',
                                    color: 'white'
                                },
                                ticks: {
                                    color: 'white',
                                    stepSize: 1
                                },
                                grid: {
                                    color: 'rgba(255,255,255,0.1)'
                                }
                            },
                            y: {
                                type: 'linear',
                                title: {
                                    display: true,
                                    text: 'Feature Value',
                                    color: 'white'
                                },
                                ticks: {
                                    color: 'white'
                                },
                                grid: {
                                    color: 'rgba(255,255,255,0.1)'
                                }
                            }
                        }
                    }
                });

            } catch (error) {
                console.error('Sliding window chart rendering error:', error);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('Sliding window chart rendering failed', canvas.width / 2, canvas.height / 2);
            }
        };

        const renderDimensionalityReductionCharts = (dimensionalityData) => {
            if (!dimensionalityData) {
                console.warn('Dimensionality reduction data is empty, skipping rendering');
                return;
            }

            // 渲染PCA图表
            renderPCAChart(dimensionalityData.pca);
            
            // 渲染t-SNE图表
            renderTSNEChart(dimensionalityData.tsne);
        };

        const renderPCAChart = (pcaData) => {
            const canvas = document.getElementById('pcaChart');
            if (!canvas || !pcaData || pcaData.length === 0) return;

            try {
                // 分离正负样本数据
                const positiveData = pcaData.filter(d => d.prediction === 1);
                const negativeData = pcaData.filter(d => d.prediction === 0);

                // 销毁现有图表
                let existingChart = Chart.getChart(canvas);
                if (existingChart) existingChart.destroy();

                // 创建散点图
                new Chart(canvas.getContext('2d'), {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: `Anti-Gram-negative (n=${positiveData.length})`,
                            data: positiveData.map(d => ({ x: d.x, y: d.y })),
                            backgroundColor: 'rgba(40, 167, 69, 0.7)',
                            borderColor: 'rgba(40, 167, 69, 1)',
                            pointRadius: 5,
                            pointHoverRadius: 7
                        }, {
                            label: `Non-Anti-Gram-negative (n=${negativeData.length})`,
                            data: negativeData.map(d => ({ x: d.x, y: d.y })),
                            backgroundColor: 'rgba(108, 117, 125, 0.7)',
                            borderColor: 'rgba(108, 117, 125, 1)',
                            pointRadius: 5,
                            pointHoverRadius: 7
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 'PCA Dimensionality Reduction Analysis',
                                color: 'white',
                                font: { size: 16 }
                            },
                            legend: {
                                labels: { color: 'white' }
                            },
                            tooltip: {
                                callbacks: {
                                    title: function(tooltipItems) {
                                        const dataIndex = tooltipItems[0].dataIndex;
                                        const datasetIndex = tooltipItems[0].datasetIndex;
                                        const data = datasetIndex === 0 ? positiveData : negativeData;
                                        return `Sequence: ${data[dataIndex].id}`;
                                    },
                                    label: function(context) {
                                        const dataIndex = context.dataIndex;
                                        const datasetIndex = context.datasetIndex;
                                        const data = datasetIndex === 0 ? positiveData : negativeData;
                                        const sample = data[dataIndex];
                                        return [
                                            `Prediction probability: ${sample.probability.toFixed(4)}`,
                                            `PCA coordinates: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`,
                                            `Sequence: ${sample.sequence.substring(0, 20)}...`
                                        ];
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'PC1',
                                    color: 'white'
                                },
                                ticks: { color: 'white' },
                                grid: { color: 'rgba(255,255,255,0.1)' }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'PC2',
                                    color: 'white'
                                },
                                ticks: { color: 'white' },
                                grid: { color: 'rgba(255,255,255,0.1)' }
                            }
                        }
                    }
                });

            } catch (error) {
                console.error('PCA chart rendering error:', error);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('PCA chart rendering failed', canvas.width / 2, canvas.height / 2);
            }
        };

        const renderTSNEChart = (tsneData) => {
            const canvas = document.getElementById('tsneChart');
            if (!canvas || !tsneData || tsneData.length === 0) return;

            try {
                // 分离正负样本数据
                const positiveData = tsneData.filter(d => d.prediction === 1);
                const negativeData = tsneData.filter(d => d.prediction === 0);

                // 销毁现有图表
                let existingChart = Chart.getChart(canvas);
                if (existingChart) existingChart.destroy();

                // 创建散点图
                new Chart(canvas.getContext('2d'), {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: `Anti-Gram-negative (n=${positiveData.length})`,
                            data: positiveData.map(d => ({ x: d.x, y: d.y })),
                            backgroundColor: 'rgba(40, 167, 69, 0.7)',
                            borderColor: 'rgba(40, 167, 69, 1)',
                            pointRadius: 5,
                            pointHoverRadius: 7
                        }, {
                            label: `Non-Anti-Gram-negative (n=${negativeData.length})`,
                            data: negativeData.map(d => ({ x: d.x, y: d.y })),
                            backgroundColor: 'rgba(108, 117, 125, 0.7)',
                            borderColor: 'rgba(108, 117, 125, 1)',
                            pointRadius: 5,
                            pointHoverRadius: 7
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            title: {
                                display: true,
                                text: 't-SNE Dimensionality Reduction Analysis',
                                color: 'white',
                                font: { size: 16 }
                            },
                            legend: {
                                labels: { color: 'white' }
                            },
                            tooltip: {
                                callbacks: {
                                    title: function(tooltipItems) {
                                        const dataIndex = tooltipItems[0].dataIndex;
                                        const datasetIndex = tooltipItems[0].datasetIndex;
                                        const data = datasetIndex === 0 ? positiveData : negativeData;
                                        return `Sequence: ${data[dataIndex].id}`;
                                    },
                                    label: function(context) {
                                        const dataIndex = context.dataIndex;
                                        const datasetIndex = context.datasetIndex;
                                        const data = datasetIndex === 0 ? positiveData : negativeData;
                                        const sample = data[dataIndex];
                                        return [
                                            `Prediction probability: ${sample.probability.toFixed(4)}`,
                                            `t-SNE coordinates: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`,
                                            `Sequence: ${sample.sequence.substring(0, 20)}...`
                                        ];
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 't-SNE Dimension 1',
                                    color: 'white'
                                },
                                ticks: { color: 'white' },
                                grid: { color: 'rgba(255,255,255,0.1)' }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 't-SNE Dimension 2',
                                    color: 'white'
                                },
                                ticks: { color: 'white' },
                                grid: { color: 'rgba(255,255,255,0.1)' }
                            }
                        }
                    }
                });

            } catch (error) {
                console.error('t-SNE chart rendering error:', error);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.textAlign = 'center';
                ctx.fillText('t-SNE chart rendering failed', canvas.width / 2, canvas.height / 2);
            }
        };
    };

    // --- GENERATION PAGE LOGIC ---
    const initGeneratePage = () => {
        const parametersSection = document.getElementById('parameters-section');
        const resultsSection = document.getElementById('resultsSection');
        const loadingSection = document.getElementById('loadingSection');
        const errorSection = document.getElementById('errorSection');
        const errorMessageEl = document.getElementById('errorMessage');

        const generateForm = document.getElementById('generateForm');
        const tempValueDisplay = document.getElementById('tempValue');
        const temperatureSlider = document.getElementById('temperature');
        
        const newGenerationBtn = document.getElementById('newGenerationBtn');
        const resetViewBtn = document.getElementById('resetViewBtn');
        const predictGeneratedBtn = document.getElementById('predictGeneratedBtn');
        const exportGeneratedBtn = document.getElementById('exportGeneratedBtn');
        const copyGeneratedFastaBtn = document.getElementById('copyGeneratedFastaBtn');
        const predictionForGeneratedResultsEl = document.getElementById('predictionForGeneratedResults');
        const predictionForGeneratedContentEl = document.getElementById('predictionForGeneratedContent');

        let generatedTable = null;
        let currentGeneratedData = [];

        const switchView = (view) => {
            parametersSection.classList.toggle('d-none', view === 'results');
            resultsSection.classList.toggle('d-none', view !== 'results');
            loadingSection.classList.add('d-none');
            errorSection.classList.add('d-none');
        };

        const handleGenerationSubmit = async () => {
            loadingSection.classList.remove('d-none');
            parametersSection.classList.add('d-none');
            errorSection.classList.add('d-none');

            const formData = {
                num_sequences: parseInt(document.getElementById('numSequences').value),
                seq_length: parseInt(document.getElementById('seqLength').value),
                sampling_method: document.getElementById('samplingMethod').value,
                temperature: parseFloat(document.getElementById('temperature').value),
                k: parseInt(document.getElementById('topK').value),
                p: parseFloat(document.getElementById('nucleusP').value),
                diversity_strength: parseFloat(document.getElementById('diversityStrength').value),
                reference_sequences: document.getElementById('referenceSeqs').value.split('\n').map(s => s.trim()).filter(s => s)
            };

            try {
                const response = await fetch('/api/generate_sequences', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                const data = await response.json();
                if (!response.ok || !data.success) {
                    throw new Error(data.error || 'Generation failed.');
                }
                renderResults(data);
                switchView('results');
            } catch (error) {
                errorMessageEl.textContent = error.message;
                errorSection.classList.remove('d-none');
                loadingSection.classList.add('d-none');
            }
        };

        const renderResults = (data) => {
            currentGeneratedData = data.sequences;
            const statsContainer = document.getElementById('generationStats');
            statsContainer.innerHTML = `
                <div class="row text-center">
                    <div class="col-md-3 col-6 mb-2"><small class="text-muted d-block">Generated</small><span class="badge bg-primary fs-6">${data.sequences.length}</span></div>
                    <div class="col-md-3 col-6 mb-2"><small class="text-muted d-block">Method</small><span class="badge bg-info fs-6 text-truncate">${data.parameters.sampling_method}</span></div>
                    <div class="col-md-3 col-6 mb-2"><small class="text-muted d-block">Length</small><span class="badge bg-secondary fs-6">${data.parameters.seq_length}</span></div>
                    <div class="col-md-3 col-6 mb-2"><small class="text-muted d-block">Temp</small><span class="badge bg-warning fs-6">${data.parameters.temperature}</span></div>
                </div>`;

            if ($.fn.dataTable.isDataTable('#generatedDataTable')) {
                generatedTable.destroy();
            }
            generatedTable = $('#generatedDataTable').DataTable({
                data: data.sequences,
                columns: [
                    { title: "ID", data: "id" },
                    { title: "Sequence", data: "sequence", className: "sequence-text" },
                    { title: "Length", data: "length" }
                ],
                responsive: true,
                "lengthMenu": [[5, 10, 25, -1], [5, 10, 25, "All"]],
                "pageLength": 5
            });
        };

        const handlePredictGenerated = async () => {
            if (currentGeneratedData.length === 0) return showToast('No sequences to predict.', 'warning');
            
            predictGeneratedBtn.disabled = true;
            predictGeneratedBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...';

            const fastaText = currentGeneratedData.map(s => `>${s.id}\n${s.sequence}`).join('\n');
            const formData = new FormData();
            formData.append('fasta_text', fastaText);

            try {
                const response = await fetch('/api/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if (!response.ok || !data.success) throw new Error(data.error || 'Prediction failed.');
                
                let resultsHtml = '<ul class="list-group">';
                data.results.forEach(res => {
                    resultsHtml += `<li class="list-group-item d-flex justify-content-between align-items-center">
                        ${res.id}: <code class="mx-2">${res.sequence.substring(0,15)}...</code>
                        <span class="badge ${res.prediction === 1 ? 'bg-success' : 'bg-secondary'}">${res.label} (${res.probability.toFixed(3)})</span>
                    </li>`;
                });
                resultsHtml += '</ul>';
                predictionForGeneratedContentEl.innerHTML = resultsHtml;
                predictionForGeneratedResultsEl.style.display = 'block';
                showToast('Prediction complete!', 'success');

            } catch (error) {
                showToast(error.message, 'danger');
            } finally {
                predictGeneratedBtn.disabled = false;
                predictGeneratedBtn.innerHTML = '<i class="bi bi-search"></i> Predict Activity';
            }
        };

        // Event Listeners
        generateForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            handleGenerationSubmit();
        });

        temperatureSlider?.addEventListener('input', () => {
            if(tempValueDisplay) tempValueDisplay.textContent = temperatureSlider.value;
        });

        newGenerationBtn?.addEventListener('click', () => switchView('params'));
        resetViewBtn?.addEventListener('click', () => switchView('params'));
        predictGeneratedBtn?.addEventListener('click', handlePredictGenerated);

        exportGeneratedBtn?.addEventListener('click', () => {
            if (currentGeneratedData.length === 0) return showToast('No data to export.', 'warning');
            const csvHeader = "data:text/csv;charset=utf-8,ID,Sequence,Length\n";
            const csvBody = currentGeneratedData.map(s => `${s.id},${s.sequence},${s.length}`).join('\n');
            const encodedUri = encodeURI(csvHeader + csvBody);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "generated_sequences.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });

        copyGeneratedFastaBtn?.addEventListener('click', () => {
            if (currentGeneratedData.length === 0) return showToast('No data to copy.', 'warning');
            const fastaText = currentGeneratedData.map(s => `>${s.id}\n${s.sequence}`).join('\n');
            navigator.clipboard.writeText(fastaText).then(() => {
                showToast('FASTA copied to clipboard!', 'success');
            }, () => {
                showToast('Failed to copy.', 'danger');
            });
        });
    };

    // --- DATABASE PAGE LOGIC ---
    const initDatabasePage = () => {
        console.log('Database page initialized');
        
        // Initialize database functionality
        let allSequences = [];
        let filteredSequences = [];
        let currentPage = 1;
        const sequencesPerPage = 25;

        // Show toast message function
        function showToast(message, type = 'info') {
            if (window.showToast) {
                window.showToast(message, type);
            } else {
                console.log(`${type.toUpperCase()}: ${message}`);
                alert(message);
            }
        }

        // Load database data
        async function loadDatabase() {
            console.log('Starting loadDatabase function...');
            
            // Show loading
            const loadingEl = document.getElementById('database-loading');
            const errorEl = document.getElementById('database-error');
            const tableEl = document.getElementById('database-table-container');
            
            console.log('DOM elements check:');
            console.log('- database-loading:', loadingEl);
            console.log('- database-error:', errorEl);
            console.log('- database-table-container:', tableEl);
            
            if (loadingEl) loadingEl.style.display = 'block';
            if (errorEl) errorEl.style.display = 'none';
            if (tableEl) tableEl.style.display = 'none';
            
            try {
                console.log('Fetching /api/database...');
                const response = await fetch('/api/database');
                console.log('Response status:', response.status);
                
                const data = await response.json();
                console.log('Response data keys:', Object.keys(data));
                console.log('Number of sequences:', data.sequences ? data.sequences.length : 0);
                
                if (!response.ok || !data.success) {
                    throw new Error(data.error || 'Failed to load database');
                }

                allSequences = data.sequences;
                filteredSequences = [...allSequences];
                console.log(`Successfully loaded ${allSequences.length} sequences`);
                
                updateStats(data.stats);
                renderTable();
                
                console.log('Hiding loading, showing table...');
                if (loadingEl) loadingEl.style.display = 'none';
                if (tableEl) tableEl.style.display = 'block';

            } catch (error) {
                console.error('Database loading error:', error);
                const errorMsgEl = document.getElementById('database-error-message');
                if (errorMsgEl) errorMsgEl.textContent = error.message;
                if (errorEl) errorEl.style.display = 'block';
                if (loadingEl) loadingEl.style.display = 'none';
            }
        }

        // Update statistics
        function updateStats(stats) {
            const totalEl = document.getElementById('total-sequences');
            const avgEl = document.getElementById('avg-length');
            if (totalEl) totalEl.textContent = stats.total;
            if (avgEl) avgEl.textContent = stats.avg_length.toFixed(1);
        }

        // Render table
        function renderTable() {
            const startIndex = (currentPage - 1) * sequencesPerPage;
            const endIndex = startIndex + sequencesPerPage;
            const pageSequences = filteredSequences.slice(startIndex, endIndex);
            
            const tbody = document.getElementById('sequence-table-body');
            if (!tbody) {
                console.error('Table body not found');
                return;
            }
            
            tbody.innerHTML = '';
            
            pageSequences.forEach(seq => {
                const row = document.createElement('tr');
                const truncatedSeq = seq.sequence.length > 50 ? 
                    seq.sequence.substring(0, 50) + '...' : seq.sequence;
                
                row.innerHTML = `
                    <td>${seq.id}</td>
                    <td><code class="text-light">${truncatedSeq}</code></td>
                    <td>${seq.length}</td>
                `;
                tbody.appendChild(row);
            });
            
            updatePagination();
        }

        // Update pagination
        function updatePagination() {
            const totalPages = Math.ceil(filteredSequences.length / sequencesPerPage);
            const startIndex = (currentPage - 1) * sequencesPerPage + 1;
            const endIndex = Math.min(currentPage * sequencesPerPage, filteredSequences.length);
            
            const paginationInfo = document.getElementById('pagination-info');
            const pageInfo = document.getElementById('page-info');
            
            if (paginationInfo) {
                paginationInfo.textContent = `Showing ${startIndex}-${endIndex} of ${filteredSequences.length} sequences`;
            }
            if (pageInfo) {
                pageInfo.textContent = `Page ${currentPage} of ${totalPages}`;
            }
            
            const prevBtn = document.getElementById('prevPage');
            const nextBtn = document.getElementById('nextPage');
            
            if (prevBtn) prevBtn.disabled = currentPage === 1;
            if (nextBtn) nextBtn.disabled = currentPage === totalPages || totalPages === 0;
        }

        // Add event listeners for pagination and other controls
        function setupEventListeners() {
            // Search functionality
            const searchInput = document.getElementById('sequenceSearch');
            if (searchInput) {
                searchInput.addEventListener('input', performSearch);
            }
            
            const clearSearchBtn = document.getElementById('clearSearch');
            if (clearSearchBtn) {
                clearSearchBtn.addEventListener('click', function() {
                    const searchInput = document.getElementById('sequenceSearch');
                    if (searchInput) {
                        searchInput.value = '';
                        performSearch();
                    }
                });
            }


            // Pagination buttons
            const prevBtn = document.getElementById('prevPage');
            if (prevBtn) {
                prevBtn.addEventListener('click', function() {
                    console.log('Previous page clicked, current page:', currentPage);
                    if (currentPage > 1) {
                        currentPage--;
                        renderTable();
                    }
                });
            }

            const nextBtn = document.getElementById('nextPage');
            if (nextBtn) {
                nextBtn.addEventListener('click', function() {
                    const totalPages = Math.ceil(filteredSequences.length / sequencesPerPage);
                    console.log('Next page clicked, current page:', currentPage, 'total pages:', totalPages);
                    if (currentPage < totalPages) {
                        currentPage++;
                        renderTable();
                    }
                });
            }

            // Download FASTA
            const downloadBtn = document.getElementById('downloadFasta');
            if (downloadBtn) {
                downloadBtn.addEventListener('click', function() {
                    const fastaContent = allSequences.map(seq => `>${seq.id}\n${seq.sequence}`).join('\n');
                    downloadFile(fastaContent, 'gram_negative_database.fasta', 'text/plain');
                });
            }

        }

        // Search functionality
        function performSearch() {
            const searchInput = document.getElementById('sequenceSearch');
            if (!searchInput) return;
            
            const searchTerm = searchInput.value.toLowerCase();
            if (searchTerm === '') {
                filteredSequences = [...allSequences];
            } else {
                filteredSequences = allSequences.filter(seq => 
                    seq.id.toLowerCase().includes(searchTerm) || 
                    seq.sequence.toLowerCase().includes(searchTerm)
                );
            }
            currentPage = 1;
            renderTable();
        }

        // Helper function to download file
        function downloadFile(content, filename, contentType) {
            const blob = new Blob([content], { type: contentType });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        }

        // Initialize database loading
        setTimeout(() => {
            console.log('Initializing database loading...');
            loadDatabase();
            // Setup event listeners after a short delay to ensure DOM is ready
            setTimeout(setupEventListeners, 200);
        }, 100);
    };