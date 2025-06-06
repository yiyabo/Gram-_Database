document.addEventListener('DOMContentLoaded', () => {
    const appContainer = document.getElementById('app-container');
    const navLinks = document.querySelectorAll('[data-nav-link]');

    const routes = {
        '/': '/content/predict',
        '/predict': '/content/predict',
        '/generate': '/content/generate',
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

        const renderResults = (data) => {
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

            renderFeatureBarChart(data.results);

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

        const renderFeatureBarChart = (results) => {
            const canvas = document.getElementById('featureBarChart');
            if (!canvas) return;

            const positiveData = results.filter(r => r.prediction === 1);
            const negativeData = results.filter(r => r.prediction === 0);
            const features = ['Charge', 'Hydrophobicity', 'Hydrophobic_Moment'];

            const calcAverage = (data, key) => {
                if (data.length === 0) return 0;
                const sum = data.reduce((acc, curr) => acc + curr.features[key], 0);
                return sum / data.length;
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
                            label: 'Positive',
                            data: positiveAvgs,
                            backgroundColor: 'rgba(40, 167, 69, 0.7)',
                            borderColor: 'rgba(40, 167, 69, 1)',
                            borderWidth: 1
                        },
                        {
                            label: 'Negative',
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