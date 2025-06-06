document.addEventListener('DOMContentLoaded', () => {
    // tsParticles background initialization
    if (document.getElementById('tsparticles-background') && typeof tsParticles !== 'undefined') {
        tsParticles.load("tsparticles-background", {
            fpsLimit: 60, background: { color: "transparent" },
            particles: {
                number: { value: 100, density: { enable: true, area: 800 } },
                color: { value: ["#333333", "#666666", "#999999", "#00E5FF", "#FFFFFF"] },
                shape: { type: "square" },
                opacity: { value: { min: 0.1, max: 0.6 }, animation: { enable: true, speed: 0.5, minimumValue: 0.1, sync: false } },
                size: { value: { min: 1, max: 4 }, animation: { enable: true, speed: 2, minimumValue: 0.5, sync: false } },
                links: { enable: true, distance: 120, color: "#444444", opacity: 0.3, width: 1 },
                move: { enable: true, speed: 0.8, direction: "none", random: true, straight: false, outModes: { default: "out" }, attract: { enable: false } }
            },
            interactivity: {
                detectsOn: "canvas",
                events: { onHover: { enable: true, mode: "repulse" }, onClick: { enable: true, mode: "push" }, resize: true },
                modes: { repulse: { distance: 80, duration: 0.4 }, push: { quantity: 3 } }
            },
            detectRetina: true
        }).then(() => console.log("tsParticles loaded successfully for main.js context"))
          .catch(error => console.error("Error loading tsParticles in main.js:", error));
    } else if (document.getElementById('tsparticles-background')) {
        console.error("tsParticles library not found (main.js).");
    }

    const PREDICTION_RESULT_STORAGE_KEY = 'gramNegativePredictionResult';

    // Shared DOM elements
    const loadingSection = document.getElementById('loadingSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessageEl = document.getElementById('errorMessage');
    let resultChart = null;

    // --- GENERAL HELPER: SHOW TOAST NOTIFICATION ---
    function showToast(message, type = 'info', title = '') {
        const toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            console.warn('Toast container not found on this page. Falling back to alert.');
            alert((title ? title + ': ' : '') + message);
            return;
        }

        const toastId = 'toast-' + Math.random().toString(36).substring(2, 9);
        let iconHtml = '', headerClass = '', toastClass = ''; // toastClass for potential custom styling

        switch (type.toLowerCase()) {
            case 'success':
                iconHtml = '<i class="bi bi-check-circle-fill text-success me-2"></i>';
                headerClass = 'text-success';
                toastClass = 'toast-success';
                if (!title) title = 'Success';
                break;
            case 'danger':
            case 'error':
                iconHtml = '<i class="bi bi-x-octagon-fill text-danger me-2"></i>';
                headerClass = 'text-danger';
                toastClass = 'toast-danger';
                if (!title) title = 'Error';
                break;
            case 'warning':
                iconHtml = '<i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>';
                headerClass = 'text-warning';
                toastClass = 'toast-warning';
                if (!title) title = 'Warning';
                break;
            default: // info
                iconHtml = '<i class="bi bi-info-circle-fill text-info me-2"></i>';
                headerClass = 'text-info';
                toastClass = 'toast-info';
                if (!title) title = 'Information';
                break;
        }

        const toastHtml = `
            <div class="toast ${toastClass}" role="alert" aria-live="assertive" aria-atomic="true" id="${toastId}" data-bs-delay="4000">
                <div class="toast-header">
                    ${iconHtml}
                    <strong class="me-auto ${headerClass}">${title}</strong>
                    <small class="text-muted">Just now</small>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>`;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);

        if (toastElement && typeof bootstrap !== 'undefined' && bootstrap.Toast) {
            const bsToast = new bootstrap.Toast(toastElement);
            bsToast.show();
            toastElement.addEventListener('hidden.bs.toast', () => {
                bsToast.dispose(); 
                toastElement.remove();
            });
        } else {
            console.error('Bootstrap Toast API not available or toast element not found.');
        }
    }

    // --- SHARED HELPER FUNCTIONS for UI state ---
    function showLoadingState(isLoading, pageType = 'submit') {
        if (loadingSection) loadingSection.classList.toggle('d-none', !isLoading);
        if (errorSection) errorSection.classList.add('d-none'); // Always hide error when loading changes

        if (pageType === 'submit') {
            // Select the main card body that contains tabs and tab content for submission
            const submitCardBody = document.querySelector('.submit-form-container .card-body');
            const submitButtons = document.querySelectorAll('#fileForm button[type="submit"], #textForm button[type="submit"]');
            
            if (submitCardBody) { // Hide the entire card body content area
                // Find direct children that are nav-tabs and tab-content to hide them
                 const navTabs = submitCardBody.querySelector('.nav-tabs');
                 const tabContent = submitCardBody.querySelector('.tab-content');
                 if(navTabs) navTabs.style.display = isLoading ? 'none' : '';
                 if(tabContent) tabContent.style.display = isLoading ? 'none' : '';
            }
            submitButtons.forEach(btn => btn.disabled = isLoading);

        } else if (pageType === 'results') {
            const resultsSectionEl = document.getElementById('resultsSection');
            if (resultsSectionEl) resultsSectionEl.classList.toggle('d-none', isLoading);
        }
    }

    function displayError(message, pageType = 'submit') {
        showLoadingState(false, pageType); // Ensure loading is hidden
        if (errorSection) errorSection.classList.remove('d-none');
        if (errorMessageEl) errorMessageEl.textContent = message;
        
        if (pageType === 'submit') { // Show forms again on submit page error
            const submitCardBody = document.querySelector('.submit-form-container .card-body');
             if (submitCardBody) {
                 const navTabs = submitCardBody.querySelector('.nav-tabs');
                 const tabContent = submitCardBody.querySelector('.tab-content');
                 if(navTabs) navTabs.style.display = '';
                 if(tabContent) tabContent.style.display = '';
            }
            const submitButtons = document.querySelectorAll('#fileForm button[type="submit"], #textForm button[type="submit"]');
            submitButtons.forEach(btn => btn.disabled = false);
        }
    }
    
    // --- SUBMIT PAGE SPECIFIC LOGIC (predict_submit.html) ---
    if (window.location.pathname === '/' || window.location.pathname.endsWith('predict_submit.html') || window.location.pathname.endsWith('/predict/')) {
        const fileForm = document.getElementById('fileForm');
        const textForm = document.getElementById('textForm');
        const fastaFileEl = document.getElementById('fastaFile');
        const fastaTextEl = document.getElementById('fastaText');
        const dropZone = document.getElementById('dropZone');
        const selectedFileNameEl = dropZone ? dropZone.querySelector('.selected-file-name') : null;
        const loadExampleBtn = document.getElementById('loadExample');
        const resetFormsBtnOnSubmitError = errorSection ? errorSection.querySelector('#resetForms') : null;

        function resetSubmitPageForms() {
            showLoadingState(false, 'submit'); 
            if (fastaFileEl) fastaFileEl.value = '';
            if (fastaTextEl) fastaTextEl.value = '';
            if (selectedFileNameEl) selectedFileNameEl.textContent = '';
            if (dropZone) {
                dropZone.classList.remove('file-dropped', 'drag-over');
                const dropZoneIcon = dropZone.querySelector('.file-upload-content .bi');
                if (dropZoneIcon) dropZoneIcon.classList.remove('file-upload-icon-pulse');
            }
        }
        
        async function handlePredictionSubmit(formData) {
            showLoadingState(true, 'submit');
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                if (response.ok && data.success) {
                    localStorage.setItem(PREDICTION_RESULT_STORAGE_KEY, JSON.stringify(data));
                    window.location.href = '/predict/results';
                } else {
                    displayError(data.error || 'Prediction failed. Please check your input.', 'submit');
                }
            } catch (error) {
                console.error('Prediction submission error:', error);
                displayError('Server connection error during prediction. Please try again.', 'submit');
            }
        }

        if (fileForm && fastaFileEl) {
            fileForm.addEventListener('submit', (e) => {
                e.preventDefault();
                if (fastaFileEl.files.length === 0) {
                    displayError('Please select a FASTA file.', 'submit');
                    return;
                }
                const formData = new FormData();
                formData.append('fasta_file', fastaFileEl.files[0]);
                handlePredictionSubmit(formData);
            });
        }

        if (textForm && fastaTextEl) {
            textForm.addEventListener('submit', (e) => {
                e.preventDefault();
                const fastaContent = fastaTextEl.value.trim();
                if (!fastaContent) {
                    displayError('Please enter FASTA sequence data.', 'submit');
                    return;
                }
                const formData = new FormData();
                formData.append('fasta_text', fastaContent);
                handlePredictionSubmit(formData);
            });
        }

        if (fastaFileEl && selectedFileNameEl) {
            fastaFileEl.addEventListener('change', () => {
                if (fastaFileEl.files.length > 0) {
                    selectedFileNameEl.textContent = fastaFileEl.files[0].name;
                    if (dropZone) dropZone.classList.add('file-dropped');
                } else {
                    selectedFileNameEl.textContent = '';
                    if (dropZone) dropZone.classList.remove('file-dropped');
                }
            });
        }

        if (dropZone) {
            const dropZoneIcon = dropZone.querySelector('.file-upload-content .bi');
            dropZone.addEventListener('dragenter', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); if (dropZoneIcon) dropZoneIcon.classList.add('file-upload-icon-pulse'); });
            dropZone.addEventListener('dragover', (e) => e.preventDefault());
            dropZone.addEventListener('dragleave', (e) => { if (e.target === dropZone || !dropZone.contains(e.relatedTarget)) { dropZone.classList.remove('drag-over'); if (dropZoneIcon) dropZoneIcon.classList.remove('file-upload-icon-pulse'); }});
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                if (dropZoneIcon) dropZoneIcon.classList.remove('file-upload-icon-pulse');
                if (e.dataTransfer.files.length > 0) {
                    const file = e.dataTransfer.files[0];
                    if (file.name.endsWith('.fasta') || file.name.endsWith('.fa') || file.name.endsWith('.txt')) {
                        if (fastaFileEl) fastaFileEl.files = e.dataTransfer.files;
                        if (selectedFileNameEl) selectedFileNameEl.textContent = file.name;
                        if (dropZone) dropZone.classList.add('file-dropped');
                    } else {
                        displayError('Invalid file type. Please upload a .fasta, .fa, or .txt file.', 'submit');
                        if (selectedFileNameEl) selectedFileNameEl.textContent = '';
                        if (dropZone) dropZone.classList.remove('file-dropped');
                    }
                }
            });
            dropZone.addEventListener('click', (e) => { if (e.target.closest('button') || e.target.closest('label') || e.target.tagName === 'INPUT') return; if (fastaFileEl) fastaFileEl.click(); });
        }

        if (loadExampleBtn && fastaTextEl) {
            loadExampleBtn.addEventListener('click', async () => {
                try {
                    const response = await fetch('/example');
                    const data = await response.json();
                    if (data.fasta) {
                        fastaTextEl.value = data.fasta;
                        const textTabButton = document.getElementById('text-tab');
                        if (textTabButton && typeof bootstrap !== 'undefined' && bootstrap.Tab) new bootstrap.Tab(textTabButton).show();
                    }
                } catch (error) { console.error('Error loading example:', error); displayError('Failed to load example data.', 'submit'); }
            });
        }
        
        if (resetFormsBtnOnSubmitError) { // This button is inside the errorSection on submit page
            resetFormsBtnOnSubmitError.addEventListener('click', resetSubmitPageForms);
        }
        resetSubmitPageForms(); // Initial UI state for submit page
    }
    // --- RESULTS PAGE SPECIFIC LOGIC (predict_results.html) ---
    else if (window.location.pathname === '/predict/results') {
        const resultsSectionEl = document.getElementById('resultsSection');
        const totalSequencesEl = document.getElementById('totalSequences');
        const positiveCountEl = document.getElementById('positiveCount');
        const negativeCountEl = document.getElementById('negativeCount');
        const positivePercentageEl = document.getElementById('positivePercentage');
        const resultsTableBodyEl = document.getElementById('resultsTableBody');
        const exportCSVBtn = document.getElementById('exportCSV');
        const exportFASTABtn = document.getElementById('exportFASTA');
        let currentResultsDataForExport = null;

        function renderResultsToPage(data) {
            showLoadingState(false, 'results');
            if (resultsSectionEl) resultsSectionEl.classList.remove('d-none');
            
            currentResultsDataForExport = data.results;

            if (totalSequencesEl) totalSequencesEl.textContent = data.stats.total;
            if (positiveCountEl) positiveCountEl.textContent = data.stats.positive;
            if (negativeCountEl) negativeCountEl.textContent = data.stats.negative; // Make sure this element exists in predict_results.html
            if (positivePercentageEl) positivePercentageEl.textContent = `${data.stats.positive_percentage}%`;

            if (resultsTableBodyEl) {
                resultsTableBodyEl.innerHTML = '';
                const SEQUENCE_TRUNCATE_LENGTH = 30;
                data.results.forEach(result => {
                    const row = resultsTableBodyEl.insertRow();
                    row.insertCell().textContent = result.id;

                    const sequenceCell = row.insertCell();
                    const sequenceWrapper = document.createElement('div');
                    sequenceWrapper.classList.add('sequence-wrapper');
                    const sequenceContent = document.createElement('span');
                    sequenceContent.classList.add('sequence-content');
                    const fullSequence = result.sequence;
                    let isExpanded = false;

                    const updateSequenceDisplay = () => {
                        sequenceContent.innerHTML = '';
                        if (isExpanded) {
                            const fullSpan = document.createElement('span');
                            fullSpan.classList.add('sequence-full');
                            fullSpan.textContent = fullSequence;
                            sequenceContent.appendChild(fullSpan);
                        } else {
                            sequenceContent.textContent = fullSequence.length > SEQUENCE_TRUNCATE_LENGTH 
                                                        ? fullSequence.substring(0, SEQUENCE_TRUNCATE_LENGTH) + '...' 
                                                        : fullSequence;
                        }
                    };
                    updateSequenceDisplay();
                    sequenceWrapper.appendChild(sequenceContent);

                    if (fullSequence.length > SEQUENCE_TRUNCATE_LENGTH) {
                        const toggleBtn = document.createElement('button');
                        toggleBtn.classList.add('btn', 'btn-link', 'btn-sm', 'p-0', 'toggle-sequence-view');
                        toggleBtn.type = 'button';
                        toggleBtn.textContent = 'View More';
                        toggleBtn.addEventListener('click', () => {
                            isExpanded = !isExpanded;
                            updateSequenceDisplay();
                            toggleBtn.textContent = isExpanded ? 'View Less' : 'View More';
                        });
                        sequenceWrapper.appendChild(toggleBtn);
                    }
                    sequenceCell.appendChild(sequenceWrapper);

                    const probabilityCell = row.insertCell();
                    probabilityCell.textContent = result.probability !== -1.0 ? result.probability.toFixed(4) : 'N/A';
                    
                    const predictionCell = row.insertCell();
                    const badge = document.createElement('span');
                    badge.classList.add('badge');
                    if (result.prediction === 1) badge.classList.add('bg-success');
                    else if (result.prediction === 0) badge.classList.add('bg-secondary');
                    else badge.classList.add('bg-danger');
                    badge.textContent = result.label || (result.prediction === 1 ? 'Anti-Gram-Negative' : (result.prediction === 0 ? 'Non-Anti-Gram-Negative' : 'Error'));
                    predictionCell.appendChild(badge);

                    const actionsCell = row.insertCell();
                    actionsCell.classList.add('actions-cell');
                    const copyBtn = document.createElement('button');
                    copyBtn.classList.add('btn', 'btn-sm', 'copy-sequence-btn');
                    copyBtn.title = 'Copy Sequence';
                    copyBtn.innerHTML = '<i class="bi bi-clipboard"></i>';
                    copyBtn.addEventListener('click', async () => {
                        try {
                            await navigator.clipboard.writeText(fullSequence);
                            showToast('Sequence copied to clipboard!', 'success');
                            copyBtn.innerHTML = '<i class="bi bi-check-lg text-success"></i>';
                            setTimeout(() => { copyBtn.innerHTML = '<i class="bi bi-clipboard"></i>'; }, 2000);
                        } catch (err) {
                            showToast('Failed to copy sequence.', 'danger');
                            console.error('Failed to copy sequence: ', err);
                            copyBtn.innerHTML = '<i class="bi bi-x-lg text-danger"></i>';
                            setTimeout(() => { copyBtn.innerHTML = '<i class="bi bi-clipboard"></i>'; }, 2000);
                        }
                    });
                    actionsCell.appendChild(copyBtn);
                });
            }
            if (typeof Chart !== 'undefined' && data.stats) renderResultsChart(data.stats);
        }
        
        showLoadingState(true, 'results'); // Show loading initially on results page
        const storedResult = localStorage.getItem(PREDICTION_RESULT_STORAGE_KEY);
        if (storedResult) {
            try {
                const data = JSON.parse(storedResult);
                renderResultsToPage(data);
                // localStorage.removeItem(PREDICTION_RESULT_STORAGE_KEY); // Consider clearing after use
            } catch (e) {
                console.error("Error parsing stored results:", e);
                displayError("Could not display stored results. Please try submitting again.", 'results');
            }
        } else {
            displayError("No prediction results found. Please submit sequences for prediction first.", 'results');
        }

        async function exportResultsData(format) {
            if (!currentResultsDataForExport || currentResultsDataForExport.length === 0) {
                showToast('No results to export.', 'warning', 'Export');
                return;
            }
            showToast(`Preparing ${format.toUpperCase()} export...`, 'info', 'Export');
            try {
                const response = await fetch('/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ results: currentResultsDataForExport, format: format })
                });
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none'; a.href = url;
                    const contentDisposition = response.headers.get('content-disposition');
                    let filename = `prediction_results.${format}`;
                    if (contentDisposition) {
                        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
                        if (filenameMatch && filenameMatch.length > 1) filename = filenameMatch[1];
                    }
                    a.download = filename; document.body.appendChild(a); a.click();
                    window.URL.revokeObjectURL(url); a.remove();
                    showToast(`Results exported as ${format.toUpperCase()}.`, 'success', 'Export Successful');
                } else {
                    const errorData = await response.json();
                    showToast(errorData.error || `Failed to export ${format.toUpperCase()}.`, 'danger', 'Export Error');
                }
            } catch (error) {
                console.error(`Export ${format} error:`, error);
                showToast(`An error occurred while exporting ${format.toUpperCase()}.`, 'danger', 'Export Error');
            }
        }

        if (exportCSVBtn) exportCSVBtn.addEventListener('click', () => exportResultsData('csv'));
        if (exportFASTABtn) exportFASTABtn.addEventListener('click', () => exportResultsData('fasta'));
    }

    function renderResultsChart(stats) {
        const chartEl = document.getElementById('resultChart');
        if (!chartEl || typeof Chart === 'undefined' || !stats) {
            console.warn('Chart element, Chart.js, or stats not available for rendering chart.');
            return;
        }
        const ctx = chartEl.getContext('2d');
        if (resultChart) resultChart.destroy();
        
        const failedCount = stats.failed !== undefined ? stats.failed : (stats.total - stats.positive - (stats.negative !== undefined ? stats.negative : 0));

        resultChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Predicted Positive', 'Predicted Negative', 'Failed/Other'],
                datasets: [{
                    label: 'Prediction Distribution',
                    data: [stats.positive || 0, stats.negative || 0, failedCount < 0 ? 0 : failedCount],
                    backgroundColor: ['rgba(40, 167, 69, 0.8)', 'rgba(108, 117, 125, 0.8)', 'rgba(220, 53, 69, 0.7)'],
                    borderColor: ['rgba(40, 167, 69, 1)', 'rgba(108, 117, 125, 1)', 'rgba(220, 53, 69, 1)'],
                    borderWidth: 1.5, hoverOffset: 8
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false, animation: { animateScale: true, animateRotate: true },
                plugins: {
                    legend: { position: 'bottom', labels: { padding: 20, font: { size: 13 } } },
                    title: { display: true, text: 'Prediction Outcome Distribution', font: { size: 16, weight: '600' }, padding: { top:10, bottom:20 } },
                    tooltip: { callbacks: { label: (context) => (context.label || '') + (context.parsed !== null ? ': ' + context.parsed : '') } }
                }
            }
        });
    }
});
