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
        }).then(() => console.log("tsParticles loaded for generate page"))
          .catch(error => console.error("Error loading tsParticles for generate page:", error));
    } else if (document.getElementById('tsparticles-background')) {
        console.error("tsParticles library not found for generate page.");
    }

    const GENERATION_RESULT_STORAGE_KEY = 'gramNegativeGenerationResult';
    // const PREDICTION_RESULT_STORAGE_KEY_FROM_GENERATE = 'gramNegativePredictionResultFromGenerate'; // Not strictly needed if displaying on same page

    // Shared DOM Elements
    const loadingSection = document.getElementById('loadingSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessageEl = document.getElementById('errorMessage');

    // --- GENERAL HELPER: SHOW TOAST NOTIFICATION ---
    function showToast(message, type = 'info', title = '') {
        const toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            console.warn('Toast container not found on this page. Falling back to alert.');
            alert((title ? title + ': ' : '') + message);
            return;
        }

        const toastId = 'toast-gen-' + Math.random().toString(36).substring(2, 9);
        let iconHtml = '', headerClass = '', toastClass = '';

        switch (type.toLowerCase()) {
            case 'success':
                iconHtml = '<i class="bi bi-check-circle-fill text-success me-2"></i>';
                headerClass = 'text-success'; toastClass = 'toast-success';
                if (!title) title = 'Success';
                break;
            case 'danger': case 'error':
                iconHtml = '<i class="bi bi-x-octagon-fill text-danger me-2"></i>';
                headerClass = 'text-danger'; toastClass = 'toast-danger';
                if (!title) title = 'Error';
                break;
            case 'warning':
                iconHtml = '<i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>';
                headerClass = 'text-warning'; toastClass = 'toast-warning';
                if (!title) title = 'Warning';
                break;
            default: // info
                iconHtml = '<i class="bi bi-info-circle-fill text-info me-2"></i>';
                headerClass = 'text-info'; toastClass = 'toast-info';
                if (!title) title = 'Information';
                break;
        }

        const toastHtml = `
            <div class="toast ${toastClass}" role="alert" aria-live="assertive" aria-atomic="true" id="${toastId}" data-bs-delay="4000">
                <div class="toast-header">
                    ${iconHtml}<strong class="me-auto ${headerClass}">${title}</strong>
                    <small class="text-muted">Just now</small>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">${message}</div>
            </div>`;
        
        toastContainer.insertAdjacentHTML('beforeend', toastHtml);
        const toastElement = document.getElementById(toastId);
        if (toastElement && typeof bootstrap !== 'undefined' && bootstrap.Toast) {
            const bsToast = new bootstrap.Toast(toastElement);
            bsToast.show();
            toastElement.addEventListener('hidden.bs.toast', () => { bsToast.dispose(); toastElement.remove(); });
        } else {
            console.error('Bootstrap Toast API not available or toast element not found (generate.js).');
        }
    }

    function showGenLoading(isLoading, pageType = 'params') {
        if (loadingSection) loadingSection.style.display = isLoading ? 'block' : 'none';
        if (errorSection) errorSection.style.display = 'none';
        if (pageType === 'params') {
            const generateBtn = document.getElementById('generateBtn');
            if (generateBtn) generateBtn.disabled = isLoading;
        } else if (pageType === 'results') {
            const resultsSectionEl = document.getElementById('resultsSection'); // resultsSection is a good name
            if (resultsSectionEl) resultsSectionEl.style.display = isLoading ? 'none' : 'block';
        }
    }

    function showGenError(message, pageType = 'params') {
        showGenLoading(false, pageType); // Ensure loading is hidden
        if (errorSection) errorSection.style.display = 'block';
        if (errorMessageEl) errorMessageEl.textContent = message;
        if (pageType === 'params') {
            const generateBtn = document.getElementById('generateBtn');
            if (generateBtn) generateBtn.disabled = false;
        }
    }

    // --- PARAMETERS PAGE LOGIC (generate_params.html) ---
    if (window.location.pathname.endsWith('/generate') || window.location.pathname.endsWith('/generate/')) {
        const generateForm = document.getElementById('generateForm');
        const temperatureSlider = document.getElementById('temperature');
        const tempValueDisplay = document.getElementById('tempValue');
        const resetGenerateFormBtn = document.getElementById('resetGenerateFormBtn');

        if (temperatureSlider && tempValueDisplay) {
            temperatureSlider.addEventListener('input', function() { tempValueDisplay.textContent = this.value; });
        }

        if (generateForm) {
            generateForm.addEventListener('submit', async function(e) {
                e.preventDefault();
                showGenLoading(true, 'params');
                const formData = {
                    num_sequences: parseInt(document.getElementById('numSequences').value),
                    seq_length: parseInt(document.getElementById('seqLength').value),
                    sampling_method: document.getElementById('samplingMethod').value,
                    temperature: parseFloat(document.getElementById('temperature').value),
                    k: parseInt(document.getElementById('topK').value),
                    p: parseFloat(document.getElementById('nucleusP').value),
                    diversity_strength: parseFloat(document.getElementById('diversityStrength').value),
                    reference_sequences: document.getElementById('referenceSeqs').value.split('\n').map(s => s.trim()).filter(s => s.length > 0)
                };
                try {
                    const response = await fetch('/generate_sequences', {
                        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(formData)
                    });
                    const result = await response.json();
                    if (result.success) {
                        localStorage.setItem(GENERATION_RESULT_STORAGE_KEY, JSON.stringify(result));
                        window.location.href = '/generate/results';
                    } else {
                        showGenError(result.error || 'Sequence generation failed.', 'params');
                    }
                } catch (error) {
                    console.error('Generation submission error:', error);
                    showGenError('Server connection error during generation.', 'params');
                }
            });
        }
        if (resetGenerateFormBtn) {
            resetGenerateFormBtn.addEventListener('click', () => {
                 if (generateForm) generateForm.reset();
                 if (temperatureSlider && tempValueDisplay) tempValueDisplay.textContent = temperatureSlider.value; // Reset display
                 showGenError('', 'params'); // Clear error message and show form
                 if(errorSection) errorSection.style.display = 'none';
            });
        }
    }
    // --- RESULTS PAGE LOGIC (generate_results.html) ---
    else if (window.location.pathname.endsWith('/generate/results')) {
        const resultsSectionEl = document.getElementById('resultsSection');
        const generationStatsEl = document.getElementById('generationStats');
        const sequenceResultsTableBodyEl = document.getElementById('sequenceResultsTableBody');
        const predictGeneratedBtn = document.getElementById('predictGeneratedBtn');
        const exportGeneratedBtn = document.getElementById('exportGeneratedBtn');
        const copyGeneratedFastaBtn = document.getElementById('copyGeneratedFastaBtn');
        const predictionForGeneratedResultsEl = document.getElementById('predictionForGeneratedResults');
        const predictionForGeneratedContentEl = document.getElementById('predictionForGeneratedContent');
        
        let currentGeneratedSequences = [];

        function displayPredictionForGenerated(predictionData) { // Moved into this scope
            if (!predictionForGeneratedContentEl || !predictionForGeneratedResultsEl) return;
            let contentHtml = '<div class="table-responsive"><table class="table table-sm table-striped mt-2">';
            contentHtml += '<thead><tr><th>ID</th><th>Sequence (Start)</th><th>Prediction</th><th>Probability</th></tr></thead><tbody>';
            predictionData.results.forEach(item => {
                contentHtml += `<tr>
                    <td><small>${item.id}</small></td>
                    <td><small><code>${item.sequence.substring(0,15)}...</code></small></td>
                    <td><span class="badge ${item.prediction === 1 ? 'bg-success' : 'bg-secondary'}">${item.label}</span></td>
                    <td><small>${item.probability !== -1.0 ? item.probability.toFixed(3) : 'N/A'}</small></td>
                </tr>`;
            });
            contentHtml += '</tbody></table></div>';
            const stats = predictionData.stats;
            let statsHtml = `<p class="mb-1 small text-center">Prediction for Generated - Total: ${stats.total}, Positive: ${stats.positive} (${stats.positive_percentage}%)</p>`;
            predictionForGeneratedContentEl.innerHTML = statsHtml + contentHtml;
            predictionForGeneratedResultsEl.style.display = 'block';
            predictionForGeneratedResultsEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }

        function displayGenerationResults(result) {
            showGenLoading(false, 'results');
            if (!resultsSectionEl || !result || !result.sequences) {
                showGenError('Invalid result data received for generation.', 'results');
                return;
            }
            resultsSectionEl.style.display = 'block';
            currentGeneratedSequences = result.sequences;

            if (generationStatsEl && result.parameters) {
                const avgLength = result.sequences.length > 0 ?
                    Math.round(result.sequences.reduce((sum, seq) => sum + (seq.length || 0), 0) / result.sequences.length) : 0;
                generationStatsEl.innerHTML = `
                    <div class="row text-center">
                        <div class="col-md-3 col-6 mb-2"><small class="text-muted d-block">Generated</small><span class="badge bg-primary fs-6">${result.sequences.length}</span></div>
                        <div class="col-md-3 col-6 mb-2"><small class="text-muted d-block">Method</small><span class="badge bg-info fs-6 text-truncate" title="${result.parameters.sampling_method}">${result.parameters.sampling_method}</span></div>
                        <div class="col-md-3 col-6 mb-2"><small class="text-muted d-block">Avg. Length</small><span class="badge bg-success fs-6">${avgLength}</span></div>
                        <div class="col-md-3 col-6 mb-2"><small class="text-muted d-block">Temp.</small><span class="badge bg-warning fs-6">${result.parameters.temperature}</span></div>
                    </div>
                    ${result.model_info && result.model_info.model_name ? `<p class="text-center mt-2 mb-0 text-muted small">Model: ${result.model_info.model_name}</p>` : ''}`;
            }

            if (sequenceResultsTableBodyEl) {
                sequenceResultsTableBodyEl.innerHTML = '';
                result.sequences.forEach(seq => {
                    const row = sequenceResultsTableBodyEl.insertRow();
                    row.insertCell().textContent = seq.id || 'N/A';
                    const seqCell = row.insertCell();
                    const codeEl = document.createElement('code');
                    codeEl.classList.add('text-break'); codeEl.textContent = seq.sequence;
                    seqCell.appendChild(codeEl);
                    row.insertCell().innerHTML = `<span class="badge bg-secondary">${seq.length || 'N/A'}</span>`;
                    row.insertCell().textContent = seq.method || (result.parameters ? result.parameters.sampling_method : 'N/A');
                });
            }
        }

        showGenLoading(true, 'results');
        const storedResult = localStorage.getItem(GENERATION_RESULT_STORAGE_KEY);
        if (storedResult) {
            try {
                const data = JSON.parse(storedResult);
                displayGenerationResults(data);
            } catch (e) {
                console.error("Error parsing stored generation results:", e);
                showGenError("Could not display stored generation results.", 'results');
            }
        } else {
            showGenError("No generation results found. Please generate sequences first.", 'results');
        }

        if (predictGeneratedBtn) {
            const originalPredictBtnHTML = predictGeneratedBtn.innerHTML;
            predictGeneratedBtn.addEventListener('click', async () => {
                if (!currentGeneratedSequences || currentGeneratedSequences.length === 0) {
                    showToast('No sequences to predict.', 'warning', 'Prediction'); return;
                }
                predictGeneratedBtn.disabled = true;
                predictGeneratedBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Predicting...';
                let fastaText = '';
                currentGeneratedSequences.forEach(seq => { fastaText += `>${seq.id || 'GeneratedSeq'}\n${seq.sequence}\n`; });
                try {
                    const response = await fetch('/predict', {
                        method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: `fasta_text=${encodeURIComponent(fastaText)}`
                    });
                    const predictionResult = await response.json();
                    if (predictionResult.success) {
                        showToast('Activity prediction successful!', 'success', 'Prediction');
                        displayPredictionForGenerated(predictionResult);
                    } else {
                        showToast(predictionResult.error || 'Unknown prediction error', 'danger', 'Prediction Error');
                    }
                } catch (error) {
                    showToast('Prediction request failed: ' + error.message, 'danger', 'Prediction Error');
                    console.error('Prediction request error:', error);
                } finally {
                    predictGeneratedBtn.disabled = false;
                    predictGeneratedBtn.innerHTML = originalPredictBtnHTML;
                }
            });
        }

        if (exportGeneratedBtn) {
            exportGeneratedBtn.addEventListener('click', () => {
                if (!currentGeneratedSequences || currentGeneratedSequences.length === 0) {
                    showToast('No sequences to export.', 'warning', 'Export'); return;
                }
                try {
                    let csvContent = "data:text/csv;charset=utf-8,ID,Sequence,Length,Method\n";
                    currentGeneratedSequences.forEach(row => { csvContent += `${row.id || 'N/A'},${row.sequence || ''},${row.length || 'N/A'},${row.method || 'N/A'}\n`; });
                    const encodedUri = encodeURI(csvContent);
                    const link = document.createElement("a");
                    link.setAttribute("href", encodedUri); link.setAttribute("download", "generated_sequences.csv");
                    document.body.appendChild(link); link.click(); document.body.removeChild(link);
                    showToast('Generated sequences exported as CSV.', 'success', 'Export');
                } catch (e) {
                    showToast('Failed to export CSV.', 'danger', 'Export Error'); console.error("CSV Export error:", e);
                }
            });
        }

        if (copyGeneratedFastaBtn) {
            copyGeneratedFastaBtn.addEventListener('click', () => {
                if (!currentGeneratedSequences || currentGeneratedSequences.length === 0) {
                    showToast('No sequences to copy.', 'warning', 'Copy FASTA'); return;
                }
                let fastaText = '';
                currentGeneratedSequences.forEach(seq => { fastaText += `>${seq.id || 'GeneratedSeq'}\n${seq.sequence}\n`; });
                navigator.clipboard.writeText(fastaText).then(() => {
                    showToast('FASTA sequences copied to clipboard!', 'success', 'Copy FASTA');
                }).catch(err => {
                    showToast('Failed to copy FASTA sequences. Check browser permissions.', 'danger', 'Copy FASTA Error');
                    console.error('Copy FASTA error:', err);
                });
            });
        }
    }
});