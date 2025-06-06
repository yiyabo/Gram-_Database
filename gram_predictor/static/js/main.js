document.addEventListener('DOMContentLoaded', () => {
    if (typeof tsParticles !== 'undefined') {
        tsParticles.load("tsparticles-background", {
            fpsLimit: 60,
            background: {
                color: "transparent" // Body background is set in CSS
            },
            particles: {
                number: {
                    value: 100, // Restored value
                    density: {
                        enable: true,
                        area: 800 // Restored value
                    }
                },
                color: {
                    value: ["#333333", "#666666", "#999999", "#00E5FF", "#FFFFFF"] // Restored: Greys, accent-cyan, white
                },
                shape: {
                    type: "square", // Restored shape
                },
                opacity: {
                    value: { min: 0.1, max: 0.6 }, // Random opacity
                    animation: {
                        enable: true,
                        speed: 0.5, // Restored speed
                        minimumValue: 0.1, // Restored min value
                        sync: false
                    }
                },
                size: {
                    value: { min: 1, max: 4 }, // Restored size
                    animation: {
                        enable: true,
                        speed: 2, // Restored speed
                        minimumValue: 0.5, // Restored min value
                        sync: false
                    }
                },
                links: {
                    enable: true,
                    distance: 120, // Restored distance
                    color: "#444444", // Restored link color (can be var(--glass-border-color) later)
                    opacity: 0.3, // Restored opacity
                    width: 1 // Restored width
                },
                move: {
                    enable: true,
                    speed: 0.8, // Restored speed
                    direction: "none",
                    random: true,
                    straight: false,
                    outModes: {
                        default: "out"
                    },
                    attract: {
                        enable: false // Kept false as per original restored config
                    }
                    // Removed wobble and noise from Nebula Haze
                }
            },
            interactivity: {
                detectsOn: "canvas",
                events: {
                    onHover: {
                        enable: true,
                        mode: "repulse" // Restored mode
                    },
                    onClick: {
                        enable: true, // Restored enable state
                        mode: "push" // Restored mode
                    },
                    resize: true
                },
                modes: {
                    grab: {
                        distance: 150,
                        links: {
                            opacity: 0.5
                        }
                    },
                    bubble: {
                        distance: 200,
                        size: 10,
                        duration: 2,
                        opacity: 0.8
                    },
                    repulse: {
                        distance: 80, // Repulse distance
                        duration: 0.4
                    },
                    push: {
                        quantity: 3 // Number of particles to push on click
                    },
                    remove: {
                        quantity: 2
                    }
                }
            },
            detectRetina: true // For high-density displays
        }).then(container => {
            console.log("tsParticles loaded successfully");
            // You can further interact with the container here if needed
            // For example, to simulate rotation, you might try to slowly update
            // a global angle and re-calculate particle positions if using custom paths,
            // or adjust some global movement parameters if the library allows.
            // tsParticles itself doesn't have a simple "rotate entire scene" for 2D canvas.
            // The "orbit" feature is per-particle.
        }).catch(error => {
            console.error("Error loading tsParticles:", error);
        });
    } else {
        console.error("tsParticles library not found.");
    }

    // Prediction page specific JavaScript
    const fileForm = document.getElementById('fileForm');
    const textForm = document.getElementById('textForm');
    const fastaFileEl = document.getElementById('fastaFile');
    const fastaTextEl = document.getElementById('fastaText');
    const dropZone = document.getElementById('dropZone');
    const selectedFileNameEl = document.querySelector('.selected-file-name');
    
    const loadingSection = document.getElementById('loadingSection');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const errorMessageEl = document.getElementById('errorMessage');
    
    const totalSequencesEl = document.getElementById('totalSequences');
    const positiveCountEl = document.getElementById('positiveCount');
    const positivePercentageEl = document.getElementById('positivePercentage');
    const resultsTableBodyEl = document.getElementById('resultsTableBody');
    const exportCSVBtn = document.getElementById('exportCSV');
    const exportFASTABtn = document.getElementById('exportFASTA');
    const loadExampleBtn = document.getElementById('loadExample');
    const resetFormsBtn = document.getElementById('resetForms');

    let resultChart = null;
    let currentResultsData = null;

    function showLoading() {
        loadingSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        errorSection.classList.add('d-none');
    }

    function showResults(data) {
        loadingSection.classList.add('d-none');
        resultsSection.classList.remove('d-none');
        errorSection.classList.add('d-none');
        
        currentResultsData = data.results; // Store for export

        totalSequencesEl.textContent = data.stats.total;
        positiveCountEl.textContent = data.stats.positive;
        positivePercentageEl.textContent = `${data.stats.positive_percentage}%`;

        resultsTableBodyEl.innerHTML = ''; // Clear previous results
        const SEQUENCE_TRUNCATE_LENGTH = 30;

        data.results.forEach(result => {
            const row = resultsTableBodyEl.insertRow();
            
            // Sequence ID
            row.insertCell().textContent = result.id;
            
            // Sequence (with View More/Less)
            const sequenceCell = row.insertCell();
            const sequenceWrapper = document.createElement('div');
            sequenceWrapper.classList.add('sequence-wrapper');
            
            const sequenceContent = document.createElement('span');
            sequenceContent.classList.add('sequence-content');
            
            const fullSequence = result.sequence;
            let isExpanded = false;

            const updateSequenceDisplay = () => {
                if (isExpanded) {
                    sequenceContent.innerHTML = ''; // Clear previous
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

            updateSequenceDisplay(); // Initial display
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

            // Probability
            const probabilityCell = row.insertCell();
            probabilityCell.textContent = result.probability !== -1.0 ? result.probability.toFixed(4) : 'N/A';
            
            // Prediction Label
            const predictionCell = row.insertCell();
            const badge = document.createElement('span');
            badge.classList.add('badge');
            if (result.prediction === 1) {
                badge.classList.add('bg-success');
                badge.textContent = 'Anti-Gram-Negative';
            } else if (result.prediction === 0) {
                badge.classList.add('bg-secondary');
                badge.textContent = 'Non-Anti-Gram-Negative';
            } else {
                badge.classList.add('bg-danger');
                badge.textContent = 'Error';
            }
            predictionCell.appendChild(badge);

            // Actions Cell (for Copy button)
            const actionsCell = row.insertCell();
            actionsCell.classList.add('actions-cell'); // For potential specific styling
            const copyBtn = document.createElement('button');
            copyBtn.classList.add('btn', 'btn-sm', 'copy-sequence-btn');
            copyBtn.title = 'Copy Sequence';
            copyBtn.innerHTML = '<i class="bi bi-clipboard"></i>';
            copyBtn.addEventListener('click', async () => {
                try {
                    await navigator.clipboard.writeText(fullSequence);
                    copyBtn.innerHTML = '<i class="bi bi-check-lg"></i>'; // Success icon
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="bi bi-clipboard"></i>'; // Revert icon
                    }, 1500);
                } catch (err) {
                    console.error('Failed to copy sequence: ', err);
                    // Optionally show an error to the user, e.g., using a toast notification
                    copyBtn.innerHTML = '<i class="bi bi-x-lg"></i>'; // Error icon
                     setTimeout(() => {
                        copyBtn.innerHTML = '<i class="bi bi-clipboard"></i>'; // Revert icon
                    }, 1500);
                }
            });
            actionsCell.appendChild(copyBtn);
        });

        renderChart(data.stats);
    }

    function showError(message) {
        loadingSection.classList.add('d-none');
        resultsSection.classList.add('d-none');
        errorSection.classList.remove('d-none');
        errorMessageEl.textContent = message;
    }
    
    function resetUI() {
        loadingSection.classList.add('d-none');
        resultsSection.classList.add('d-none');
        errorSection.classList.add('d-none');
        if (fastaFileEl) fastaFileEl.value = '';
        if (fastaTextEl) fastaTextEl.value = '';
        if (selectedFileNameEl) selectedFileNameEl.textContent = '';
        if (dropZone) dropZone.classList.remove('file-dropped');
        currentResultsData = null;
    }

    async function handlePrediction(formData) {
        showLoading();
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (response.ok && data.success) {
                showResults(data);
            } else {
                showError(data.error || 'An unknown error occurred.');
            }
        } catch (error) {
            console.error('Prediction error:', error);
            showError('Failed to connect to the server. Please try again.');
        }
    }

    if (fileForm) {
        fileForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const formData = new FormData();
            if (fastaFileEl.files.length > 0) {
                formData.append('fasta_file', fastaFileEl.files[0]);
                handlePrediction(formData);
            } else {
                showError('Please select a FASTA file.');
            }
        });
    }

    if (textForm) {
        textForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const fastaContent = fastaTextEl.value.trim();
            if (fastaContent) {
                const formData = new FormData();
                formData.append('fasta_text', fastaContent);
                handlePrediction(formData);
            } else {
                showError('Please enter FASTA sequence data.');
            }
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
        const dropZoneH5 = dropZone.querySelector('.file-upload-content h5');
        const dropZoneP = dropZone.querySelector('.file-upload-content p:not(.selected-file-name)');


        dropZone.addEventListener('dragenter', (e) => { // Use dragenter for initial visual cue
            e.preventDefault();
            dropZone.classList.add('drag-over');
            if (dropZoneIcon) dropZoneIcon.classList.add('file-upload-icon-pulse');
            if (dropZoneH5) dropZoneH5.style.fontWeight = '700'; // Example direct style change
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault(); // Necessary to allow drop
            // drag-over class and icon pulse should already be active from dragenter
        });

        dropZone.addEventListener('dragleave', (e) => {
            // Only remove if not dragging over a child element (less flicker)
            if (e.target === dropZone || !dropZone.contains(e.relatedTarget)) {
                dropZone.classList.remove('drag-over');
                if (dropZoneIcon) dropZoneIcon.classList.remove('file-upload-icon-pulse');
                if (dropZoneH5) dropZoneH5.style.fontWeight = ''; // Reset
            }
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (dropZoneIcon) dropZoneIcon.classList.remove('file-upload-icon-pulse');
            if (dropZoneH5) dropZoneH5.style.fontWeight = ''; // Reset

            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.name.endsWith('.fasta') || file.name.endsWith('.fa') || file.name.endsWith('.txt')) {
                    fastaFileEl.files = e.dataTransfer.files; // Assign to the hidden file input
                    if (selectedFileNameEl) selectedFileNameEl.textContent = file.name;
                    dropZone.classList.add('file-dropped'); // Optional: class to indicate a file is selected
                } else {
                    showError('Invalid file type. Please upload a .fasta, .fa, or .txt file.');
                    if (selectedFileNameEl) selectedFileNameEl.textContent = '';
                    dropZone.classList.remove('file-dropped');
                }
            }
        });

        // Allow clicking on dropzone to open file dialog
        dropZone.addEventListener('click', (e) => {
            // Prevent opening file dialog if a button or label inside dropzone is clicked
            if (e.target.closest('button') || e.target.closest('label') || e.target.tagName === 'INPUT') {
                return;
            }
            if (fastaFileEl) fastaFileEl.click();
        });
    }

    if (loadExampleBtn && fastaTextEl) {
        loadExampleBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/example');
                const data = await response.json();
                if (data.fasta) {
                    fastaTextEl.value = data.fasta;
                    // Switch to text input tab if not active
                    const textTabButton = document.getElementById('text-tab');
                    if (textTabButton) {
                        new bootstrap.Tab(textTabButton).show();
                    }
                }
            } catch (error) {
                console.error('Error loading example:', error);
                showError('Failed to load example data.');
            }
        });
    }
    
    if (resetFormsBtn) {
        resetFormsBtn.addEventListener('click', resetUI);
    }

    function renderChart(stats) {
        const ctx = document.getElementById('resultChart').getContext('2d');
        if (resultChart) {
            resultChart.destroy();
        }
        resultChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Predicted Positive', 'Predicted Negative/Failed'],
                datasets: [{
                    label: 'Prediction Distribution',
                    data: [stats.positive, stats.total - stats.positive],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.7)', // Success (green)
                        'rgba(108, 117, 125, 0.7)'  // Secondary (gray)
                    ],
                    borderColor: [
                        'rgba(40, 167, 69, 1)',
                        'rgba(108, 117, 125, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Prediction Outcome Distribution'
                    }
                }
            }
        });
    }

    async function exportData(format) {
        if (!currentResultsData || currentResultsData.length === 0) {
            showError('No results to export.');
            return;
        }
        try {
            const response = await fetch('/export', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    results: currentResultsData,
                    format: format
                })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                
                const contentDisposition = response.headers.get('content-disposition');
                let filename = `prediction_results.${format}`; // Default filename
                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename="?(.+)"?/i);
                    if (filenameMatch && filenameMatch.length > 1) {
                        filename = filenameMatch[1];
                    }
                }
                a.download = filename;
                
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                a.remove();
            } else {
                const errorData = await response.json();
                showError(errorData.error || `Failed to export ${format.toUpperCase()}.`);
            }
        } catch (error) {
            console.error(`Export ${format} error:`, error);
            showError(`An error occurred while exporting ${format.toUpperCase()}.`);
        }
    }

    if (exportCSVBtn) {
        exportCSVBtn.addEventListener('click', () => exportData('csv'));
    }
    if (exportFASTABtn) {
        exportFASTABtn.addEventListener('click', () => exportData('fasta'));
    }
    
    // Initial UI reset
    resetUI();
});
