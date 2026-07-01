/**
 * RecycleAI - Client-side Interactive Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // Elements Selection
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeImageBtn = document.getElementById('remove-image-btn');
    const classifyBtn = document.getElementById('classify-btn');
    
    // Result panels
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultLoading = document.getElementById('result-loading');
    const resultContent = document.getElementById('result-content');
    
    // Result details
    const resultConfidenceText = document.getElementById('result-confidence');
    const confidenceRing = document.getElementById('confidence-ring');
    const resultClassName = document.getElementById('result-class-name');
    const resultDescription = document.getElementById('result-description');
    const resultRecycleTip = document.getElementById('result-recycle-tip');
    const recycleBadge = document.getElementById('recycle-badge');
    
    // Status
    const modelStatusIndicator = document.getElementById('model-status-indicator');
    const modelStatusText = document.getElementById('model-status-text');
    
    // SVG Circle Setup (Radius = 50)
    const radius = 50;
    const circumference = 2 * Math.PI * radius;
    
    // Initialize progress ring
    confidenceRing.style.strokeDasharray = circumference;
    confidenceRing.style.strokeDashoffset = circumference;
    
    // Current file storage
    let selectedFile = null;

    // Check backend connection and model info
    checkModelStatus();

    function checkModelStatus() {
        fetch('/info')
            .then(res => res.json())
            .then(data => {
                if (data.model_loaded) {
                    modelStatusIndicator.className = 'status-indicator online';
                    modelStatusText.textContent = 'Model Aktif';
                    console.log('[RecycleAI] Model loaded. Classes:', data.classes);
                    
                    // Update footer list of categories dynamically
                    updateFooterBadges(data.classes);
                } else {
                    modelStatusIndicator.className = 'status-indicator offline';
                    modelStatusText.textContent = '(Model Tidak Terdeteksi)';
                    console.warn('[RecycleAI] Keras model not loaded. Running in simulation mode.');
                }
            })
            .catch(err => {
                console.error('[RecycleAI] Error connecting to server:', err);
                modelStatusIndicator.className = 'status-indicator offline';
                modelStatusText.textContent = 'Server Offline';
            });
    }

    function updateFooterBadges(classesList) {
        const footerLinks = document.querySelector('.footer-links');
        if (footerLinks && classesList && classesList.length > 0) {
            // Rebuild footer elements
            footerLinks.innerHTML = '<span>Kategori Terdaftar:</span>';
            classesList.forEach(cls => {
                const badge = document.createElement('span');
                badge.className = 'footer-badge';
                badge.textContent = cls;
                footerLinks.appendChild(badge);
            });
        }
    }

    // Set Progress Ring level
    function setProgress(percent) {
        const offset = circumference - (percent * circumference);
        confidenceRing.style.strokeDashoffset = offset;
    }

    // Upload zone drag-and-drop event listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // File input click listener is handled natively by absolute position over dropZone
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Process selected file
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('File harus berupa gambar (JPG, JPEG, PNG)!');
            return;
        }
        
        selectedFile = file;
        
        // Show Image Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            
            // Toggle visibility classes
            uploadPrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            
            // Enable Button
            classifyBtn.classList.remove('btn-disabled');
            classifyBtn.removeAttribute('disabled');
            
            // Reset results view to placeholder
            resultContent.classList.add('hidden');
            resultPlaceholder.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    // Remove file listener
    removeImageBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '#';
        
        // Toggle view visibility
        previewContainer.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        
        // Disable classify button
        classifyBtn.classList.add('btn-disabled');
        classifyBtn.setAttribute('disabled', 'true');
        
        // Reset Result
        resultContent.classList.add('hidden');
        resultLoading.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
    }

    // Prediction trigger
    classifyBtn.addEventListener('click', () => {
        if (!selectedFile) return;

        // Visual transitions
        resultPlaceholder.classList.add('hidden');
        resultContent.classList.add('hidden');
        resultLoading.classList.remove('hidden');
        
        // Disable buttons temporarily
        classifyBtn.classList.add('btn-disabled');
        classifyBtn.setAttribute('disabled', 'true');
        removeImageBtn.style.pointerEvents = 'none';

        // Prepare image data
        const formData = new FormData();
        formData.append('image', selectedFile);

        // Fetch predict endpoint
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('API server returned an error');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert(data.error);
                showErrorState(data.error);
                return;
            }

            // Display Results
            const confPercent = Math.round(data.confidence * 100);
            
            resultConfidenceText.textContent = `${confPercent}%`;
            setProgress(data.confidence);
            
            resultClassName.textContent = data.class_name;
            resultDescription.textContent = data.description;
            resultRecycleTip.textContent = data.recycle_tip;

            // Set recycling status badge
            recycleBadge.textContent = data.recyclable_text;
            recycleBadge.className = 'badge'; // Reset styling classes
            
            // Map badge color theme
            if (data.class_name.toLowerCase().includes('organik') || data.class_name.toLowerCase().includes('biological')) {
                recycleBadge.classList.add('badge-compostable');
            } else if (data.recyclable) {
                recycleBadge.classList.add('badge-recyclable');
            } else {
                recycleBadge.classList.add('badge-non-recyclable');
            }

            // Transition animations
            resultLoading.classList.add('hidden');
            resultContent.classList.remove('hidden');
        })
        .catch(error => {
            console.error('[RecycleAI] Error making prediction request:', error);
            showErrorState('Koneksi server terputus atau terjadi gangguan pemrosesan data model.');
        })
        .finally(() => {
            // Re-enable actions
            classifyBtn.classList.remove('btn-disabled');
            classifyBtn.removeAttribute('disabled');
            removeImageBtn.style.pointerEvents = 'auto';
        });
    });

    function showErrorState(message) {
        resultLoading.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
        alert(`Gagal menganalisis gambar: ${message}`);
    }
});
