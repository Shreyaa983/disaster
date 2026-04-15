// Disaster Report System - Frontend JavaScript

class DisasterReportSystem {
    constructor() {
        this.apiEndpoint = 'http://localhost:5000/predict'; // Backend API endpoint
        this.selectedImage = null;
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        this.form = document.getElementById('reportForm');
        this.textDescription = document.getElementById('textDescription');
        this.imageUpload = document.getElementById('imageUpload');
        this.imageUploadArea = document.getElementById('imageUploadArea');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.location = document.getElementById('location');
        this.emergencyLevel = document.getElementById('emergencyLevel');
        this.submitBtn = document.getElementById('submitBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorMessage = document.getElementById('errorMessage');
        this.newReportBtn = document.getElementById('newReportBtn');
        this.downloadReportBtn = document.getElementById('downloadReportBtn');
        this.modelBreakdown = document.getElementById('modelBreakdown');
        this.textSourcePill = document.getElementById('textSourcePill');
        this.openFormBtn = document.getElementById('openFormBtn');
        this.reportWorkspace = document.getElementById('reportWorkspace');
    }

    attachEventListeners() {
        // Image upload
        this.imageUploadArea.addEventListener('click', () => this.imageUpload.click());
        this.imageUpload.addEventListener('change', (e) => this.handleImageUpload(e));

        // Drag and drop
        this.imageUploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.imageUploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.imageUploadArea.addEventListener('drop', (e) => this.handleDrop(e));

        // Remove image button
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-image')) {
                this.removeImage();
            }
        });

        // Form submission
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));

        // New report button
        this.newReportBtn.addEventListener('click', () => this.resetForm());

        // Download report button
        this.downloadReportBtn.addEventListener('click', () => this.downloadReport());

        if (this.openFormBtn) {
            this.openFormBtn.addEventListener('click', () => this.openReportWorkspace());
        }
    }

    handleImageUpload(e) {
        const file = e.target.files[0];
        if (file) {
            this.processImage(file);
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        this.imageUploadArea.style.borderColor = '#4ecdc4';
        this.imageUploadArea.style.background = 'rgba(78, 205, 196, 0.1)';
    }

    handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        this.imageUploadArea.style.borderColor = '#bdc3c7';
        this.imageUploadArea.style.background = '#f8f9fa';
    }

    handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        this.imageUploadArea.style.borderColor = '#bdc3c7';
        this.imageUploadArea.style.background = '#f8f9fa';

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                this.processImage(file);
            } else {
                this.showError('Please upload a valid image file.');
            }
        }
    }

    processImage(file) {
        // Validate file size (10MB max)
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showError('Image size should not exceed 10MB.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.selectedImage = file;
            this.previewImg.src = e.target.result;
            this.previewImg.style.display = 'block';
            this.imagePreview.style.display = 'block';
            document.querySelector('.upload-placeholder').style.display = 'none';
            this.hideError();
        };
        reader.readAsDataURL(file);
    }

    removeImage() {
        this.selectedImage = null;
        this.previewImg.src = '';
        this.imagePreview.style.display = 'none';
        document.querySelector('.upload-placeholder').style.display = 'block';
        this.imageUpload.value = '';
    }

    async handleSubmit(e) {
        e.preventDefault();
        this.hideError();

        // Validate form
        if (!this.textDescription.value.trim()) {
            this.showError('Please describe the situation.');
            return;
        }

        // Prepare data
        const formData = new FormData();
        formData.append('text', this.textDescription.value.trim());
        formData.append('location', this.location.value.trim() || 'Not provided');
        formData.append('emergency_level', this.emergencyLevel.value || 'Not specified');

        if (this.selectedImage) {
            formData.append('image', this.selectedImage);
        }

        // Submit
        await this.submitReport(formData);
    }

    openReportWorkspace() {
        if (this.reportWorkspace) {
            this.reportWorkspace.classList.remove('is-hidden');
            this.reportWorkspace.classList.add('is-open');
            this.reportWorkspace.style.display = 'grid';
            this.reportWorkspace.setAttribute('aria-hidden', 'false');
            this.reportWorkspace.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        this.textDescription.focus({ preventScroll: true });
    }

    async submitReport(formData) {
        try {
            this.setSubmitButtonLoading(true);

            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const result = await response.json();
            this.displayResults(result);
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Failed to process report: ${error.message}. Make sure the backend server is running at ${this.apiEndpoint}`);
            this.setSubmitButtonLoading(false);
        }
    }

    displayResults(result) {
        // Parse results
        const imagePrediction = result.image_prediction || 'Not available';
        const imageConfidence = result.image_confidence || 0;
        const textKeywords = result.text_keywords || [];
        const textClassification = result.text_classification || 'Unknown';
        const textConfidence = result.text_confidence || 0;
        const textSource = result.text_analysis_source || 'unknown';
        const finalDecision = result.final_decision || 'Unable to determine';
        const priority = result.priority_level || 'Medium';
        const ensemble = result.model_ensemble || {};

        // Display image prediction
        document.getElementById('imagePrediction').textContent = imagePrediction;
        document.getElementById('imageConfidence').textContent = `${(imageConfidence * 100).toFixed(2)}%`;

        // Confidence bar
        const confidenceBar = document.getElementById('imageConfidenceBar');
        const confidenceFill = confidenceBar.querySelector('.confidence-fill');
        confidenceBar.style.display = 'block';
        confidenceFill.style.width = `${imageConfidence * 100}%`;

        // Display text analysis
        this.displayKeywords(textKeywords);
        document.getElementById('textClassification').textContent = textClassification;
        document.getElementById('textConfidence').textContent = `${(textConfidence * 100).toFixed(2)}%`;
        this.updateTextSourcePill(textSource);

        // Model breakdown
        this.displayModelBreakdown(ensemble);

        // Final output
        this.displayFinalOutput(finalDecision, priority);

        // Show results
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Reset submit button
        this.setSubmitButtonLoading(false);
    }

    displayKeywords(keywords) {
        const keywordsList = document.getElementById('detectedKeywords');
        keywordsList.innerHTML = '';

        if (keywords.length === 0) {
            keywordsList.innerHTML = '<span>' + 'No keywords detected' + '</span>';
            return;
        }

        keywords.forEach((keyword) => {
            const tag = document.createElement('span');
            tag.className = 'keyword-tag';
            tag.textContent = keyword;
            keywordsList.appendChild(tag);
        });
    }

    updateTextSourcePill(source) {
        if (!this.textSourcePill) {
            return;
        }

        const normalizedSource = String(source || '').toLowerCase();

        if (normalizedSource === 'tfidf') {
            this.textSourcePill.textContent = 'TF-IDF';
            this.textSourcePill.className = 'result-pill';
            return;
        }

        if (normalizedSource === 'keyword_fallback') {
            this.textSourcePill.textContent = 'Fallback';
            this.textSourcePill.className = 'result-pill neutral';
            return;
        }

        this.textSourcePill.textContent = source || 'Unknown';
        this.textSourcePill.className = 'result-pill neutral';
    }

    displayModelBreakdown(ensemble) {
        if (!this.modelBreakdown) {
            return;
        }

        this.modelBreakdown.innerHTML = '';

        if (ensemble.cnn_prediction) {
            const line = document.createElement('div');
            line.className = 'breakdown-line';

            const name = document.createElement('span');
            name.textContent = 'CNN';

            const value = document.createElement('span');
            value.textContent = `${ensemble.cnn_prediction} · ${(Number(ensemble.cnn_confidence || 0) * 100).toFixed(2)}%`;

            line.appendChild(name);
            line.appendChild(value);
            this.modelBreakdown.appendChild(line);
        }

        if (ensemble.resnet_prediction) {
            const line = document.createElement('div');
            line.className = 'breakdown-line';

            const name = document.createElement('span');
            name.textContent = 'ResNet18';

            const value = document.createElement('span');
            value.textContent = `${ensemble.resnet_prediction} · ${(Number(ensemble.resnet_confidence || 0) * 100).toFixed(2)}%`;

            line.appendChild(name);
            line.appendChild(value);
            this.modelBreakdown.appendChild(line);
        }
    }

    displayFinalOutput(decision, priority) {
        const finalOutput = document.getElementById('finalOutput');
        const priorityBadge = document.getElementById('priorityLevel');

        const iconClass = priority === 'High'
            ? 'fa-solid fa-triangle-exclamation'
            : priority === 'Medium'
                ? 'fa-solid fa-circle-info'
                : 'fa-solid fa-circle-check';

        finalOutput.className = `final-output ${priority.toLowerCase()}`;
        finalOutput.innerHTML = '';

        const icon = document.createElement('span');
        icon.className = 'final-output-icon';

        const iconNode = document.createElement('i');
        iconNode.className = iconClass;
        icon.appendChild(iconNode);

        const text = document.createElement('span');
        text.textContent = decision;

        finalOutput.appendChild(icon);
        finalOutput.appendChild(text);

        // Set priority badge
        priorityBadge.textContent = priority;
        priorityBadge.className = `priority-badge ${priority.toLowerCase()}`;
    }

    resetForm() {
        this.form.reset();
        this.removeImage();
        this.resultsSection.style.display = 'none';
        this.hideError();
        this.openReportWorkspace();
        this.textDescription.focus();
    }

    downloadReport() {
        // Collect data
        const reportData = {
            timestamp: new Date().toISOString(),
            description: this.textDescription.value,
            location: this.location.value || 'Not provided',
            emergencyLevel: this.emergencyLevel.value || 'Not specified',
            imageResults: {
                prediction: document.getElementById('imagePrediction').textContent,
                confidence: document.getElementById('imageConfidence').textContent,
            },
            textResults: {
                keywords: Array.from(document.querySelectorAll('.keyword-tag')).map(tag => tag.textContent),
                classification: document.getElementById('textClassification').textContent,
                source: this.textSourcePill ? this.textSourcePill.textContent : 'Unknown',
            },
            finalDecision: document.getElementById('finalOutput').textContent,
            priorityLevel: document.getElementById('priorityLevel').textContent,
        };

        // Convert to JSON and download
        const dataStr = JSON.stringify(reportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `disaster_report_${new Date().getTime()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    setSubmitButtonLoading(isLoading) {
        if (isLoading) {
            this.submitBtn.disabled = true;
            this.submitBtn.querySelector('span:first-child').style.display = 'none';
            this.submitBtn.querySelector('.spinner').style.display = 'inline';
        } else {
            this.submitBtn.disabled = false;
            this.submitBtn.querySelector('span:first-child').style.display = 'inline';
            this.submitBtn.querySelector('.spinner').style.display = 'none';
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.style.display = 'block';
    }

    hideError() {
        this.errorMessage.style.display = 'none';
        this.errorMessage.textContent = '';
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    new DisasterReportSystem();
});
