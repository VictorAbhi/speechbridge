let currentEventSource = null;
let isShowingSummary = false;

document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('audioFile');
    const fileUploadContent = document.getElementById('fileUploadContent');
    const selectedFileDiv = document.getElementById('selectedFile');
    const fileNameSpan = document.getElementById('fileName');
    const form = document.getElementById('transcriptionForm');

    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                fileNameSpan.textContent = file.name;
                fileUploadContent.style.display = 'none';
                selectedFileDiv.style.display = 'block';
            } else {
                fileUploadContent.style.display = 'block';
                selectedFileDiv.style.display = 'none';
            }
        });
    }

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            if (fileInput.files.length === 0) {
                alert('Please select an audio file first.');
                return;
            }
            startTranscription();
        });
    }

    // Drag and drop functionality
    const uploadArea = document.querySelector('.file-upload-area');
    
    if (uploadArea) {
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#3b82f6';
            uploadArea.style.background = '#eff6ff';
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#e5e7eb';
            uploadArea.style.background = '#f9fafb';
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#e5e7eb';
            uploadArea.style.background = '#f9fafb';
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                const allowedTypes = ['audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/flac', 'audio/ogg', 'audio/aac', 'audio/x-ms-wma'];
                if (allowedTypes.some(type => file.type.includes(type.split('/')[1]) || file.name.toLowerCase().includes(type.split('/')[1]))) {
                    fileInput.files = files;
                    fileInput.dispatchEvent(new Event('change'));
                } else {
                    alert('Please select a valid audio file.');
                }
            }
        });
    }
});

function startTranscription() {
    const fileInput = document.getElementById('audioFile');
    const languageInput = document.querySelector('input[name="language"]');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('language', languageInput.value);

    // Show progress container
    showProgress();

    // Upload file and start transcription
    fetch('/transcribe', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
        } else {
            // Start listening for progress updates
            listenForProgress(data.session_id);
        }
    })
    .catch(error => {
        showError('Upload failed: ' + error.message);
    });
}

function listenForProgress(sessionId) {
    // Close existing connection if any
    if (currentEventSource) {
        currentEventSource.close();
    }

    currentEventSource = new EventSource(`/progress/${sessionId}`);
    
    currentEventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateProgress(data);
    };

    currentEventSource.onerror = function(event) {
        console.error('EventSource failed:', event);
        currentEventSource.close();
        showError('Connection lost. Please try again.');
    };
}

function updateProgress(data) {
    const { stage, progress, message, details, result, partial_text } = data;

    // Update progress bar
    document.getElementById('progressFill').style.width = progress + '%';
    
    // Always remove percentage from main progress text message
    let displayMessage = message.replace(/\s*\(\d+%\)/, '');
    document.getElementById('progressText').textContent = displayMessage;
    document.getElementById('progressDetails').textContent = details || '';

    // Always show percentage in the bottom segment
    document.getElementById('progressPercentage').textContent = progress + '%';

    // Update live transcription if available
    if (partial_text) {
        showLiveTranscription(partial_text);
    }

    // Update stage indicators
    const stages = ['upload', 'analyze', 'process', 'transcribe', 'summarize'];
    stages.forEach(stageName => {
        const stageElement = document.getElementById(`stage-${stageName}`);
        stageElement.classList.remove('active', 'completed', 'pending');
        
        if (stageName === stage) {
            stageElement.classList.add('active');
        } else if (stages.indexOf(stageName) < stages.indexOf(stage)) {
            stageElement.classList.add('completed');
        } else {
            stageElement.classList.add('pending');
        }
    });

    // Handle completion
    if (stage === 'complete' && result) {
        setTimeout(() => {
            showResult(result);
            if (currentEventSource) {
                currentEventSource.close();
            }
        }, 1000);
    } else if (stage === 'error') {
        showError(details || message);
        if (currentEventSource) {
            currentEventSource.close();
        }
    }
}

function showLiveTranscription(text) {
    const container = document.getElementById('liveTranscriptionContainer');
    const textElement = document.getElementById('liveTranscriptionText');
    const typingIndicator = document.getElementById('typingIndicator');
    
    // Show container if hidden
    if (container.style.display === 'none') {
        container.style.display = 'block';
    }
    
    // Update text with typing effect
    textElement.innerHTML = text;
    
    // Show typing indicator briefly
    typingIndicator.style.display = 'inline';
    setTimeout(() => {
        typingIndicator.style.display = 'none';
    }, 1000);
    
    // Auto-scroll to bottom
    textElement.scrollTop = textElement.scrollHeight;
}

function showProgress() {
    document.getElementById('uploadForm').style.display = 'none';
    document.getElementById('progressContainer').style.display = 'block';
    document.getElementById('liveTranscriptionContainer').style.display = 'none';
    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'none';
}

function showResult(result) {
    document.getElementById('progressContainer').style.display = 'none';
    document.getElementById('liveTranscriptionContainer').style.display = 'none';
    document.getElementById('resultContainer').style.display = 'block';
    document.getElementById('errorContainer').style.display = 'none';

    // Populate result data
    document.getElementById('resultFilename').textContent = `(${result.filename})`;
    document.getElementById('transcriptionText').textContent = result.transcription;
    document.getElementById('summaryText').textContent = result.summary;

    // Populate info
    const infoHtml = `
        <span><strong>Duration:</strong> ${result.duration}</span>
        <span><strong>Size:</strong> ${result.file_size}</span>
        <span><strong>Method:</strong> ${result.method}</span>
        <span><strong>Quality:</strong> ${result.audio_quality}</span>
    `;
    document.getElementById('resultInfo').innerHTML = infoHtml;
}

function showError(message) {
    document.getElementById('uploadForm').style.display = 'none';
    document.getElementById('progressContainer').style.display = 'none';
    document.getElementById('liveTranscriptionContainer').style.display = 'none';
    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'block';
    document.getElementById('errorText').textContent = message;
}

function resetForm() {
    // Close event source if active
    if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
    }

    // Reset form
    document.getElementById('audioFile').value = '';
    document.getElementById('fileUploadContent').style.display = 'block';
    document.getElementById('selectedFile').style.display = 'none';
    
    // Show upload form
    document.getElementById('uploadForm').style.display = 'block';
    document.getElementById('progressContainer').style.display = 'none';
    document.getElementById('liveTranscriptionContainer').style.display = 'none';
    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('errorContainer').style.display = 'none';
    
    // Reset progress
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressPercentage').textContent = '0%';
    
    // Reset stages
    const stages = ['upload', 'analyze', 'process', 'transcribe', 'summarize'];
    stages.forEach(stageName => {
        const stageElement = document.getElementById(`stage-${stageName}`);
        stageElement.classList.remove('active', 'completed');
        stageElement.classList.add('pending');
    });

    isShowingSummary = false;
}

function copyTranscription() {
    const textToCopy = isShowingSummary ? 
        document.getElementById('summaryText').textContent : 
        document.getElementById('transcriptionText').textContent;
    
    navigator.clipboard.writeText(textToCopy).then(function() {
        const btn = event.target;
        const originalText = btn.innerHTML;
        btn.innerHTML = 'Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
        alert('Failed to copy text to clipboard');
    });
}

function toggleSummary() {
    const transcriptionText = document.getElementById('transcriptionText');
    const summaryText = document.getElementById('summaryText');
    const btn = event.target;
    
    if (summaryText.style.display === 'none') {
        transcriptionText.style.display = 'none';
        summaryText.style.display = 'block';
        btn.innerHTML = 'Show Full Text';
        isShowingSummary = true;
    } else {
        transcriptionText.style.display = 'block';
        summaryText.style.display = 'none';
        btn.innerHTML = 'Show Summary';
        isShowingSummary = false;
    }
}