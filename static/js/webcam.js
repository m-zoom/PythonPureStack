document.addEventListener('DOMContentLoaded', function() {
    // Get elements
    const webcamModal = document.getElementById('webcamModal');
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const capturedImage = document.getElementById('capturedImage');
    const toggleWebcamButton = document.getElementById('toggleWebcam');
    const captureButton = document.getElementById('captureButton');
    const usePhotoButton = document.getElementById('usePhotoButton');
    const webcamButtonText = document.getElementById('webcamButtonText');
    
    // Variables
    let stream = null;
    let webcamActive = false;
    let photoTaken = false;
    
    // Initialize Bootstrap modal
    const modal = new bootstrap.Modal(webcamModal);
    
    // The webcam modal is now opened by Bootstrap data-bs-toggle="modal"
    
    // Toggle webcam
    toggleWebcamButton.addEventListener('click', function() {
        if (webcamActive) {
            stopWebcam();
        } else {
            startWebcam();
        }
    });
    
    // Take photo
    captureButton.addEventListener('click', function() {
        takePhoto();
    });
    
    // Use photo
    usePhotoButton.addEventListener('click', function() {
        usePhoto();
    });
    
    // Function to start webcam
    function startWebcam() {
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then(function(mediaStream) {
                stream = mediaStream;
                video.srcObject = stream;
                video.play();
                webcamActive = true;
                
                // Show video, hide captured image
                video.classList.remove('d-none');
                capturedImage.classList.add('d-none');
                
                // Update button states
                captureButton.disabled = false;
                usePhotoButton.disabled = true;
                toggleWebcamButton.textContent = 'Stop Camera';
                toggleWebcamButton.classList.remove('btn-secondary');
                toggleWebcamButton.classList.add('btn-danger');
                
                photoTaken = false;
            })
            .catch(function(error) {
                console.error('Error accessing webcam:', error);
                alert('Error accessing webcam. Please make sure your camera is enabled and try again.');
            });
    }
    
    // Function to stop webcam
    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        
        webcamActive = false;
        
        // Update button states
        captureButton.disabled = true;
        webcamButtonText.textContent = 'Start Camera';
        toggleWebcamButton.classList.remove('btn-danger');
        toggleWebcamButton.classList.add('btn-secondary');
        
        if (!photoTaken) {
            usePhotoButton.disabled = true;
        }
    }
    
    // Function to take a photo
    function takePhoto() {
        if (!webcamActive) return;
        
        const context = canvas.getContext('2d');
        
        // Draw current video frame to canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to data URL
        const imageDataURL = canvas.toDataURL('image/jpeg');
        
        // Show captured image, hide video
        capturedImage.src = imageDataURL;
        capturedImage.classList.remove('d-none');
        video.classList.add('d-none');
        
        // Update button states
        usePhotoButton.disabled = false;
        captureButton.disabled = true;
        
        photoTaken = true;
        
        // Stop the webcam
        stopWebcam();
    }
    
    // Function to use the captured photo
    function usePhoto() {
        // Convert canvas data to blob
        canvas.toBlob(function(blob) {
            // Create form data
            const formData = new FormData();
            formData.append('image', blob, 'webcam_capture.jpg');
            
            // Upload image
            fetch('/api/webcam_capture', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Close modal
                    modal.hide();
                    
                    // Redirect to search page with the captured image
                    window.location.href = '/search?image=' + encodeURIComponent(data.file_path);
                } else {
                    alert(data.error || 'Error uploading image');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error uploading image. Please try again.');
            });
        }, 'image/jpeg', 0.9);
    }
    
    // Clean up when modal is hidden
    webcamModal.addEventListener('hidden.bs.modal', function() {
        if (webcamActive) {
            stopWebcam();
        }
        photoTaken = false;
        capturedImage.classList.add('d-none');
        video.classList.remove('d-none');
    });
});