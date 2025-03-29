document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const captureButton = document.getElementById('capture_button');
    const webcamContainer = document.getElementById('webcam_container');
    const webcam = document.getElementById('webcam');
    const takePhotoButton = document.getElementById('take_photo');
    const cancelPhotoButton = document.getElementById('cancel_photo');
    const canvas = document.getElementById('canvas');
    const previewContainer = document.getElementById('preview_container');
    const photoPreview = document.getElementById('photo_preview');
    const usePhotoButton = document.getElementById('use_photo');
    const retakePhotoButton = document.getElementById('retake_photo');
    const fileInput = document.getElementById('face_image');
    
    let stream = null;
    
    // Start webcam
    function startWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            webcamContainer.classList.remove('d-none');
            fileInput.classList.add('d-none');
            
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(mediaStream) {
                    stream = mediaStream;
                    webcam.srcObject = stream;
                    webcam.play();
                })
                .catch(function(error) {
                    console.error('Error accessing webcam:', error);
                    alert('Could not access the webcam. Please ensure you have given permission to use the camera and try again.');
                    stopWebcam();
                });
        } else {
            alert('Your browser does not support webcam access. Please try using a different browser or upload an image instead.');
        }
    }
    
    // Stop webcam
    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(function(track) {
                track.stop();
            });
            stream = null;
        }
        
        webcamContainer.classList.add('d-none');
        fileInput.classList.remove('d-none');
    }
    
    // Take photo from webcam
    function takePhoto() {
        const context = canvas.getContext('2d');
        
        // Set canvas dimensions to match webcam
        canvas.width = webcam.videoWidth;
        canvas.height = webcam.videoHeight;
        
        // Draw the current webcam frame to the canvas
        context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to data URL
        const dataURL = canvas.toDataURL('image/jpeg');
        
        // Show the preview
        photoPreview.src = dataURL;
        webcamContainer.classList.add('d-none');
        previewContainer.classList.remove('d-none');
    }
    
    // Use the captured photo
    function usePhoto() {
        // Convert data URL to Blob
        const dataURL = photoPreview.src;
        const byteString = atob(dataURL.split(',')[1]);
        const mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        
        const blob = new Blob([ab], { type: mimeString });
        
        // Create a File object
        const fileName = `webcam_capture_${new Date().getTime()}.jpg`;
        const file = new File([blob], fileName, { type: 'image/jpeg' });
        
        // Create a new FileList
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        
        // Set the file input's files
        fileInput.files = dataTransfer.files;
        
        // Clean up
        previewContainer.classList.add('d-none');
        fileInput.classList.remove('d-none');
        stopWebcam();
        
        // Optionally, for forms that don't have a separate submit button
        // and rely on the file input change event, trigger it
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
    
    // Event listeners
    if (captureButton) {
        captureButton.addEventListener('click', startWebcam);
    }
    
    if (takePhotoButton) {
        takePhotoButton.addEventListener('click', takePhoto);
    }
    
    if (cancelPhotoButton) {
        cancelPhotoButton.addEventListener('click', stopWebcam);
    }
    
    if (usePhotoButton) {
        usePhotoButton.addEventListener('click', usePhoto);
    }
    
    if (retakePhotoButton) {
        retakePhotoButton.addEventListener('click', function() {
            previewContainer.classList.add('d-none');
            webcamContainer.classList.remove('d-none');
        });
    }
    
    // Clean up when leaving the page
    window.addEventListener('beforeunload', stopWebcam);
});
