document.addEventListener('DOMContentLoaded', function() {
    const cameraFeed = document.getElementById('camera-feed');
    const startCameraBtn = document.getElementById('startCamera');
    const stopCameraBtn = document.getElementById('stopCamera');
    const toggleAudioBtn = document.getElementById('toggleAudio');
    const gestureText = document.getElementById('gesture-text');
    const gestureTypeValue = document.getElementById('gesture-type-value');
    const leftHandIndicator = document.querySelector('.left-hand');
    const rightHandIndicator = document.querySelector('.right-hand');
    
    let isAudioEnabled = false;
    let audioQueue = [];
    let isProcessing = false;
    
    // Initialize camera
    startCameraBtn.addEventListener('click', function() {
        cameraFeed.src = '/video_feed';
        startCameraBtn.disabled = true;
        stopCameraBtn.disabled = false;
    });
    
    // Stop camera
    stopCameraBtn.addEventListener('click', function() {
        fetch('/stop_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                cameraFeed.src = '';
                startCameraBtn.disabled = false;
                stopCameraBtn.disabled = true;
                gestureText.textContent = 'No gesture detected';
                gestureTypeValue.textContent = 'Single Hand';
                leftHandIndicator.style.display = 'none';
                rightHandIndicator.style.display = 'none';
            }
        });
    });
    
    // Toggle audio
    toggleAudioBtn.addEventListener('click', function() {
        isAudioEnabled = !isAudioEnabled;
        const audioIcon = toggleAudioBtn.querySelector('.audio-icon');
        const audioText = toggleAudioBtn.querySelector('.audio-text');
        
        if (isAudioEnabled) {
            audioIcon.textContent = '🔊';
            audioText.textContent = 'Audio Enabled';
            toggleAudioBtn.classList.add('active');
        } else {
            audioIcon.textContent = '🔇';
            audioText.textContent = 'Audio Disabled';
            toggleAudioBtn.classList.remove('active');
        }
    });
    
    // Poll for gesture updates
    function pollGestures() {
        if (cameraFeed.src) {
            fetch('/get_gesture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.gesture) {
                    // Update gesture text
                    gestureText.textContent = data.gesture;
                    gestureText.classList.add('gesture-detected');
                    
                    // Update gesture type
                    gestureTypeValue.textContent = data.is_two_hand ? 'Two Hands' : 'Single Hand';
                    
                    // Update hand indicators
                    leftHandIndicator.style.display = data.hands_detected.includes('left') ? 'block' : 'none';
                    rightHandIndicator.style.display = data.hands_detected.includes('right') ? 'block' : 'none';
                    
                    // Handle audio
                    if (isAudioEnabled && data.audio_path) {
                        playAudio(data.audio_path);
                    }
                    
                    // Remove animation class after it completes
                    setTimeout(() => {
                        gestureText.classList.remove('gesture-detected');
                    }, 500);
                } else {
                    gestureText.textContent = 'No gesture detected';
                    gestureTypeValue.textContent = 'Single Hand';
                    leftHandIndicator.style.display = 'none';
                    rightHandIndicator.style.display = 'none';
                }
                
                // Continue polling
                setTimeout(pollGestures, 100);
            })
            .catch(error => {
                console.error('Error polling gestures:', error);
                setTimeout(pollGestures, 1000); // Retry after error
            });
        } else {
            setTimeout(pollGestures, 1000); // Check less frequently when camera is off
        }
    }
    
    // Play audio with queue management
    function playAudio(audioPath) {
        if (!isProcessing) {
            isProcessing = true;
            const audio = new Audio(audioPath);
            
            audio.onended = function() {
                isProcessing = false;
                if (audioQueue.length > 0) {
                    const nextAudio = audioQueue.shift();
                    playAudio(nextAudio);
                }
            };
            
            audio.onerror = function() {
                console.error('Error playing audio:', audioPath);
                isProcessing = false;
                if (audioQueue.length > 0) {
                    const nextAudio = audioQueue.shift();
                    playAudio(nextAudio);
                }
            };
            
            audio.play().catch(error => {
                console.error('Error playing audio:', error);
                isProcessing = false;
            });
        } else {
            audioQueue.push(audioPath);
        }
    }
    
    // Start polling
    pollGestures();
}); 