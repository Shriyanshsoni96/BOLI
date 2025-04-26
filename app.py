from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
import os
from datetime import datetime
import base64
from gesture_recognition import GestureRecognizer
import time
import threading

app = Flask(__name__)
gesture_recognizer = GestureRecognizer()

# Ensure the audio directory exists
os.makedirs('static/audio', exist_ok=True)

# Initialize camera with proper error handling
def init_camera():
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        
        return camera
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return None

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition parameters
CONFIDENCE_THRESHOLD = 0.7
GESTURE_HISTORY_SIZE = 5
GESTURE_COOLDOWN = 1.0  # seconds

# Store gesture history
gesture_history = []
last_gesture_time = 0
current_gesture = None
current_confidence = 0
fps_counter = 0
fps_timer = time.time()

# Global camera variable
camera = None

def calculate_hand_landmarks_distance(landmarks1, landmarks2):
    """Calculate the average distance between corresponding landmarks of two hands"""
    if not landmarks1 or not landmarks2:
        return float('inf')
    
    distances = []
    for lm1, lm2 in zip(landmarks1, landmarks2):
        distance = np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)
        distances.append(distance)
    return np.mean(distances)

def recognize_gesture(hand_landmarks):
    """Enhanced gesture recognition with 30 basic gestures"""
    if not hand_landmarks:
        return None, 0.0

    # Get hand landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Get finger PIP (Proximal Interphalangeal) joints
    thumb_pip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    # Get finger MCP (Metacarpophalangeal) joints
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # Calculate finger states (extended or not)
    thumb_extended = thumb_tip.y < thumb_pip.y
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    ring_extended = ring_tip.y < ring_pip.y
    pinky_extended = pinky_tip.y < pinky_pip.y

    # Calculate finger angles
    thumb_angle = np.arctan2(thumb_tip.y - thumb_mcp.y, thumb_tip.x - thumb_mcp.x)
    index_angle = np.arctan2(index_tip.y - index_mcp.y, index_tip.x - index_mcp.x)
    middle_angle = np.arctan2(middle_tip.y - middle_mcp.y, middle_tip.x - middle_mcp.x)
    ring_angle = np.arctan2(ring_tip.y - ring_mcp.y, ring_tip.x - ring_mcp.x)
    pinky_angle = np.arctan2(pinky_tip.y - pinky_mcp.y, pinky_tip.x - pinky_mcp.x)

    # Calculate distances between fingertips
    thumb_index_dist = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    index_middle_dist = np.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
    middle_ring_dist = np.sqrt((middle_tip.x - ring_tip.x)**2 + (middle_tip.y - ring_tip.y)**2)
    ring_pinky_dist = np.sqrt((ring_tip.x - pinky_tip.x)**2 + (ring_tip.y - pinky_tip.y)**2)

    # Initialize gesture and confidence
    gesture = None
    confidence = 0.0

    # 1. Hello (previously Open Hand)
    if all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
        gesture = "Hello"
        confidence = 0.9

    # 2. No
    elif index_extended and not any([thumb_extended, middle_extended, ring_extended, pinky_extended]) and index_angle > 0:
        gesture = "No"
        confidence = 0.8

    # 3. Fist
    elif not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
        gesture = "Fist"
        confidence = 0.9

    # 4. Point
    elif index_extended and not any([middle_extended, ring_extended, pinky_extended]):
        gesture = "Point"
        confidence = 0.8

    # 5. Peace
    elif index_extended and middle_extended and not any([ring_extended, pinky_extended]):
        gesture = "Peace"
        confidence = 0.8

    # 6. Thumbs Up
    elif thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        gesture = "Thumbs Up"
        confidence = 0.8

    # 7. Rock
    elif index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        gesture = "Rock"
        confidence = 0.7

    # 8. OK
    elif thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]) and thumb_index_dist < 0.1:
        gesture = "OK"
        confidence = 0.8

    # 9. Three
    elif index_extended and middle_extended and ring_extended and not any([thumb_extended, pinky_extended]):
        gesture = "Three"
        confidence = 0.8

    # 10. Four
    elif all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended:
        gesture = "Four"
        confidence = 0.8

    # 11. Call Me
    elif pinky_extended and thumb_extended and not any([index_extended, middle_extended, ring_extended]):
        gesture = "Call Me"
        confidence = 0.7

    # 12. Stop
    elif all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]) and all([thumb_angle > 0, index_angle > 0, middle_angle > 0, ring_angle > 0, pinky_angle > 0]):
        gesture = "Stop"
        confidence = 0.8

    # 13. Victory
    elif index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]) and index_angle < 0 and middle_angle < 0:
        gesture = "Victory"
        confidence = 0.8

    # 14. Love
    elif thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]) and thumb_angle > 0 and index_angle > 0:
        gesture = "Love"
        confidence = 0.7

    # 15. Gun
    elif thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]) and thumb_angle < 0 and index_angle < 0:
        gesture = "Gun"
        confidence = 0.7

    # 16. Horns
    elif index_extended and pinky_extended and not any([thumb_extended, middle_extended, ring_extended]) and index_angle < 0 and pinky_angle < 0:
        gesture = "Horns"
        confidence = 0.7

    # 17. Spider-Man
    elif thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]):
        gesture = "Spider-Man"
        confidence = 0.7

    # 18. Shaka
    elif thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]) and thumb_angle > 0 and pinky_angle > 0:
        gesture = "Shaka"
        confidence = 0.7

    # 19. Money
    elif thumb_extended and index_extended and middle_extended and not any([ring_extended, pinky_extended]) and thumb_angle > 0:
        gesture = "Money"
        confidence = 0.7

    # 20. Telephone
    elif thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]) and thumb_angle < 0 and pinky_angle < 0:
        gesture = "Telephone"
        confidence = 0.7

    # 21. Loser
    elif thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]) and thumb_angle < 0:
        gesture = "Loser"
        confidence = 0.7

    # 22. Perfect
    elif thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]) and thumb_angle < 0 and index_angle < 0 and thumb_index_dist < 0.1:
        gesture = "Perfect"
        confidence = 0.7

    # 23. Cross
    elif index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]) and index_angle > 0 and middle_angle < 0:
        gesture = "Cross"
        confidence = 0.7

    # 24. Wave
    elif all([index_extended, middle_extended, ring_extended, pinky_extended]) and not thumb_extended and all([index_angle < 0, middle_angle < 0, ring_angle < 0, pinky_angle < 0]):
        gesture = "Wave"
        confidence = 0.7

    # 25. Pinch
    elif thumb_extended and index_extended and not any([middle_extended, ring_extended, pinky_extended]) and thumb_index_dist < 0.05:
        gesture = "Pinch"
        confidence = 0.7

    # 26. Hang Loose
    elif thumb_extended and pinky_extended and not any([index_extended, middle_extended, ring_extended]) and thumb_angle > 0 and pinky_angle > 0 and thumb_pip.y < thumb_tip.y:
        gesture = "Hang Loose"
        confidence = 0.7

    # 27. I Love You
    elif thumb_extended and index_extended and pinky_extended and not any([middle_extended, ring_extended]):
        gesture = "I Love You"
        confidence = 0.7

    # 28. Metal
    elif index_extended and pinky_extended and not any([thumb_extended, middle_extended, ring_extended]) and index_angle > 0 and pinky_angle > 0:
        gesture = "Metal"
        confidence = 0.7

    # 29. High Five
    elif all([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]) and all([thumb_angle < 0, index_angle < 0, middle_angle < 0, ring_angle < 0, pinky_angle < 0]):
        gesture = "High Five"
        confidence = 0.7

    # 30. Fingers Crossed
    elif index_extended and middle_extended and not any([thumb_extended, ring_extended, pinky_extended]) and index_angle > 0 and middle_angle < 0 and index_middle_dist < 0.1:
        gesture = "Fingers Crossed"
        confidence = 0.7

    # 31. Salute
    elif thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]) and thumb_angle > 0 and thumb_pip.y > thumb_tip.y:
        gesture = "Salute"
        confidence = 0.7

    return gesture, confidence

def generate_audio(gesture):
    """Generate audio for the recognized gesture"""
    try:
        timestamp = int(time.time())
        audio_path = f"static/audio/gesture_{timestamp}.mp3"
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        tts = gTTS(text=f"Gesture recognized: {gesture}", lang='en')
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def process_frame(frame, audio_enabled=False):
    """Process a single frame for gesture recognition"""
    global gesture_history, last_gesture_time, current_gesture, current_confidence, fps_counter, fps_timer

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_timer = time.time()
    else:
        fps = fps_counter

    # Initialize response data
    response_data = {
        'success': False,
        'gesture': None,
        'confidence': 0,
        'hands_detected': False,
        'fps': fps,
        'audio_path': None
    }

    if results.multi_hand_landmarks:
        response_data['hands_detected'] = True
        
        # Process each detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Recognize gesture for this hand
            gesture, confidence = recognize_gesture(hand_landmarks)
            
            if gesture and confidence >= CONFIDENCE_THRESHOLD:
                current_time = time.time()
                
                # Check if enough time has passed since the last gesture
                if current_time - last_gesture_time >= GESTURE_COOLDOWN:
                    # Update gesture history
                    gesture_history.append((gesture, confidence))
                    if len(gesture_history) > GESTURE_HISTORY_SIZE:
                        gesture_history.pop(0)
                    
                    # Calculate average confidence for this gesture in history
                    gesture_confidence = sum(c for g, c in gesture_history if g == gesture) / len([g for g, _ in gesture_history if g == gesture])
                    
                    if gesture_confidence >= CONFIDENCE_THRESHOLD:
                        current_gesture = gesture
                        current_confidence = gesture_confidence
                        last_gesture_time = current_time
                        
                        # Generate audio if enabled
                        if audio_enabled:
                            audio_path = generate_audio(gesture)
                            if audio_path:
                                response_data['audio_path'] = audio_path

    # Update response data
    response_data['success'] = True
    response_data['gesture'] = current_gesture
    response_data['confidence'] = current_confidence

    return response_data

def generate_frames(audio_enabled=False):
    """Generate video frames with gesture recognition"""
    global camera
    
    # Initialize camera if not already initialized
    if camera is None:
        camera = init_camera()
        if camera is None:
            return
    
    while True:
        try:
            success, frame = camera.read()
            if not success:
                print("Error: Failed to capture frame")
                break
            else:
                # Process the frame
                response_data = process_frame(frame, audio_enabled)
                
                # Draw hand landmarks on the frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                
                # Add gesture text to frame
                if response_data['gesture']:
                    cv2.putText(
                        frame,
                        f"{response_data['gesture']} ({response_data['confidence']:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in video feed: {e}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    audio_enabled = request.args.get('audio', 'false').lower() == 'true'
    return Response(generate_frames(audio_enabled),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_gesture')
def get_gesture():
    global camera
    
    # Initialize camera if not already initialized
    if camera is None:
        camera = init_camera()
        if camera is None:
            return jsonify({'success': False, 'error': 'Failed to initialize camera'})
    
    success, frame = camera.read()
    if not success:
        return jsonify({'success': False, 'error': 'Failed to capture frame'})
    
    audio_enabled = request.args.get('audio', 'false').lower() == 'true'
    response_data = process_frame(frame, audio_enabled)
    return jsonify(response_data)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
            return jsonify({'success': True, 'message': 'Camera stopped successfully'})
        return jsonify({'success': True, 'message': 'Camera was not active'})
    except Exception as e:
        print(f"Error stopping camera: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/gestures')
def gestures():
    return render_template('gestures.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True) 