import mediapipe as mp
import cv2
import numpy as np
import time
from collections import deque

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only detect one hand
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.latest_gesture = None
        self.hands_detected = False
        self.last_processed_time = 0
        self.processing_interval = 0.1  # Process every 100ms
        
        # Gesture history for smoothing
        self.gesture_history = deque(maxlen=5)
        self.gesture_confidence = {}
        
        # Basic gesture mappings
        self.gesture_mappings = {
            # Numbers
            'one': 'One',
            'two': 'Two',
            'three': 'Three',
            'four': 'Four',
            'five': 'Five',
            
            # Basic gestures
            'hello': 'Hello',  # Added hello gesture
            'thumbs_up': 'Yes',
            'thumbs_down': 'No',
            'ok': 'OK',
            'peace': 'Peace',
            'rock': 'Rock',
            'call_me': 'Call Me',
            'stop': 'Stop',
            'point': 'Point',
            'wave': 'Wave',
            'fist': 'Fist',
            
            # Letters
            'a': 'A',
            'b': 'B',
            'c': 'C',
            'd': 'D',
            'e': 'E',
            'f': 'F',
            'g': 'G',
            'h': 'H',
            'i': 'I',
            'j': 'J',
            'k': 'K',
            'l': 'L',
            'm': 'M',
            'n': 'N',
            'o': 'O'
        }

    def process_frame(self, frame):
        current_time = time.time()
        if current_time - self.last_processed_time < self.processing_interval:
            return frame
            
        self.last_processed_time = current_time
        
        # Convert to RGB and process
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        self.hands_detected = False
        self.latest_gesture = None
        
        if results.multi_hand_landmarks:
            self.hands_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract key points and calculate states
                points = self._extract_key_points(hand_landmarks)
                finger_states = self._calculate_finger_states(points)
                angles = self._calculate_angles(points)
                distances = self._calculate_finger_distances(points)
                
                # Recognize gesture with confidence scoring
                gesture, confidence = self._recognize_gesture_with_confidence(
                    points, finger_states, angles, distances)
                
                if gesture and confidence > 0.7:  # Only accept high confidence gestures
                    self.gesture_history.append(gesture)
                    self.gesture_confidence[gesture] = confidence
                    
                    # Apply smoothing using majority voting
                    smoothed_gesture = self._smooth_gesture()
                    if smoothed_gesture:
                        self.latest_gesture = {
                            'gesture': self.gesture_mappings.get(smoothed_gesture, 'Unknown'),
                            'is_two_hand': False,
                            'confidence': confidence
                        }
                        
                        # Display the recognized gesture with confidence
                        cv2.putText(frame, f"{self.latest_gesture['gesture']} ({confidence:.2f})", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

    def _smooth_gesture(self):
        if not self.gesture_history:
            return None
            
        # Count occurrences of each gesture
        gesture_counts = {}
        for gesture in self.gesture_history:
            if gesture in gesture_counts:
                gesture_counts[gesture] += 1
        else:
                gesture_counts[gesture] = 1
        
        # Find the most common gesture
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        
        # Only return if we have a clear majority
        if most_common[1] > len(self.gesture_history) / 2:
            return most_common[0]
        return None

    def _recognize_gesture_with_confidence(self, points, states, angles, distances):
        # Initialize confidence scores
        gesture_scores = {}
        
        # Hello gesture with enhanced accuracy
        gesture_scores['hello'] = self._score_hello(points, states, angles, distances)
        
        # Number gestures with confidence scoring
        gesture_scores['one'] = self._score_number_one(points, states, angles)
        gesture_scores['two'] = self._score_number_two(points, states, angles)
        gesture_scores['three'] = self._score_number_three(points, states, angles)
        gesture_scores['four'] = self._score_number_four(points, states, angles)
        gesture_scores['five'] = self._score_number_five(points, states, angles)
        
        # Basic gestures with confidence scoring
        gesture_scores['thumbs_up'] = self._score_thumbs_up(points, states, angles)
        gesture_scores['thumbs_down'] = self._score_thumbs_down(points, states, angles)
        gesture_scores['ok'] = self._score_ok_sign(points, states, angles, distances)
        gesture_scores['peace'] = self._score_peace(points, states, angles)
        gesture_scores['rock'] = self._score_rock(points, states, angles)
        gesture_scores['call_me'] = self._score_call_me(points, states, angles)
        gesture_scores['stop'] = self._score_stop(points, states, angles)
        gesture_scores['point'] = self._score_point(points, states, angles)
        gesture_scores['wave'] = self._score_wave(points, states, angles)
        gesture_scores['fist'] = self._score_fist(points, states, angles)
        
        # Find the gesture with highest confidence
        if gesture_scores:
            best_gesture = max(gesture_scores.items(), key=lambda x: x[1])
            if best_gesture[1] > 0.5:  # Minimum confidence threshold
                return best_gesture
        return None, 0.0

    def _score_hello(self, points, states, angles, distances):
        # Hello gesture is typically a wave with all fingers extended
        # and the hand moving side to side
        
        # First check if all fingers are extended
        if not all(states[f] for f in ['index', 'middle', 'ring', 'pinky']):
            return 0.0
        
        # Check if fingers are straight (angles close to 180 degrees)
        if any(angles.get(f, 180) < 160 for f in ['index', 'middle', 'ring', 'pinky']):
            return 0.0
        
        # Check hand orientation (should be roughly vertical)
        wrist = points['wrist']
        middle_base = points['middle'][0]
        
        # Calculate vertical angle of the hand
        hand_angle = np.degrees(np.arctan2(
            middle_base.y - wrist.y,
            middle_base.x - wrist.x
        ))
        
        # Hand should be roughly vertical (between 60 and 120 degrees)
        if not (60 <= abs(hand_angle) <= 120):
            return 0.0
        
        # Check finger spread
        if 'index_middle' in distances and distances['index_middle'] < 0.05:
            return 0.0
        if 'middle_ring' in distances and distances['middle_ring'] < 0.05:
            return 0.0
        if 'ring_pinky' in distances and distances['ring_pinky'] < 0.05:
            return 0.0
        
        # Calculate confidence based on multiple factors
        confidence = 0.9
        
        # Penalize if thumb is extended (should be relaxed)
        if states['thumb']:
            confidence -= 0.2
        
        # Penalize if fingers are too close together
        if 'index_middle' in distances and distances['index_middle'] < 0.1:
            confidence -= 0.1
        if 'middle_ring' in distances and distances['middle_ring'] < 0.1:
            confidence -= 0.1
        if 'ring_pinky' in distances and distances['ring_pinky'] < 0.1:
            confidence -= 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def _calculate_angles(self, points):
        angles = {}
        for finger, finger_points in points.items():
            if finger != 'wrist' and len(finger_points) >= 3:
                # Calculate angle between three consecutive points
                v1 = np.array([finger_points[0].x - finger_points[1].x,
                              finger_points[0].y - finger_points[1].y])
                v2 = np.array([finger_points[2].x - finger_points[1].x,
                              finger_points[2].y - finger_points[1].y])
                angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                angles[finger] = np.degrees(angle)
        return angles

    def _calculate_finger_distances(self, points):
        distances = {}
        if 'index' in points and 'middle' in points:
            distances['index_middle'] = self._calculate_distance(
                points['index'][-1], points['middle'][-1])
        if 'middle' in points and 'ring' in points:
            distances['middle_ring'] = self._calculate_distance(
                points['middle'][-1], points['ring'][-1])
        if 'ring' in points and 'pinky' in points:
            distances['ring_pinky'] = self._calculate_distance(
                points['ring'][-1], points['pinky'][-1])
        if 'thumb' in points and 'index' in points:
            distances['thumb_index'] = self._calculate_distance(
                points['thumb'][-1], points['index'][-1])
        return distances

    def _calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    # Confidence scoring functions
    def _score_number_one(self, points, states, angles):
        if not states['index'] or any(states[f] for f in ['middle', 'ring', 'pinky']):
            return 0.0
        
        # Check if index finger is straight
        if 'index' in angles and angles['index'] < 160:
            return 0.0
            
        return 0.9

    def _score_number_two(self, points, states, angles):
        if not (states['index'] and states['middle']) or any(states[f] for f in ['ring', 'pinky']):
            return 0.0
        
        # Check if fingers are straight
        if ('index' in angles and angles['index'] < 160) or \
           ('middle' in angles and angles['middle'] < 160):
            return 0.0
            
        return 0.9

    def _score_number_three(self, points, states, angles):
        if not all(states[f] for f in ['index', 'middle', 'ring']) or states['pinky']:
            return 0.0
        
        # Check if fingers are straight
        if any(angles.get(f, 180) < 160 for f in ['index', 'middle', 'ring']):
            return 0.0
            
        return 0.9

    def _score_number_four(self, points, states, angles):
        if not all(states[f] for f in ['index', 'middle', 'ring', 'pinky']):
            return 0.0
        
        # Check if fingers are straight
        if any(angles.get(f, 180) < 160 for f in ['index', 'middle', 'ring', 'pinky']):
            return 0.0
            
        return 0.9

    def _score_number_five(self, points, states, angles):
        if not all(states.values()):
            return 0.0
        
        # Check if fingers are straight
        if any(angles.get(f, 180) < 160 for f in states.keys()):
            return 0.0
            
        return 0.9

    def _score_thumbs_up(self, points, states, angles):
        if not states['thumb'] or any(states[f] for f in ['index', 'middle', 'ring', 'pinky']):
            return 0.0
        
        # Check if thumb is straight
        if 'thumb' in angles and angles['thumb'] < 160:
            return 0.0
            
        return 0.9

    def _score_thumbs_down(self, points, states, angles):
        if states['thumb'] or any(states[f] for f in ['index', 'middle', 'ring', 'pinky']):
            return 0.0
        
        return 0.9

    def _score_ok_sign(self, points, states, angles, distances):
        if not (states['thumb'] and states['index']) or any(states[f] for f in ['middle', 'ring', 'pinky']):
            return 0.0
        
        # Check if thumb and index form a circle
        if 'thumb_index' in distances and distances['thumb_index'] > 0.1:
            return 0.0
            
        return 0.9

    def _score_peace(self, points, states, angles):
        if not (states['index'] and states['middle']) or any(states[f] for f in ['ring', 'pinky']):
            return 0.0
        
        # Check if fingers are straight
        if ('index' in angles and angles['index'] < 160) or \
           ('middle' in angles and angles['middle'] < 160):
            return 0.0
            
        return 0.9

    def _score_rock(self, points, states, angles):
        if not (states['index'] and states['pinky']) or any(states[f] for f in ['middle', 'ring']):
            return 0.0
        
        # Check if fingers are straight
        if ('index' in angles and angles['index'] < 160) or \
           ('pinky' in angles and angles['pinky'] < 160):
            return 0.0
            
        return 0.9

    def _score_call_me(self, points, states, angles):
        if not states['pinky'] or any(states[f] for f in ['index', 'middle', 'ring']):
            return 0.0
        
        # Check if pinky is straight
        if 'pinky' in angles and angles['pinky'] < 160:
            return 0.0
            
        return 0.9

    def _score_stop(self, points, states, angles):
        if any(states.values()):
            return 0.0
        return 0.9

    def _score_point(self, points, states, angles):
        if not states['index'] or any(states[f] for f in ['middle', 'ring', 'pinky']):
            return 0.0
        
        # Check if index is straight
        if 'index' in angles and angles['index'] < 160:
            return 0.0
            
        return 0.9

    def _score_wave(self, points, states, angles):
        if not all(states[f] for f in ['index', 'middle', 'ring', 'pinky']):
            return 0.0
        
        # Check if fingers are straight
        if any(angles.get(f, 180) < 160 for f in ['index', 'middle', 'ring', 'pinky']):
            return 0.0
            
        return 0.9

    def _score_fist(self, points, states, angles):
        if any(states.values()):
            return 0.0
        return 0.9

    def _extract_key_points(self, landmarks):
        return {
            'thumb': [landmarks.landmark[i] for i in range(1, 5)],
            'index': [landmarks.landmark[i] for i in range(5, 9)],
            'middle': [landmarks.landmark[i] for i in range(9, 13)],
            'ring': [landmarks.landmark[i] for i in range(13, 17)],
            'pinky': [landmarks.landmark[i] for i in range(17, 21)],
            'wrist': landmarks.landmark[0]
        }

    def _calculate_finger_states(self, points):
        states = {}
        for finger, finger_points in points.items():
            if finger != 'wrist':
                if finger == 'thumb':
                    # Thumb is extended if the tip is above the base
                    states[finger] = finger_points[-1].y < finger_points[0].y
                else:
                    # Other fingers are extended if the tip is above the middle joint
                    states[finger] = finger_points[-1].y < finger_points[-2].y
        return states

    def get_latest_gesture(self):
        return self.latest_gesture
        
    def get_hands_detected(self):
        return self.hands_detected 