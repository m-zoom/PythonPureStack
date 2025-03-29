import cv2
import numpy as np
import os
import logging
import base64
from database import DatabaseManager

class FaceRecognition:
    def __init__(self):
        """Initialize face recognition with improved detection methods"""
        # Load the Haar cascade for legacy support and faster initial detection
        self.cascade_path = os.path.join(os.path.dirname(__file__), 'static', 'haarcascades', 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        logging.debug(f"Loading face cascade from: {self.cascade_path}")
        if self.face_cascade.empty():
            logging.error(f"Error loading cascade classifier from {self.cascade_path}")
            raise Exception(f"Failed to load Haar cascade from {self.cascade_path}")
        
        # Configure advanced DNN-based face detector for better accuracy
        self.use_dnn_detection = True
        try:
            # Define model paths
            model_dir = os.path.join(os.path.dirname(__file__), 'static/models')
            os.makedirs(model_dir, exist_ok=True)
            
            model_file = os.path.join(model_dir, 'opencv_face_detector_uint8.pb')
            config_file = os.path.join(model_dir, 'opencv_face_detector.pbtxt')
            
            # Check if model files exist
            if not os.path.exists(model_file):
                logging.warning("DNN model files not found. Using Haar cascade for detection.")
                self.use_dnn_detection = False
            else:
                try:
                    self.face_net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                    logging.info("Loaded DNN face detector model")
                except Exception as e:
                    logging.error(f"Error loading DNN face detector: {str(e)}")
                    self.use_dnn_detection = False
        except Exception as e:
            logging.error(f"Error setting up DNN detection: {str(e)}")
            self.use_dnn_detection = False
        
        # Initialize the face recognizer - LBPH for better handling of different lighting conditions
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,        # Increased radius for more detail
            neighbors=8,    # More neighbors for better accuracy 
            grid_x=8,        # More grid cells for better spatial representation
            grid_y=8,        # More grid cells for better spatial representation
            threshold=70     # Lower threshold for accepting matches
        )
        
        # Path to save trained model
        self.model_path = os.path.join(os.path.dirname(__file__), 'static/models', 'trained_model.yml')
        
        # Check if a trained model exists
        self.model_trained = os.path.exists(self.model_path)
        if self.model_trained:
            try:
                self.recognizer.read(self.model_path)
                logging.info("Loaded existing face recognition model")
            except Exception as e:
                self.model_trained = False
                logging.error(f"Error loading model: {str(e)}")
    
    def detect_faces(self, image_path):
        """
        Detect faces in an image using advanced methods
        Uses DNN-based detector when available, falls back to Haar cascade
        Returns list of face regions and the processed grayscale image
        """
        try:
            # Normalize path separators to forward slashes for web URLs
            image_path = image_path.replace('\\', '/')
            
            # Handle image paths correctly for both local and Replit environments
            if not os.path.exists(image_path):
                # Try adding static/ prefix if needed
                if not image_path.startswith('static/'):
                    potential_path = os.path.join(os.path.dirname(__file__), 'static', image_path).replace('\\', '/')
                    if os.path.exists(potential_path):
                        image_path = potential_path
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to load image: {image_path}")
                return [], None
                
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Convert to grayscale for Haar cascade and LBPH
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to improve detection in different lighting
            gray = cv2.equalizeHist(gray)
            
            # Apply face detection
            faces = []
            
            # Try DNN-based detection first (better for different poses and angles)
            if self.use_dnn_detection:
                try:
                    # Prepare the image for DNN detection
                    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
                    self.face_net.setInput(blob)
                    detections = self.face_net.forward()
                    
                    # Process detections
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        
                        # Filter by confidence - use lower threshold to catch more potential faces
                        if confidence > 0.5:  # Lower threshold for better sensitivity
                            # Get box coordinates
                            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                            x1, y1, x2, y2 = box.astype(int)
                            
                            # Ensure coordinates are within image bounds
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(width, x2)
                            y2 = min(height, y2)
                            
                            # Convert to x, y, w, h format for compatibility
                            faces.append((x1, y1, x2-x1, y2-y1))
                    
                    # If DNN found faces, return them
                    if len(faces) > 0:
                        logging.debug(f"DNN detector found {len(faces)} faces")
                        return faces, gray
                        
                except Exception as e:
                    logging.error(f"Error in DNN face detection: {str(e)}")
                    # Continue to Haar cascade as fallback
            
            # Fall back to enhanced Haar cascade for detection
            # Try multiple scales and parameters for better detection
            # Use a combination of parameters to find faces
            cascade_faces = []
            
            # First attempt with standard parameters
            faces1 = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces1) > 0:
                cascade_faces.extend(faces1)
            
            # Second attempt with more aggressive scaling but higher threshold
            faces2 = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # More gradual scaling
                minNeighbors=3,    # Lower threshold for detection
                minSize=(20, 20),  # Smaller faces
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Add new faces not already detected
            for face in faces2:
                is_duplicate = False
                for existing_face in cascade_faces:
                    # Check if there's significant overlap
                    x1, y1, w1, h1 = face
                    x2, y2, w2, h2 = existing_face
                    
                    # Calculate overlap
                    overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                    overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                    overlap_area = overlap_x * overlap_y
                    
                    if overlap_area > 0.5 * min(w1*h1, w2*h2):
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    cascade_faces.append(face)
            
            # If Haar cascade found faces, use them
            if len(cascade_faces) > 0:
                logging.debug(f"Haar cascade found {len(cascade_faces)} faces")
                return cascade_faces, gray
                
            # If no faces were found by either method
            return [], gray
            
        except Exception as e:
            logging.error(f"Error detecting faces: {str(e)}")
            return [], None
    
    def extract_face(self, image_path, save_path=None):
        """Extract face from an image and optionally save it"""
        faces, gray = self.detect_faces(image_path)
        
        if len(faces) == 0:
            logging.warning(f"No faces detected in {image_path}")
            return None
        
        # Get the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract the face
        face_img = gray[y:y+h, x:x+w]
        
        # Resize to a standard size
        face_img = cv2.resize(face_img, (100, 100))
        
        # Save the face if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, face_img)
            logging.debug(f"Saved extracted face to {save_path}")
        
        return face_img
    
    def train_model(self):
        """Train the face recognizer model with all faces in the database"""
        faces = []
        labels = []
        label_mapping = {}
        
        # Get all face images from the database
        all_face_images = DatabaseManager.get_all_face_images()
        
        if not all_face_images:
            logging.warning("No face images in database to train model")
            return False
        
        # Prepare training data
        for idx, face_image in enumerate(all_face_images):
            # Handle image paths with direct path processing
            # Make sure to use forward slashes for consistency
            image_path = os.path.join(os.path.dirname(__file__), face_image.image_path).replace('\\', '/')
            
            extracted_face = self.extract_face(image_path)
            
            if extracted_face is not None:
                faces.append(extracted_face)
                labels.append(face_image.user_id)
                label_mapping[face_image.user_id] = face_image.user_id
        
        if not faces:
            logging.warning("No valid faces extracted for training")
            return False
        
        # Train the model
        try:
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save(self.model_path)
            self.model_trained = True
            logging.info("Face recognition model trained successfully")
            return True
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return False
    
    def recognize_face(self, image_path, confidence_threshold=50):
        """Recognize a face from an image and return user ID and confidence"""
        if not self.model_trained:
            logging.warning("Face recognition model not trained yet")
            return None, 0
        
        # Extract face from image
        face_img = self.extract_face(image_path)
        
        if face_img is None:
            return None, 0
        
        # Predict
        try:
            user_id, confidence = self.recognizer.predict(face_img)
            
            # In OpenCV LBPH, lower confidence means better match
            # Convert to a more intuitive scale (0-100) where higher is better
            confidence_score = 100 - min(confidence, 100)
            
            if confidence_score >= confidence_threshold:
                return user_id, confidence_score
            else:
                return None, confidence_score
        except Exception as e:
            logging.error(f"Error recognizing face: {str(e)}")
            return None, 0
    
    def find_matching_faces(self, image_path, confidence_threshold=40):
        """
        Find all potential matches for a face above the confidence threshold
        Returns primary matches, similar people, and relatives
        """
        if not self.model_trained:
            logging.warning("Face recognition model not trained yet")
            return {
                'primary_matches': [],
                'similar_people': [],
                'relatives': []
            }
        
        primary_matches = []
        all_matches = []
        
        # Get the face image
        face_img = self.extract_face(image_path)
        
        if face_img is None:
            return {
                'primary_matches': [],
                'similar_people': [],
                'relatives': []
            }
        
        # Estimate gender from face image (simplified approximation)
        # Note: This is not a reliable gender detection method and should be improved
        # with a proper gender classification model in production
        gender = self._estimate_gender(face_img)
        
        # Get all users
        all_users = DatabaseManager.get_all_users()
        
        # Check each user
        for user in all_users:
            user_face_images = DatabaseManager.get_user_face_images(user.id)
            
            highest_confidence = 0
            visual_similarity = 0
            
            for face_image in user_face_images:
                try:
                    # Get the path to the user's face image
                    image_path = os.path.join(os.path.dirname(__file__), face_image.image_path).replace('\\', '/')
                    
                    # Extract the face from the image
                    user_face = self.extract_face(image_path)
                    
                    if user_face is not None:
                        # Use the model for prediction
                        # Predict using the model
                        predicted_id, confidence = self.recognizer.predict(face_img)
                        
                        # Convert confidence score
                        confidence_score = 100 - min(confidence, 100)
                        
                        if confidence_score > highest_confidence:
                            highest_confidence = confidence_score
                        
                        # Calculate visual similarity separately
                        # This would use feature-based comparison in a more sophisticated implementation
                        visual_similarity = max(visual_similarity, confidence_score * 0.8)
                    
                except Exception as e:
                    logging.error(f"Error processing face image: {str(e)}")
                    continue
            
            # Create match object with all relevant info
            match = {
                'user': user,
                'confidence': highest_confidence,
                'visual_similarity': visual_similarity,
                'same_gender': self._has_same_gender(user, gender),
                'relationship_info': self._get_relationship_info(user)
            }
            
            # Add to the appropriate lists based on confidence threshold
            all_matches.append(match)
            if highest_confidence >= confidence_threshold:
                primary_matches.append(match)
        
        # Sort matches by confidence (highest first)
        primary_matches.sort(key=lambda x: x['confidence'], reverse=True)
        all_matches.sort(key=lambda x: x['visual_similarity'], reverse=True)
        
        # Get similar-looking people who aren't in primary matches
        # These are people who look similar but didn't meet the confidence threshold
        similar_people = [m for m in all_matches if m not in primary_matches and m['visual_similarity'] >= 20][:5]
        
        # Get relatives of matched people
        relatives = self._get_relatives_of_matched_people(primary_matches)
        
        return {
            'primary_matches': primary_matches,
            'similar_people': similar_people,
            'relatives': relatives
        }
    
    def _estimate_gender(self, face_img):
        """
        Simple gender estimation based on facial features
        This is a placeholder for a more sophisticated gender detection model
        In a production system, this should be replaced with a proper ML-based gender classifier
        """
        # Placeholder: in the real implementation, this would use a trained gender detection model
        # For now, we'll return "unknown" as we should use a proper ML model for this task
        return "unknown"
    
    def _has_same_gender(self, user, gender):
        """
        Check if user has the same gender as estimated from the face
        In a real implementation, this would compare detected gender with user profile gender
        """
        # Simplified implementation
        # In a real system, this would compare the detected gender with the user's gender field
        return True if gender == "unknown" else False
    
    def _get_relationship_info(self, user):
        """Get relationship information for a user"""
        relationship_type = user.relationship_type if user.relationship_type else "Not specified"
        return {
            'type': relationship_type
        }
    
    def _get_relatives_of_matched_people(self, matches):
        """Get relatives of people in the primary matches list"""
        relatives = []
        seen_ids = set(m['user'].id for m in matches)
        
        # For each matched person
        for match in matches:
            user = match['user']
            
            # Get user's relatives
            user_relatives = DatabaseManager.get_user_relatives(user.id)
            
            # Add relatives not already in primary matches
            for relative in user_relatives:
                if relative.id not in seen_ids:
                    relatives.append({
                        'user': relative,
                        'confidence': 0,  # Relative is not visually matched
                        'visual_similarity': 0,
                        'same_gender': False,
                        'relationship_info': {
                            'type': relative.relationship_type if relative.relationship_type else "Not specified",
                            'related_to': user.name,
                            'relationship': "Relative"  # This could be more specific in a real implementation
                        }
                    })
                    seen_ids.add(relative.id)
        
        return relatives
        
    def analyze_video(self, video_path, confidence_threshold=50, sample_rate=1):
        """
        Analyze a video file and identify faces in it
        
        Args:
            video_path: Path to the video file
            confidence_threshold: Minimum confidence for face recognition
            sample_rate: Process every Nth frame (default: 1 - process every frame)
            
        Returns:
            List of dictionaries with detection results
        """
        if not self.model_trained:
            logging.warning("Face recognition model not trained yet")
            return []
        
        results = []
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            logging.info(f"Analyzing video: {video_path}, FPS: {fps}, Duration: {duration}s")
            
            # Initialize variables
            frame_number = 0
            
            # Process frames
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process every Nth frame based on sample_rate
                if frame_number % sample_rate == 0:
                    # Save frame as temporary image with forward slashes
                    temp_frame_path = os.path.join(os.path.dirname(__file__), 'static/uploads', f'temp_frame_{frame_number}.jpg').replace('\\', '/')
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Timestamp in seconds
                    timestamp = frame_number / fps if fps > 0 else 0
                    
                    # Detect faces in the frame
                    faces, gray = self.detect_faces(temp_frame_path)
                    
                    # Process each face in the frame
                    for face_idx, (x, y, w, h) in enumerate(faces):
                        # Extract and save the face with forward slashes
                        face_file = f'temp_face_{frame_number}_{face_idx}.jpg'
                        face_path = os.path.join(os.path.dirname(__file__), 'static/uploads', face_file).replace('\\', '/')
                        
                        # Create a copy of just the face region and save it
                        face_img = gray[y:y+h, x:x+w]
                        face_img = cv2.resize(face_img, (100, 100))
                        cv2.imwrite(face_path, face_img)
                        
                        # Recognize the face
                        user_id, confidence = self.recognize_face(face_path, confidence_threshold)
                        
                        if user_id is not None:
                            # Get user information
                            user = DatabaseManager.get_user(user_id)
                            
                            if user:
                                # Format timestamp as MM:SS
                                mins = int(timestamp // 60)
                                secs = int(timestamp % 60)
                                time_str = f"{mins:02d}:{secs:02d}"
                                
                                # Add detection to results
                                results.append({
                                    'frame': frame_number,
                                    'timestamp': timestamp,
                                    'time_str': time_str,
                                    'user': user,
                                    'confidence': confidence,
                                    'face_path': 'uploads/' + face_file,
                                    'face_position': (x, y, w, h)
                                })
                        
                        # Clean up temporary face file
                        os.remove(face_path)
                    
                    # Clean up temporary frame file
                    os.remove(temp_frame_path)
                
                frame_number += 1
            
            # Release the video capture object
            cap.release()
            
            # Group results by user
            user_detections = {}
            for result in results:
                user_id = result['user'].id
                if user_id not in user_detections:
                    user_detections[user_id] = {
                        'user': result['user'],
                        'detections': [],
                        'first_seen': result['timestamp'],
                        'last_seen': result['timestamp'],
                        'highest_confidence': result['confidence'],
                        'total_appearances': 0
                    }
                
                user_detections[user_id]['detections'].append(result)
                user_detections[user_id]['total_appearances'] += 1
                user_detections[user_id]['last_seen'] = max(user_detections[user_id]['last_seen'], result['timestamp'])
                user_detections[user_id]['highest_confidence'] = max(user_detections[user_id]['highest_confidence'], result['confidence'])
            
            # Convert to list and sort by most appearances
            summary = list(user_detections.values())
            summary.sort(key=lambda x: x['total_appearances'], reverse=True)
            
            logging.info(f"Video analysis complete. Found {len(results)} face detections of {len(summary)} unique individuals.")
            
            return {
                'results': results,
                'summary': summary,
                'video_info': {
                    'path': video_path,
                    'fps': fps,
                    'frame_count': frame_count,
                    'duration': duration,
                    'duration_str': f"{int(duration//60):02d}:{int(duration%60):02d}"
                }
            }
        
        except Exception as e:
            logging.error(f"Error analyzing video: {str(e)}")
            return []