import cv2
import numpy as np
import os
import logging
from database import DatabaseManager

class FaceRecognition:
    def __init__(self):
        # Load the face detector
        self.cascade_path = os.path.join(os.path.dirname(__file__), 'static', 'haarcascades', 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        logging.debug(f"Loading face cascade from: {self.cascade_path}")
        if self.face_cascade.empty():
            logging.error(f"Error loading cascade classifier from {self.cascade_path}")
            raise Exception(f"Failed to load Haar cascade from {self.cascade_path}")
        
        # Initialize the face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
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
        """Detect faces in an image and return list of face regions"""
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to load image: {image_path}")
                return [], None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return faces, gray
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
            image_path = os.path.join(os.path.dirname(__file__), face_image.image_path)
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
    
    def recognize_face(self, image_path, confidence_threshold=70):
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
    
    def find_matching_faces(self, image_path, confidence_threshold=50):
        """Find all potential matches for a face above the confidence threshold"""
        if not self.model_trained:
            logging.warning("Face recognition model not trained yet")
            return []
        
        matches = []
        
        # Get the face image
        face_img = self.extract_face(image_path)
        
        if face_img is None:
            return matches
        
        # Get all users
        all_users = DatabaseManager.get_all_users()
        
        # Check each user
        for user in all_users:
            user_face_images = DatabaseManager.get_user_face_images(user.id)
            
            highest_confidence = 0
            for face_image in user_face_images:
                try:
                    # Predict using the model
                    predicted_id, confidence = self.recognizer.predict(face_img)
                    
                    # Convert confidence score
                    confidence_score = 100 - min(confidence, 100)
                    
                    if confidence_score > highest_confidence:
                        highest_confidence = confidence_score
                except Exception:
                    continue
            
            if highest_confidence >= confidence_threshold:
                matches.append({
                    'user': user,
                    'confidence': highest_confidence
                })
        
        # Sort matches by confidence (highest first)
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches
