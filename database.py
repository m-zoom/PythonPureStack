from models import User, FaceImage
from app import db
import logging

class DatabaseManager:
    @staticmethod
    def add_user(name, email=None, phone=None, details=None):
        """Add a new user to the database"""
        try:
            user = User(name=name, email=email, phone=phone, details=details)
            db.session.add(user)
            db.session.commit()
            logging.debug(f"Added new user: {name}")
            return user
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error adding user: {str(e)}")
            raise

    @staticmethod
    def get_user(user_id):
        """Get a user by ID"""
        return User.query.get(user_id)

    @staticmethod
    def get_all_users():
        """Get all users"""
        return User.query.all()

    @staticmethod
    def update_user(user_id, name=None, email=None, phone=None, details=None):
        """Update user details"""
        try:
            user = User.query.get(user_id)
            if not user:
                logging.error(f"User not found: {user_id}")
                return False
            
            if name:
                user.name = name
            if email:
                user.email = email
            if phone:
                user.phone = phone
            if details:
                user.details = details
                
            db.session.commit()
            logging.debug(f"Updated user: {user_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error updating user: {str(e)}")
            return False

    @staticmethod
    def delete_user(user_id):
        """Delete a user"""
        try:
            user = User.query.get(user_id)
            if not user:
                logging.error(f"User not found: {user_id}")
                return False
                
            db.session.delete(user)
            db.session.commit()
            logging.debug(f"Deleted user: {user_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error deleting user: {str(e)}")
            return False

    @staticmethod
    def add_face_image(user_id, image_path):
        """Add a face image for a user"""
        try:
            face_image = FaceImage(user_id=user_id, image_path=image_path)
            db.session.add(face_image)
            db.session.commit()
            logging.debug(f"Added face image for user {user_id}: {image_path}")
            return face_image
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error adding face image: {str(e)}")
            raise

    @staticmethod
    def get_user_face_images(user_id):
        """Get all face images for a user"""
        return FaceImage.query.filter_by(user_id=user_id).all()

    @staticmethod
    def get_all_face_images():
        """Get all face images"""
        return FaceImage.query.all()

    @staticmethod
    def delete_face_image(image_id):
        """Delete a face image"""
        try:
            face_image = FaceImage.query.get(image_id)
            if not face_image:
                logging.error(f"Face image not found: {image_id}")
                return False
                
            db.session.delete(face_image)
            db.session.commit()
            logging.debug(f"Deleted face image: {image_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error deleting face image: {str(e)}")
            return False
