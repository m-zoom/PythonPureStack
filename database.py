from models import User, FaceImage, RelationshipType
from app import db
import logging

class DatabaseManager:
    @staticmethod
    def add_user(name, email=None, phone=None, details=None, relationship_type=None):
        """Add a new user to the database"""
        try:
            user = User(
                name=name, 
                email=email, 
                phone=phone, 
                details=details,
                relationship_type=relationship_type
            )
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
        try:
            return User.query.get(user_id)
        except Exception as e:
            logging.error(f"Error retrieving user: {str(e)}")
            return None

    @staticmethod
    def get_all_users():
        """Get all users"""
        try:
            return User.query.all()
        except Exception as e:
            logging.error(f"Error retrieving all users: {str(e)}")
            return []

    @staticmethod
    def update_user(user_id, name=None, email=None, phone=None, details=None, relationship_type=None):
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
            if relationship_type:
                user.relationship_type = relationship_type
                
            db.session.commit()
            logging.debug(f"Updated user: {user_id}")
            return True
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error updating user: {str(e)}")
            return False
            
    @staticmethod
    def add_relative(user_id, relative_id, relationship_type=None):
        """Add a relative to a user"""
        try:
            user = User.query.get(user_id)
            relative = User.query.get(relative_id)
            
            if not user or not relative:
                logging.error(f"User or relative not found: {user_id} - {relative_id}")
                return False
            
            # Prevent adding the same person as their own relative
            if user_id == relative_id:
                return False
                
            success = user.add_relative(relative)
            
            if success and relationship_type:
                # If we're adding a relationship type, check if the type exists or create it
                rel_type = RelationshipType.query.filter_by(name=relationship_type).first()
                if not rel_type:
                    rel_type = RelationshipType(name=relationship_type)
                    db.session.add(rel_type)
                
                # Store the relationship type on the user
                user.relationship_type = relationship_type
            
            db.session.commit()
            logging.debug(f"Added relative {relative_id} to user {user_id}")
            return success
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error adding relative: {str(e)}")
            return False
    
    @staticmethod
    def remove_relative(user_id, relative_id):
        """Remove a relative from a user"""
        try:
            user = User.query.get(user_id)
            relative = User.query.get(relative_id)
            
            if not user or not relative:
                logging.error(f"User or relative not found: {user_id} - {relative_id}")
                return False
                
            success = user.remove_relative(relative)
            
            db.session.commit()
            logging.debug(f"Removed relative {relative_id} from user {user_id}")
            return success
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error removing relative: {str(e)}")
            return False
    
    @staticmethod
    def get_user_relatives(user_id):
        """Get all relatives for a user"""
        try:
            user = User.query.get(user_id)
            
            if not user:
                logging.error(f"User not found: {user_id}")
                return []
                
            return user.get_relatives()
        except Exception as e:
            logging.error(f"Error getting user relatives: {str(e)}")
            return []
    
    @staticmethod
    def get_relationship_types():
        """Get all relationship types"""
        try:
            return RelationshipType.query.all()
        except Exception as e:
            logging.error(f"Error getting relationship types: {str(e)}")
            return []

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
    def add_face_image(user_id, image_path, image_type='profile'):
        """Add a face image for a user with image type"""
        try:
            face_image = FaceImage(
                user_id=user_id, 
                image_path=image_path,
                image_type=image_type
            )
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
        try:
            return FaceImage.query.filter_by(user_id=user_id).all()
        except Exception as e:
            logging.error(f"Error getting user face images: {str(e)}")
            return []

    @staticmethod
    def get_all_face_images():
        """Get all face images"""
        try:
            return FaceImage.query.all()
        except Exception as e:
            logging.error(f"Error getting all face images: {str(e)}")
            return []

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
    
    @staticmethod
    def add_multiple_face_images(user_id, image_paths, image_types=None):
        """
        Add multiple face images for a user at once
        
        Args:
            user_id: The user ID to associate the images with
            image_paths: List of image paths
            image_types: Optional list of image types corresponding to the paths
        
        Returns:
            List of created FaceImage objects
        """
        created_images = []
        
        try:
            for i, path in enumerate(image_paths):
                # Get image type if provided
                image_type = 'profile'
                if image_types and i < len(image_types):
                    image_type = image_types[i]
                
                # Create face image
                face_image = FaceImage(
                    user_id=user_id,
                    image_path=path,
                    image_type=image_type
                )
                db.session.add(face_image)
                created_images.append(face_image)
            
            db.session.commit()
            logging.debug(f"Added {len(created_images)} face images for user {user_id}")
            return created_images
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error adding multiple face images: {str(e)}")
            return []
