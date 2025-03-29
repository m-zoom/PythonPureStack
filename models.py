from app import db
from datetime import datetime
from sqlalchemy import Table, Column, Integer, ForeignKey

# Association table for relatives relationship (many-to-many)
user_relatives = Table('user_relatives', db.Model.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('relative_id', Integer, ForeignKey('users.id'))
)

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=True, unique=True)
    phone = db.Column(db.String(20), nullable=True)
    details = db.Column(db.Text, nullable=True)
    
    # Add relationship type for relatives
    relationship_type = db.Column(db.String(50), nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship with FaceImage
    face_images = db.relationship('FaceImage', backref='user', lazy=True, cascade="all, delete-orphan")
    
    # Self-referential relationship for relatives
    relatives = db.relationship(
        'User', 
        secondary=user_relatives,
        primaryjoin=(user_relatives.c.user_id == id),
        secondaryjoin=(user_relatives.c.relative_id == id),
        backref=db.backref('related_to', lazy='dynamic'),
        lazy='dynamic'
    )
    
    def add_relative(self, relative, relationship_type=None):
        """Add a relative to this user"""
        if relative not in self.relatives:
            self.relatives.append(relative)
            relative.relatives.append(self)
            return True
        return False
    
    def remove_relative(self, relative):
        """Remove a relative from this user"""
        if relative in self.relatives:
            self.relatives.remove(relative)
            relative.relatives.remove(self)
            return True
        return False
    
    def get_relatives(self):
        """Get all relatives for this user"""
        return self.relatives.all()
    
    def __repr__(self):
        return f'<User {self.name}>'

class FaceImage(db.Model):
    __tablename__ = 'face_images'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    
    # Additional metadata for images
    image_type = db.Column(db.String(50), default='profile')  # profile, side, various angles
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<FaceImage {self.id} for User {self.user_id}>'

class RelationshipType(db.Model):
    """Table to store relationship types for better organization"""
    __tablename__ = 'relationship_types'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.String(255), nullable=True)
    
    def __repr__(self):
        return f'<RelationshipType {self.name}>'
