Face Recognition System
Developer: ADEBAYO MUSTURPHA KEWULERE
Tiktok: @adebayo_musturpha
Facebook: Musturhs Adebayo
x: @mzoom_olabewa

Project Description
This is a sophisticated face recognition system built with Python and OpenCV that can detect and identify individuals by matching faces against a stored database. The system provides features for face detection, recognition, user management, and video analysis capabilities.

Tech Stack
Backend Framework: Flask
Database: SQLite with SQLAlchemy ORM
Face Recognition: OpenCV with LBPH Face Recognizer
Frontend: HTML, CSS, JavaScript
Server: Gunicorn
Image Processing: NumPy, OpenCV
Development Tools: Python 3.x
Key Features
Face detection and recognition
User management system
Real-time webcam capture
Video analysis with face tracking
Confidence-based matching
Database storage for face images
Training interface for the recognition model
Setup Instructions
Clone the repository

Install required packages:

Dependencies
flask
opencv-python
numpy
gunicorn
flask-sqlalchemy
Ensure the following directory structure exists:
static/
├── haarcascades/
├── models/
└── uploads/
Initialize the database:
python3 -c "from app import db; db.create_all()"
Run the application:
gunicorn --bind 0.0.0.0:5000 main:app
The application will be available at http://0.0.0.0:5000

Usage
Access the web interface
Add users with their face images
Train the recognition model
Use the search feature to find matching faces
Analyze videos for face detection and recognition
Notes
The system uses LBPH (Local Binary Patterns Histograms) for face recognition
Face detection uses Haar Cascade Classifiers
Images are automatically processed and standardized before storage
The confidence threshold can be adjusted for matching accuracy
