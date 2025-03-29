# Federal Biometrics of Nigeria (FBN)

Developer: ADEBAYO MUSTURPHA KEWULERE

TikTok: @adebayo_musturpha  
Facebook: Musturhs Adebayo  
X: @mzoom_olabewa  

## Project Description
This is a sophisticated biometric face recognition system built for the Federal Biometrics of Nigeria (FBN). Using Python and OpenCV, the system can detect and identify individuals by matching faces against a stored database. The system provides comprehensive features for face detection, recognition, user management, and video analysis capabilities for security and identification purposes.

## Tech Stack
- **Backend Framework**: Flask
- **Database**: SQLite with SQLAlchemy ORM
- **Face Recognition**: OpenCV with LBPH Face Recognizer
- **Frontend**: HTML, CSS, JavaScript with Bootstrap
- **Server**: Gunicorn
- **Image Processing**: NumPy, OpenCV
- **Development Tools**: Python 3.x

## Key Features
- Face detection and recognition
- User management system
- Real-time webcam capture
- Video analysis with face tracking and timestamps
- Confidence-based matching
- Database storage for face images
- Training interface for the recognition model

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/federal-biometrics-nigeria.git
   cd federal-biometrics-nigeria
   ```

2. Install required packages:
   ```
   pip install flask flask-sqlalchemy sqlalchemy opencv-contrib-python numpy gunicorn werkzeug
   ```

3. Ensure the following directory structure exists:
   ```
   static/
   ├── haarcascades/
   ├── models/
   ├── uploads/
   └── css/
   ```

4. Initialize the database:
   ```
   python -c "from app import db; db.create_all()"
   ```

5. Run the application:
   ```
   python main.py
   ```
   
   For production:
   ```
   gunicorn --bind 0.0.0.0:5000 main:app
   ```
   
6. The application will be available at http://0.0.0.0:5000

## Usage

1. Access the web interface
2. Add users with their face images
3. Train the recognition model
4. Use the search feature to find matching faces
5. Analyze videos for face detection and recognition with timestamps

## Notes

- The system uses LBPH (Local Binary Patterns Histograms) for face recognition
- Face detection uses Haar Cascade Classifiers
- Images are automatically processed and standardized before storage
- The confidence threshold can be adjusted for matching accuracy
- For optimal recognition performance, multiple face images per user are recommended
- Video analysis provides chronological detection with timestamps
