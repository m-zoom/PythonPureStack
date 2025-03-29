import os
import uuid
import cv2
import numpy as np
from flask import render_template, request, redirect, url_for, flash, abort, jsonify
from app import app
from database import DatabaseManager
from face_recognition import FaceRecognition
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize face recognition
face_recognition = FaceRecognition()

# Configure upload folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static/uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/users')
def view_users():
    users = DatabaseManager.get_all_users()
    return render_template('view_users.html', users=users)

@app.route('/users/<int:user_id>')
def user_details(user_id):
    user = DatabaseManager.get_user(user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('view_users'))
    
    # Get face images
    face_images = DatabaseManager.get_user_face_images(user_id)
    
    # Get relatives
    relatives = DatabaseManager.get_user_relatives(user_id)
    
    # Get all users for the add relative form
    all_users = DatabaseManager.get_all_users()
    
    return render_template(
        'user_details.html', 
        user=user, 
        face_images=face_images,
        relatives=relatives,
        all_users=all_users
    )

@app.route('/users/add', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        details = request.form.get('details')
        relationship_type = request.form.get('relationship_type')
        
        if not name:
            flash('Name is required', 'danger')
            return render_template('add_user.html')
        
        try:
            user = DatabaseManager.add_user(
                name=name, 
                email=email, 
                phone=phone, 
                details=details, 
                relationship_type=relationship_type
            )
            
            # Handle face image uploads
            if 'face_image' in request.files:
                files = request.files.getlist('face_image')
                successful_uploads = 0
                failed_uploads = 0
                
                for file in files:
                    if file.filename != '' and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        unique_filename = f"{uuid.uuid4()}_{filename}"
                        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                        file.save(file_path)
                        
                        # Detect face in the uploaded image
                        faces, _ = face_recognition.detect_faces(file_path)
                        
                        if len(faces) > 0:
                            # Save the face image to the database with forward slashes
                            relative_path = 'static/uploads/' + unique_filename
                            DatabaseManager.add_face_image(user.id, relative_path)
                            successful_uploads += 1
                        else:
                            flash(f'No face detected in image: {filename}', 'warning')
                            os.remove(file_path)
                            failed_uploads += 1
                
                # Retrain the model if any faces were added
                if successful_uploads > 0:
                    face_recognition.train_model()
                    if successful_uploads == 1:
                        flash('Face image added successfully', 'success')
                    else:
                        flash(f'{successful_uploads} face images added successfully', 'success')
                
                if failed_uploads > 0:
                    flash(f'{failed_uploads} uploads failed because no faces were detected', 'warning')
            
            flash('User added successfully', 'success')
            return redirect(url_for('user_details', user_id=user.id))
        except Exception as e:
            flash(f'Error adding user: {str(e)}', 'danger')
            logging.error(f"Error in add_user: {str(e)}")
    
    return render_template('add_user.html')

@app.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
def edit_user(user_id):
    user = DatabaseManager.get_user(user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('view_users'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        details = request.form.get('details')
        relationship_type = request.form.get('relationship_type')
        
        if not name:
            flash('Name is required', 'danger')
            return render_template('add_user.html', user=user)
        
        try:
            success = DatabaseManager.update_user(
                user_id=user_id,
                name=name,
                email=email,
                phone=phone,
                details=details,
                relationship_type=relationship_type
            )
            
            if success:
                flash('User updated successfully', 'success')
                return redirect(url_for('user_details', user_id=user_id))
            else:
                flash('Error updating user', 'danger')
        except Exception as e:
            flash(f'Error updating user: {str(e)}', 'danger')
            logging.error(f"Error in edit_user: {str(e)}")
    
    return render_template('add_user.html', user=user)

@app.route('/users/<int:user_id>/delete', methods=['POST'])
def delete_user(user_id):
    success = DatabaseManager.delete_user(user_id)
    
    if success:
        # Retrain the model after deleting a user
        face_recognition.train_model()
        flash('User deleted successfully', 'success')
    else:
        flash('Error deleting user', 'danger')
    
    return redirect(url_for('view_users'))

@app.route('/users/<int:user_id>/add_face', methods=['POST'])
def add_face(user_id):
    user = DatabaseManager.get_user(user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('view_users'))
    
    if 'face_image' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('user_details', user_id=user_id))
    
    # Get the image type from the form
    image_type = request.form.get('image_type', 'profile')
    
    # Handle multiple file uploads
    files = request.files.getlist('face_image')
    
    if not files or files[0].filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('user_details', user_id=user_id))
    
    successful_uploads = 0
    failed_uploads = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            
            # Detect face in the uploaded image
            faces, _ = face_recognition.detect_faces(file_path)
            
            if len(faces) > 0:
                # Save the face image to the database with full relative path
                # For local development compatibility, include the 'static/' prefix
                # Use forward slashes for web URLs, regardless of the OS
                relative_path = 'static/uploads/' + unique_filename
                DatabaseManager.add_face_image(user_id, relative_path, image_type)
                successful_uploads += 1
            else:
                flash(f'No face detected in file: {filename}', 'warning')
                os.remove(file_path)
                failed_uploads += 1
        else:
            flash(f'Invalid file type: {file.filename}', 'warning')
            failed_uploads += 1
    
    # Retrain the model if any images were uploaded successfully
    if successful_uploads > 0:
        face_recognition.train_model()
        
        if successful_uploads == 1:
            flash('Face image added successfully', 'success')
        else:
            flash(f'{successful_uploads} face images added successfully', 'success')
    
    if failed_uploads > 0:
        flash(f'{failed_uploads} uploads failed', 'warning')
    
    return redirect(url_for('user_details', user_id=user_id))

@app.route('/users/<int:user_id>/delete_face/<int:image_id>', methods=['POST'])
def delete_face(user_id, image_id):
    success = DatabaseManager.delete_face_image(image_id)
    
    if success:
        # Retrain the model after deleting a face
        face_recognition.train_model()
        flash('Face image deleted successfully', 'success')
    else:
        flash('Error deleting face image', 'danger')
    
    return redirect(url_for('user_details', user_id=user_id))

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        return render_template('search.html')
    
    if request.method == 'POST':
        # Check if image is provided
        if 'face_image' not in request.files:
            flash('No file part', 'danger')
            return redirect(url_for('search'))
        
        file = request.files['face_image']
        
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(url_for('search'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            
            # Get confidence threshold from form (default to 40 - lower for better matching)
            confidence_threshold = int(request.form.get('confidence_threshold', 40))
            
            # Detect faces in the uploaded image with improved detection
            faces, _ = face_recognition.detect_faces(file_path)
            
            if len(faces) == 0:
                # Try again with more aggressive parameters
                logging.debug(f"No faces found initially, trying with more aggressive parameters")
                
                # Load image and try pre-processing
                image = cv2.imread(file_path)
                if image is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply histogram equalization for better contrast
                    gray = cv2.equalizeHist(gray)
                    
                    # Try detecting with more aggressive parameters
                    cascade_faces = face_recognition.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,  # Smaller scale factor for more detections
                        minNeighbors=2,    # Lower neighbors threshold
                        minSize=(20, 20),  # Smaller minimum face size
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    if len(cascade_faces) > 0:
                        faces = cascade_faces
                        logging.debug(f"Found {len(faces)} faces with aggressive parameters")
                
                # If still no faces found
                if len(faces) == 0:
                    flash('No face detected in the uploaded image. Try with a clearer photo or different lighting.', 'warning')
                    os.remove(file_path)
                    return redirect(url_for('search'))
            
            # Search for matching faces with enhanced matching
            match_results = face_recognition.find_matching_faces(
                file_path, 
                confidence_threshold=confidence_threshold
            )
            
            # Extract the different match categories
            primary_matches = match_results.get('primary_matches', [])
            similar_people = match_results.get('similar_people', [])
            relatives = match_results.get('relatives', [])
            
            # Store the path with 'static/' prefix for local development compatibility
            # Use forward slashes for web URLs, regardless of the OS
            search_image = 'static/uploads/' + unique_filename
            
            return render_template('search.html', 
                                 primary_matches=primary_matches,
                                 similar_people=similar_people,
                                 relatives=relatives,
                                 search_image=search_image, 
                                 confidence_threshold=confidence_threshold)
        else:
            flash('Invalid file type', 'danger')
            return redirect(url_for('search'))

@app.route('/api/webcam_capture', methods=['POST'])
def webcam_capture():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400
    
    # Save the captured image
    filename = f"webcam_capture_{uuid.uuid4()}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Detect faces
    faces, _ = face_recognition.detect_faces(file_path)
    
    if len(faces) == 0:
        os.remove(file_path)
        return jsonify({'error': 'No face detected'}), 400
    
    # For client-side display, keep the full path with forward slashes for web URLs
    relative_path = 'static/uploads/' + filename
    
    return jsonify({
        'success': True,
        'file_path': relative_path,
        'faces': len(faces)
    })

@app.route('/train_model', methods=['POST'])
def train_model():
    success = face_recognition.train_model()
    
    if success:
        flash('Model trained successfully', 'success')
    else:
        flash('Error training model or not enough face data', 'danger')
    
    return redirect(url_for('index'))

@app.route('/users/<int:user_id>/add_relative', methods=['POST'])
def add_relative(user_id):
    """Add a relative to a user"""
    user = DatabaseManager.get_user(user_id)
    if not user:
        flash('User not found', 'danger')
        return redirect(url_for('view_users'))
    
    relative_id = request.form.get('relative_id')
    relationship_type = request.form.get('relationship_type')
    
    if not relative_id:
        flash('No relative selected', 'danger')
        return redirect(url_for('user_details', user_id=user_id))
    
    # Convert to integer
    try:
        relative_id = int(relative_id)
    except ValueError:
        flash('Invalid relative ID', 'danger')
        return redirect(url_for('user_details', user_id=user_id))
    
    # Check if trying to add self as relative
    if relative_id == user_id:
        flash('Cannot add yourself as a relative', 'danger')
        return redirect(url_for('user_details', user_id=user_id))
    
    # Add the relationship
    success = DatabaseManager.add_relative(user_id, relative_id, relationship_type)
    
    if success:
        relative = DatabaseManager.get_user(relative_id)
        flash(f'Successfully added {relative.name} as a relative', 'success')
    else:
        flash('Error adding relative', 'danger')
    
    return redirect(url_for('user_details', user_id=user_id))

@app.route('/users/<int:user_id>/remove_relative/<int:relative_id>', methods=['POST'])
def remove_relative(user_id, relative_id):
    """Remove a relative from a user"""
    user = DatabaseManager.get_user(user_id)
    relative = DatabaseManager.get_user(relative_id)
    
    if not user or not relative:
        flash('User or relative not found', 'danger')
        return redirect(url_for('view_users'))
    
    success = DatabaseManager.remove_relative(user_id, relative_id)
    
    if success:
        flash(f'Successfully removed {relative.name} as a relative', 'success')
    else:
        flash('Error removing relative', 'danger')
    
    return redirect(url_for('user_details', user_id=user_id))

@app.route('/video_analysis', methods=['GET', 'POST'])
def video_analysis():
    if request.method == 'GET':
        return render_template('video_analysis.html')
    
    if request.method == 'POST':
        # Check if no file is provided
        if 'video_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['video_file']
        
        # Check if filename is empty
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        # Check if file is a valid video
        valid_extensions = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
        if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in valid_extensions:
            flash('Invalid video format. Supported formats: mp4, avi, mov, mkv, wmv', 'danger')
            return redirect(request.url)
        
        try:
            # Save the video file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(file_path)
            
            # Get confidence threshold from form
            confidence_threshold = int(request.form.get('confidence_threshold', 50))
            
            # Get sample rate from form (process every Nth frame)
            sample_rate = int(request.form.get('sample_rate', 15))
            
            # Check if the file is a valid video
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                cap.release()
                os.remove(file_path)
                flash('The uploaded file is not a valid video', 'danger')
                return redirect(request.url)
            
            # Get video info
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            # If video is too long, warn the user
            if duration > 300:  # 5 minutes
                os.remove(file_path)
                flash('Video is too long. Please upload a video under 5 minutes for better performance', 'warning')
                return redirect(request.url)
            
            # Analyze the video
            flash('Video uploaded successfully. Processing...', 'info')
            analysis_results = face_recognition.analyze_video(
                file_path, 
                confidence_threshold=confidence_threshold,
                sample_rate=sample_rate
            )
            
            # Remove the video file after processing to save space
            os.remove(file_path)
            
            if not analysis_results or not analysis_results.get('summary'):
                flash('No faces were recognized in the video', 'warning')
                return render_template('video_analysis.html', no_results=True)
            
            return render_template(
                'video_analysis.html',
                analysis=analysis_results,
                confidence_threshold=confidence_threshold
            )
            
        except Exception as e:
            logging.error(f"Error in video analysis: {str(e)}")
            flash(f'Error processing video: {str(e)}', 'danger')
            return redirect(request.url)
