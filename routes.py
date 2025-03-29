import os
import uuid
from flask import render_template, request, redirect, url_for, flash, abort, jsonify
from app import app
from database import DatabaseManager
from face_recognition import FaceRecognition
import logging
from werkzeug.utils import secure_filename

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
    
    face_images = DatabaseManager.get_user_face_images(user_id)
    return render_template('user_details.html', user=user, face_images=face_images)

@app.route('/users/add', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        details = request.form.get('details')
        
        if not name:
            flash('Name is required', 'danger')
            return render_template('add_user.html')
        
        try:
            user = DatabaseManager.add_user(name=name, email=email, phone=phone, details=details)
            
            # Handle face image upload
            if 'face_image' in request.files:
                file = request.files['face_image']
                if file.filename != '' and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                    file.save(file_path)
                    
                    # Detect face in the uploaded image
                    faces, _ = face_recognition.detect_faces(file_path)
                    
                    if len(faces) > 0:
                        # Save the face image to the database
                        relative_path = os.path.join('static/uploads', unique_filename)
                        DatabaseManager.add_face_image(user.id, relative_path)
                        
                        # Retrain the model if a face was added
                        face_recognition.train_model()
                    else:
                        flash('No face detected in the uploaded image', 'warning')
                        os.remove(file_path)
            
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
        
        if not name:
            flash('Name is required', 'danger')
            return render_template('add_user.html', user=user)
        
        try:
            success = DatabaseManager.update_user(
                user_id=user_id,
                name=name,
                email=email,
                phone=phone,
                details=details
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
    
    file = request.files['face_image']
    
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('user_details', user_id=user_id))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(file_path)
        
        # Detect face in the uploaded image
        faces, _ = face_recognition.detect_faces(file_path)
        
        if len(faces) > 0:
            # Save the face image to the database
            relative_path = os.path.join('static/uploads', unique_filename)
            DatabaseManager.add_face_image(user_id, relative_path)
            
            # Retrain the model
            face_recognition.train_model()
            
            flash('Face image added successfully', 'success')
        else:
            flash('No face detected in the uploaded image', 'warning')
            os.remove(file_path)
    else:
        flash('Invalid file type', 'danger')
    
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
            
            # Get confidence threshold from form
            confidence_threshold = int(request.form.get('confidence_threshold', 50))
            
            # Detect faces in the uploaded image
            faces, _ = face_recognition.detect_faces(file_path)
            
            if len(faces) == 0:
                flash('No face detected in the uploaded image', 'warning')
                os.remove(file_path)
                return redirect(url_for('search'))
            
            # Search for matching faces
            matches = face_recognition.find_matching_faces(
                file_path, 
                confidence_threshold=confidence_threshold
            )
            
            # Keep the uploaded file for display
            search_image = os.path.join('static/uploads', unique_filename)
            
            return render_template('search.html', 
                                 matches=matches, 
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
    
    relative_path = os.path.join('static/uploads', filename)
    
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
