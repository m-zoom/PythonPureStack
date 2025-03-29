import os
import logging
from flask import Flask
from markupsafe import Markup
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Custom Jinja2 filters
def nl2br(value):
    """Convert newlines to <br> tags."""
    if value:
        return Markup(value.replace('\n', '<br>'))
    return value

# Setup the database
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key_for_development")

# Register Jinja2 filters
app.jinja_env.filters['nl2br'] = nl2br

# Configure the database (PostgreSQL or SQLite fallback)
if os.environ.get('DATABASE_URL'):
    # Use PostgreSQL database from environment variable
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get('DATABASE_URL')
else:
    # Fallback to SQLite for development
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///facial_recognition.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize the database
db.init_app(app)

# Create the uploads directory if it doesn't exist
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

# Create the models directory if it doesn't exist
if not os.path.exists('static/models'):
    os.makedirs('static/models')

with app.app_context():
    # Import models
    import models  # noqa: F401
    
    # Create all tables
    db.create_all()

# Import routes after app is created to avoid circular imports
from routes import *
