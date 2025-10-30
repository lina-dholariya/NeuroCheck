from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
from datetime import datetime, timezone
import uuid
import logging
from ml_models import mental_health_predictor
from llm_service import llm_service
from dotenv import load_dotenv, dotenv_values
from pathlib import Path

# ✅ Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Ensure os is imported before using it
import os

# ✅ Load .env explicitly from the same folder as app.py
env_path = Path(__file__).with_name('.env')
loaded = load_dotenv(dotenv_path=env_path, override=True)

if not loaded:
    # Fallback 1: try current working directory
    cwd_env = Path.cwd() / '.env'
    if cwd_env.exists():
        load_dotenv(dotenv_path=cwd_env, override=True)
    # Fallback 2: load whatever might be on sys.path
    load_dotenv(override=True)

logger.info(f"Env load attempt: file={env_path} exists={env_path.exists()}")

# ✅ Extra safety: Manually load values if still missing
if not os.getenv('MONGODB_URI') and env_path.exists():
    try:
        env_vals = dotenv_values(dotenv_path=env_path)
        for k, v in env_vals.items():
            if k and v is not None and not os.getenv(k):
                os.environ[k] = str(v)
    except Exception as e:
        logger.error(f"Manual .env load failed: {e}")

# ✅ Debug check — confirm values loaded
uri_preview = os.getenv("MONGODB_URI")[:60] + "..." if os.getenv("MONGODB_URI") else "MISSING"
logger.info(f"MONGODB_URI (preview): {uri_preview}")
logger.info(f"MONGODB_DB: {os.getenv('MONGODB_DB')}")


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'change-me-in-production')
CORS(app)
app.config['SESSION_TYPE'] = 'filesystem'

# MongoDB Atlas connection via env
MONGODB_URI = os.getenv('MONGODB_URI', '').strip()
# Log masked URI presence for troubleshooting
try:
    _masked = 'set' if MONGODB_URI else 'missing'
    logger.info(f"MONGODB_URI status: {_masked}")
except Exception:
    pass

def _mask_mongo_uri(uri: str) -> str:
    try:
        if not uri:
            return ''
        # mask password between '//' and '@'
        if '//' in uri and '@' in uri:
            prefix, rest = uri.split('//', 1)
            creds, host = rest.split('@', 1)
            if ':' in creds:
                user, pwd = creds.split(':', 1)
                return f"{prefix}//{user}:***@{host}"
        return uri
    except Exception:
        return '***'

# Fallback in-memory database
in_memory_users = {}
in_memory_quiz_history = {}
in_memory_chat_history = {}

# File storage for persistence
USERS_FILE = 'users_data.json'
QUIZ_HISTORY_FILE = 'quiz_history.json'

def load_users_from_file():
    """Load users from JSON file"""
    global in_memory_users
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                in_memory_users = json.load(f)
            logger.info(f"Loaded {len(in_memory_users)} users from file")
    except Exception as e:
        logger.error(f"Error loading users from file: {e}")
        # Auto-repair malformed file by resetting to empty JSON
        in_memory_users = {}
        try:
            with open(USERS_FILE, 'w') as f:
                json.dump({}, f)
            logger.info("Repaired users_data.json to empty JSON {}")
        except Exception as we:
            logger.error(f"Failed to repair users_data.json: {we}")

def save_users_to_file():
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(in_memory_users, f, indent=2)
        logger.info(f"Saved {len(in_memory_users)} users to file")
    except Exception as e:
        logger.error(f"Error saving users to file: {e}")

def load_quiz_history_from_file():
    """Load quiz history from JSON file"""
    global in_memory_quiz_history
    try:
        if os.path.exists(QUIZ_HISTORY_FILE):
            with open(QUIZ_HISTORY_FILE, 'r') as f:
                in_memory_quiz_history = json.load(f)
            logger.info(f"Loaded quiz history for {len(in_memory_quiz_history)} users from file")
    except Exception as e:
        logger.error(f"Error loading quiz history from file: {e}")
        # Auto-repair malformed quiz history file
        in_memory_quiz_history = {}
        try:
            with open(QUIZ_HISTORY_FILE, 'w') as f:
                json.dump({}, f)
            logger.info("Repaired quiz_history.json to empty JSON {}")
        except Exception as we:
            logger.error(f"Failed to repair quiz_history.json: {we}")

def save_quiz_history_to_file():
    """Save quiz history to JSON file"""
    try:
        with open(QUIZ_HISTORY_FILE, 'w') as f:
            json.dump(in_memory_quiz_history, f, indent=2)
        logger.info(f"Saved quiz history for {len(in_memory_quiz_history)} users to file")
    except Exception as e:
        logger.error(f"Error saving quiz history to file: {e}")

def sync_users_from_mongodb():
    """Sync users from MongoDB to in-memory database"""
    global in_memory_users
    try:
        if use_mongodb and users_collection is not None:
            existing_users = users_collection.find({})
            for user in existing_users:
                in_memory_users[user['email']] = user
            logger.info(f"Synced {len(in_memory_users)} users from MongoDB to in-memory database")
            # Save synced users to file
            save_users_to_file()
    except Exception as e:
        logger.error(f"Error syncing users from MongoDB: {e}")

try:
    # Configure MongoDB client with proper settings
    client = None
    if MONGODB_URI:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=10000,  # 10 second timeout
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
            maxPoolSize=10
        )
    
    # Test the connection
    if client is not None:
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        # Use a database in your cluster (default name if not provided)
        db_name = os.getenv('MONGODB_DB', 'NeuroCheck')
        db = client[db_name]
        users_collection = db.users
        # Quiz history collection
        quiz_history_collection = db.quiz_history
        # Chat history collection
        chat_history_collection = db.chat_history
        # Seed the database so it appears in Atlas immediately
        try:
            seed_collection = db.seed
            seed_collection.update_one(
                {'_id': 'seed_neurocheck'},
                {'$set': {
                    'mental': 'health',
                    'created_at': datetime.now(timezone.utc).isoformat()
                }},
                upsert=True
            )
            logger.info("Database seeded with {'mental':'health'} document")
        except Exception as se:
            logger.error(f"Error seeding database: {se}")
        use_mongodb = True
    else:
        raise Exception('MONGODB_URI not configured')
    
    # Sync existing users from MongoDB to in-memory database
    sync_users_from_mongodb()
        
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    logger.info("Using in-memory database as fallback")
    client = None
    db = None
    users_collection = None
    use_mongodb = False
    quiz_history_collection = None
    chat_history_collection = None

@app.route('/debug/mongo')
def debug_mongo():
    """Return MongoDB connection diagnostics (masked)."""
    info = {
        'env_loaded': bool(MONGODB_URI),
        'mongodb_uri_present': bool(MONGODB_URI),
        'mongodb_uri_masked': _mask_mongo_uri(MONGODB_URI),
        'mongodb_db': os.getenv('MONGODB_DB', 'NeuroCheck'),
        'connected': False,
        'ping_ok': False,
        'error': None
    }
    try:
        if not MONGODB_URI:
            raise RuntimeError('MONGODB_URI missing in environment')
        _client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _client.admin.command('ping')
        info['connected'] = True
        info['ping_ok'] = True
    except Exception as e:
        info['error'] = str(e)
    return jsonify(info)

def find_user_by_email(email):
    """Find user by email from either MongoDB or in-memory database"""
    # Always check in-memory database first
    user = in_memory_users.get(email)
    if user:
        return user
    
    # If not found in memory and MongoDB is available, check MongoDB
    if use_mongodb and users_collection is not None:
        try:
            user = users_collection.find_one({'email': email})
            if user:
                # Sync to in-memory database
                in_memory_users[email] = user
                return user
        except Exception as e:
            logger.error(f"Error finding user in MongoDB: {e}")
            # Try to sync users again in case connection was restored
            sync_users_from_mongodb()
            return in_memory_users.get(email)
    
    return None

def create_user(user_data):
    """Create user in both MongoDB and in-memory database"""
    user_id = str(uuid.uuid4())
    user_data['_id'] = user_id
    
    # Always add to in-memory database
    in_memory_users[user_data['email']] = user_data
    
    # Save to file for persistence
    save_users_to_file()
    
    # Try to add to MongoDB if available
    if use_mongodb and users_collection is not None:
        try:
            result = users_collection.insert_one(user_data)
            logger.info(f"User created in MongoDB with ID: {result.inserted_id}")
        except Exception as e:
            logger.error(f"Error creating user in MongoDB: {e}")
            # Continue with in-memory only
    
    return user_id

def save_quiz_result(user_id, result_data):
    """Save quiz result to memory, file, and MongoDB if available"""
    if user_id not in in_memory_quiz_history:
        in_memory_quiz_history[user_id] = []
    
    quiz_result = {
        'user_id': user_id,
        'date': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
        'severity': result_data['severity_level'],
        'confidence': result_data['ensemble_confidence'],
        'recommendation': result_data.get('recommendation', 'No specific recommendation'),
        'timestamp': datetime.now().isoformat()
    }
    
    in_memory_quiz_history[user_id].append(quiz_result)
    
    # Save to file for persistence
    save_quiz_history_to_file()
    
    # Save to MongoDB if available
    try:
        if use_mongodb and 'quiz_history_collection' in globals() and quiz_history_collection is not None:
            quiz_history_collection.insert_one(quiz_result)
            logger.info(f"Quiz result saved to MongoDB for user {user_id}")
    except Exception as e:
        logger.error(f"Error saving quiz result to MongoDB: {e}")
    
    logger.info(f"Quiz result saved for user {user_id}")

def get_quiz_history(user_id):
    """Get quiz history for a user from memory or MongoDB"""
    history = []
    
    # Try to get from MongoDB first if available
    if use_mongodb and quiz_history_collection is not None:
        try:
            cursor = quiz_history_collection.find({'user_id': user_id}).sort('timestamp', 1)
            history = list(cursor)
        except Exception as e:
            logger.error(f"Error getting quiz history from MongoDB: {e}")
    
    # If no results from MongoDB or it's not available, use in-memory
    if not history:
        history = in_memory_quiz_history.get(user_id, [])
    
    # Ensure all entries have required fields
    validated_history = []
    for entry in history:
        try:
            # Ensure entry is a dictionary
            if not isinstance(entry, dict):
                logger.error(f"Invalid history entry format: {type(entry)}")
                continue
                
            # Validate required fields
            if 'date' not in entry and 'timestamp' in entry:
                # Convert timestamp to date string if missing
                entry['date'] = entry['timestamp'].strftime('%Y-%m-%d') if hasattr(entry['timestamp'], 'strftime') else str(entry['timestamp'])
            
            validated_history.append(entry)
        except Exception as e:
            logger.error(f"Error validating history entry: {e}")
    
    return validated_history

def analyze_progress(history):
    """Analyze user's progress over time and generate a summary"""
    if not history or len(history) < 2:
        return "Complete more quizzes to see your progress analysis."
    
    # Map severity levels to numeric values for comparison
    severity_map = {
        'Minimal': 1,
        'Mild': 2,
        'Moderate': 3,
        'Moderately Severe': 4,
        'Severe': 5
    }
    
    # Convert severity text to numeric values with error handling
    numeric_severity = []
    for entry in history:
        try:
            severity = entry.get('severity', '')
            if severity in severity_map:
                numeric_severity.append(severity_map[severity])
            else:
                # Default to moderate if severity not recognized
                numeric_severity.append(3)
        except Exception as e:
            logger.error(f"Error processing severity in analyze_progress: {e}")
            # Default to moderate
            numeric_severity.append(3)
    
    if not numeric_severity:
        return "Unable to analyze progress. Please try again later."
    
    # Get the first and most recent severity levels
    first_severity = numeric_severity[0]
    recent_severity = numeric_severity[-1]
    
    # Calculate the trend (average of last 3 entries vs average of first 3 entries)
    recent_trend = sum(numeric_severity[-min(3, len(numeric_severity)):]) / min(3, len(numeric_severity))
    initial_trend = sum(numeric_severity[:min(3, len(numeric_severity))]) / min(3, len(numeric_severity))
    
    # Determine if improving, worsening, or stable
    diff = recent_trend - initial_trend
    
    if diff <= -0.5:
        return "Your mental health appears to be improving. Keep up the good work!"
    elif diff >= 0.5:
        return "Your stress levels seem to have increased. Consider using some of the coping strategies suggested."
    else:
        return "Your mental health has been relatively stable. Continue monitoring and practicing self-care."

def get_chat_history(user_id, limit: int = 20):
    """Get recent chat history for a user as a list of {role, content} dicts."""
    history = []
    try:
        if use_mongodb and 'chat_history_collection' in globals() and chat_history_collection is not None:
            cursor = chat_history_collection.find({'user_id': user_id}).sort('timestamp', -1).limit(limit)
            history = [{
                'role': doc.get('role', 'assistant'),
                'content': doc.get('content', '')
            } for doc in cursor]
            history.reverse()  # oldest first for chat models
        else:
            history = in_memory_chat_history.get(user_id, [])[-limit:]
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        history = in_memory_chat_history.get(user_id, [])[-limit:]
    return history

def save_chat_message(user_id: str, role: str, content: str):
    """Persist a single chat message for a user to memory, file (optional), and MongoDB if available."""
    try:
        if user_id not in in_memory_chat_history:
            in_memory_chat_history[user_id] = []
        in_memory_chat_history[user_id].append({'role': role, 'content': content, 'timestamp': datetime.now().isoformat()})
        # Do not store to a file to avoid growth; MongoDB is the primary persistent store
        if use_mongodb and 'chat_history_collection' in globals() and chat_history_collection is not None:
            chat_history_collection.insert_one({
                'user_id': user_id,
                'role': role,
                'content': content,
                'timestamp': datetime.now(timezone.utc)
            })
    except Exception as e:
        logger.error(f"Error saving chat message: {e}")

def get_user_by_id(user_id):
    """Get user by ID from either MongoDB or in-memory database"""
    # Check in-memory database first
    for user in in_memory_users.values():
        if user.get('_id') == user_id:
            return user
    
    # If not found in memory and MongoDB is available, check MongoDB
    if use_mongodb and users_collection is not None:
        try:
            user = users_collection.find_one({'_id': user_id})
            if user:
                # Sync to in-memory database
                in_memory_users[user['email']] = user
                return user
        except Exception as e:
            logger.error(f"Error finding user by ID in MongoDB: {e}")
            # Try to sync users again in case connection was restored
            sync_users_from_mongodb()
            for user in in_memory_users.values():
                if user.get('_id') == user_id:
                    return user
    
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress')
def progress():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('progress.html')

@app.route('/api/progress')
def api_progress():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    user_id = session['user_id']
    history = get_quiz_history(user_id)
    
    if not history:
        return jsonify({'success': False, 'message': 'No quiz history found'})
    
    # Format history data for frontend
    formatted_history = []
    for entry in history:
        try:
            formatted_entry = {
                'date': entry.get('date', ''),
                'severity': entry.get('severity_level', ''),
                'confidence': entry.get('ensemble_confidence', 0.0)
            }
            formatted_history.append(formatted_entry)
        except Exception as e:
            logger.error(f"Error formatting history entry: {e}")
    
    if not formatted_history:
        return jsonify({'success': False, 'message': 'Error formatting quiz history'})
    
    # Generate progress summary
    summary = analyze_progress(formatted_history)
    
    return jsonify({
        'success': True,
        'history': formatted_history,
        'summary': summary
    })

@app.route('/sync-users')
def sync_users():
    """Manual endpoint to sync users from MongoDB"""
    try:
        sync_users_from_mongodb()
        return jsonify({
            'success': True, 
            'message': f'Synced {len(in_memory_users)} users from MongoDB',
            'users': list(in_memory_users.keys())
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error syncing users: {e}'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'Invalid request data'})
            
            email = data.get('email')
            password = data.get('password')
            
            if not email or not password:
                return jsonify({'success': False, 'message': 'Email and password are required'})
            
            # Try to sync users first if MongoDB is available
            if use_mongodb and users_collection is not None:
                try:
                    sync_users_from_mongodb()
                except Exception as e:
                    logger.error(f"Error syncing users during login: {e}")
            
            # Find user in database
            user = find_user_by_email(email)
            
            if user and check_password_hash(user['password'], password):
                session['user_id'] = str(user['_id'])
                session['email'] = user['email']
                logger.info(f"User {email} logged in successfully")
                logger.info(f"Session after login: {session}")
                return jsonify({'success': True, 'message': 'Login successful', 'redirect': '/dashboard'})
            else:
                logger.warning(f"Failed login attempt for email: {email}")
                # Log available users for debugging
                logger.info(f"Available users in memory: {list(in_memory_users.keys())}")
                return jsonify({'success': False, 'message': 'Invalid email or password'})
                
        except Exception as e:
            logger.error(f"Error during login: {e}")
            return jsonify({'success': False, 'message': 'An error occurred. Please try again.'})
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'Invalid request data'})
            
            name = data.get('name')
            email = data.get('email')
            password = data.get('password')
            birthday = data.get('birthday')
            age = data.get('age')
            
            if not name or not email or not password:
                return jsonify({'success': False, 'message': 'All fields are required'})

            # Backend DOB validation: enforce 1950-01-01..today and recompute age server-side
            try:
                if birthday:
                    dob = datetime.strptime(birthday, '%Y-%m-%d')
                    min_dob = datetime(1950, 1, 1)
                    max_dob = datetime.now()
                    if dob < min_dob or dob > max_dob:
                        return jsonify({'success': False, 'message': 'Please enter a valid date of birth'})
                    # compute age server-side
                    today = datetime.now()
                    computed_age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                    age = str(computed_age)
            except Exception:
                return jsonify({'success': False, 'message': 'Please enter a valid date of birth'})
            
            # Check if user already exists
            existing_user = find_user_by_email(email)
            if existing_user:
                return jsonify({'success': False, 'message': 'Email already registered'})
            
            # Hash password and create user
            hashed_password = generate_password_hash(password)
            user_data = {
                'name': name,
                'email': email,
                'password': hashed_password,
                'birthday': birthday,
                'age': age,
                'created_at': datetime.now(timezone.utc),
                'user_id': str(uuid.uuid4())
            }
            
            result_id = create_user(user_data)
            
            if result_id:
                session['user_id'] = str(result_id)
                session['email'] = email
                logger.info(f"User {email} registered successfully")
                return jsonify({'success': True, 'message': 'Registration successful', 'redirect': '/dashboard'})
            else:
                return jsonify({'success': False, 'message': 'Registration failed'})
                
        except Exception as e:
            logger.error(f"Error during signup: {e}")
            return jsonify({'success': False, 'message': 'An error occurred. Please try again.'})
    
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    logger.info(f"Dashboard access attempt - Session: {session}")
    if 'user_id' not in session:
        logger.warning("No user_id in session, redirecting to login")
        return redirect(url_for('login'))
    
    try:
        user = get_user_by_id(session['user_id'])
        if not user:
            logger.warning(f"User not found for ID: {session['user_id']}, clearing session")
            session.clear()
            return redirect(url_for('login'))
        
        logger.info(f"Dashboard access successful for user: {user['email']}")
        
        # Get quiz history for the user
        quiz_history = get_quiz_history(session['user_id'])
        
        return render_template('dashboard.html', user=user, quiz_history=quiz_history)
    except Exception as e:
        logger.error(f"Error accessing dashboard: {e}")
        session.clear()
        return redirect(url_for('login'))

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    try:
        user = get_user_by_id(session['user_id'])
        if not user:
            session.clear()
            return redirect(url_for('login'))
        return render_template('profile.html', user=user)
    except Exception as e:
        logger.error(f"Error accessing profile: {e}")
        session.clear()
        return redirect(url_for('login'))
    
@app.route('/chat')
def chat_page():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    try:
        user = get_user_by_id(session['user_id'])
        if not user:
            session.clear()
            return redirect(url_for('login'))
        return render_template('chat.html', user=user)
    except Exception as e:
        logger.error(f"Error rendering chat page: {e}")
        return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/user')
def get_user():
    if 'user_id' in session:
        try:
            user = get_user_by_id(session['user_id'])
            if user:
                return jsonify({
                    'success': True,
                    'user': {
                        'name': user['name'],
                        'email': user['email']
                    }
                })
        except Exception as e:
            logger.error(f"Error getting user: {e}")
    
    return jsonify({'success': False, 'message': 'Not logged in'})
@app.route('/quiz')
def quiz():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('quiz.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Store quiz data in session and redirect to show-result"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Extract all form data
        user_data = {}
        for key in request.form:
            user_data[key] = request.form.get(key)
        
        # Validate that all required fields are present
        required_fields = ['anxiety1', 'anxiety2', 'depression1', 'depression2']
        missing_fields = [field for field in required_fields if field not in user_data or not user_data[field]]
        
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return render_template('quiz.html', error="Please fill out all required fields")
        
        if not user_data:
            logger.error("No form data received")
            return render_template('quiz.html', error="Please fill out all fields")
        
        # Store the quiz data in session
        # Use a try-except block to handle potential session issues
        try:
            session['quiz_data'] = user_data
            session.modified = True  # Ensure session is saved
        except Exception as session_error:
            logger.error(f"Error storing data in session: {session_error}")
            return render_template('quiz.html', error="Session error. Please try again.")
        
        # Redirect to the show-result route
        return redirect(url_for('show_result'))
    except Exception as e:
        logger.error(f"Error processing quiz: {e}")
        return render_template('quiz.html', error="An error occurred. Please try again.")

@app.route('/show-result')
def show_result():
    """Process quiz data and show results"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'quiz_data' not in session:
        # If no quiz data in session, redirect to quiz page
        logger.warning("No quiz data found in session")
        return redirect(url_for('quiz'))
    
    try:
        # Get the quiz data from session
        user_data = session.get('quiz_data', {})
        
        # Make a copy of the data to avoid modifying the session directly
        user_data_copy = user_data.copy()
        
        # Validate that all required fields are present
        required_fields = ['anxiety1', 'anxiety2', 'depression1', 'depression2']
        for field in required_fields:
            if field not in user_data_copy or not user_data_copy[field]:
                logger.error(f"Missing required field: {field}")
                return redirect(url_for('quiz'))
        
        # Get mental health prediction using ML models
        prediction_result = mental_health_predictor.predict_mental_health(user_data_copy)
        
        # Analyze risk factors
        risk_factors = mental_health_predictor.analyze_risk_factors(user_data_copy)
        
        # Get treatment recommendation
        treatment_data = mental_health_predictor.get_treatment_recommendation(
            prediction_result['severity_level'], 
            prediction_result['ensemble_confidence']
        )
        
        # Create visualization charts
        risk_chart = mental_health_predictor.create_risk_factor_chart(risk_factors)
        treatment_chart = mental_health_predictor.create_treatment_chart(treatment_data)
        
        # Calculate composite score for backward compatibility
        try:
            composite_score = sum([
                int(user_data_copy.get('anxiety1', 0)), 
                int(user_data_copy.get('anxiety2', 0)),
                int(user_data_copy.get('depression1', 0)), 
                int(user_data_copy.get('depression2', 0))
            ])
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating composite score: {e}")
            composite_score = 0
        
        # Save quiz result to history
        result_data = {
            'severity_level': prediction_result['severity_level'],
            'ensemble_confidence': prediction_result['ensemble_confidence'],
            'recommendation': treatment_data['recommendation'],
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_quiz_result(session['user_id'], result_data)
        
        # Clear the quiz data from session AFTER successful processing
        session.pop('quiz_data', None)
        
        # Render the result page
        return render_template('result.html', 
                             prediction_result=prediction_result,
                             risk_factors=risk_factors,
                             treatment_data=treatment_data,
                             risk_chart=risk_chart,
                             treatment_chart=treatment_chart,
                             composite_score=composite_score,
                             user_data=user_data_copy)
    except Exception as e:
        logger.error(f"Error processing quiz result: {e}")
        # Don't clear session data on error so user can try again
        return render_template('quiz.html', error="An error occurred processing your results. Please try again.")

@app.route('/quiz-result')
def quiz_result():
    """Show the most recent quiz result for the current user"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Get the most recent quiz result for this user
        history = get_quiz_history(session['user_id'])
        if not history or len(history) == 0:
            return redirect(url_for('quiz'))
        
        # Get the most recent result
        latest_result = history[-1]
        
        # Create a simplified result template
        return render_template('result.html', 
                             latest_result=latest_result,
                             history=history)
    except Exception as e:
        logger.error(f"Error showing quiz result: {e}")
        return redirect(url_for('dashboard'))

@app.route('/api/submit-quiz', methods=['POST'])
def api_submit_quiz():
    """API endpoint for quiz submission"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    try:
        # Extract all form data
        user_data = {
            'sleep_hours': request.form.get('sleep_hours'),
            'screen_time': request.form.get('screen_time'),
            'diet': request.form.get('diet'),
            'exercise': request.form.get('exercise'),
            'social_activity': request.form.get('social_activity'),
            'mood': request.form.get('mood'),
            'anxiety1': request.form.get('anxiety1'),
            'anxiety2': request.form.get('anxiety2'),
            'depression1': request.form.get('depression1'),
            'depression2': request.form.get('depression2')
        }
        
        # Validate that all required fields are present
        for key, value in user_data.items():
            if value is None or value == '':
                logger.error(f"Missing required field: {key}")
                return jsonify({'success': False, 'message': f"Missing required field: {key}"})
        
        # Get mental health prediction using ML models
        prediction_result = mental_health_predictor.predict_mental_health(user_data)
        
        # Analyze risk factors
        risk_factors = mental_health_predictor.analyze_risk_factors(user_data)
        
        # Get treatment recommendation
        treatment_data = mental_health_predictor.get_treatment_recommendation(
            prediction_result['severity_level'], 
            prediction_result['ensemble_confidence']
        )
        
        # Calculate composite score for backward compatibility
        composite_score = sum([
            int(user_data['anxiety1']), int(user_data['anxiety2']),
            int(user_data['depression1']), int(user_data['depression2'])
        ])
        
        # Save quiz result to history
        result_data = {
            'severity_level': prediction_result['severity_level'],
            'ensemble_confidence': prediction_result['ensemble_confidence'],
            'recommendation': treatment_data['recommendation'],
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        save_quiz_result(session['user_id'], result_data)
        
        # Return success response with redirect URL
        return jsonify({
            'success': True, 
            'redirect': url_for('quiz_result')
        })
                             
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'success': False, 'message': f"An error occurred: {str(e)}"})

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not logged in'}), 401
        data = request.get_json(silent=True) or {}
        message = data.get('message', '')
        
        # Debug logging
        logger.info(f"Chat request: '{message}'")

        # Build recent chat history for better context (last 10 turns)
        history = get_chat_history(session['user_id'], limit=20)

        result = llm_service.generate_response(message, chat_history=history)
        
        # Debug logging
        logger.info(f"Chat response: {result}")
        
        # Persist user message and bot response
        try:
            save_chat_message(session['user_id'], 'user', message)
            bot_text = (result or {}).get('response', '')
            if bot_text:
                save_chat_message(session['user_id'], 'assistant', bot_text)
        except Exception as se:
            logger.error(f"Error saving chat history: {se}")

        return jsonify({'success': True, 'data': result})
    except Exception as e:
        logger.error(f"/api/chat error: {e}")
        return jsonify({'success': False, 'message': 'Chat service error'}), 500

@app.route('/api/chat/debug', methods=['GET'])
def chat_debug():
    """Debug endpoint to test chat service"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    test_messages = [
        "Hello",
        "I feel sad",
        "I'm stressed about work",
        "What is depression?",
        "I need help"
    ]
    
    results = []
    for msg in test_messages:
        result = llm_service.generate_response(msg)
        results.append({
            'message': msg,
            'response': result
        })
    
    return jsonify({'success': True, 'data': results})

if __name__ == '__main__':
    # Load existing data from files when starting the app
    load_users_from_file()
    load_quiz_history_from_file()
    
    app.run(debug=True, host='0.0.0.0', port=5000)