# NeuroCheck - Mental Health Monitoring System

A modern web application for comprehensive mental health tracking with user authentication, built with Flask backend and HTML/CSS/JavaScript frontend, connected to MongoDB Atlas.

## Features

- 🧠 **NeuroCheck Design**: Beautiful gradient UI with brain-inspired elements
- 🤖 **Machine Learning Analysis**: Advanced ML models (Logistic Regression & Random Forest) for mental health prediction
- 📊 **Risk Factor Analysis**: Comprehensive analysis of 8 key risk factors with visual charts
- 💊 **Treatment Recommendations**: AI-powered treatment suggestions with confidence scores
- 🔐 **Secure Authentication**: Password hashing and session management
- 📱 **Responsive Design**: Works perfectly on desktop and mobile devices
- 🎨 **Modern UI**: Clean, intuitive interface with smooth animations
- 🗄️ **MongoDB Integration**: Cloud database storage with MongoDB Atlas
- ⚡ **Real-time Validation**: Client-side form validation with instant feedback

## Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, numpy, pandas
- **Visualization**: matplotlib, seaborn
- **Database**: MongoDB Atlas
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Authentication**: Session-based with password hashing
- **Styling**: Custom CSS with responsive design

## Project Structure

```
Minor Project/
├── app.py                 # Main Flask application
├── ml_models.py          # Machine learning models and analysis
├── test_ml_models.py     # ML model testing script
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── templates/            # HTML templates
│   ├── index.html       # Landing page
│   ├── login.html       # Login page
│   ├── signup.html      # Signup page
│   ├── dashboard.html   # User dashboard
│   ├── quiz.html        # Mental health assessment
│   ├── result.html      # ML analysis results
│   └── profile.html     # User profile
└── static/              # Static assets
    ├── css/
    │   └── style.css    # Main stylesheet
    └── js/
        └── auth.js      # Authentication JavaScript
```

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- MongoDB Atlas account (already configured)

### Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - The application will be running on port 5000

### Testing ML Models

To test the machine learning models independently:

```bash
python test_ml_models.py
```

This will:
- Train both Logistic Regression and Random Forest models
- Test predictions with sample data
- Verify wellness factor analysis
- Check treatment recommendations
- Generate test visualizations

### Wellness Color-Coding Demo

To see the wellness color-coding system in action:

```bash
python demo_wellness_colors.py
```

This demonstrates:
- Optimal wellness user (mostly green indicators)
- Moderate wellness user (mix of colors)
- Poor wellness user (mostly red indicators)
- Proper color logic for each factor

## Usage

### For Users

1. **Home Page** (`/`): Welcome page with navigation to login/signup
2. **Sign Up** (`/signup`): Create a new account with name, email, and password
3. **Login** (`/login`): Sign in with existing email and password
4. **Dashboard** (`/dashboard`): User dashboard (requires authentication)
5. **Logout** (`/logout`): Sign out and return to home page

### Features

- **Machine Learning Assessment**: Advanced ML models analyze 10 responses to predict mental health conditions
- **Severity Levels**: Four levels (Minimal, Mild, Moderate, High) with confidence scores
- **Wellness Factor Analysis**: Visual bar chart showing 8 wellness factors with proper color-coding based on ideal levels
- **Treatment Recommendations**: Doughnut chart with AI-powered treatment suggestions
- **Model Comparison**: Side-by-side comparison of Logistic Regression and Random Forest predictions
- **Form Validation**: Real-time validation for email format, password strength, and matching passwords
- **Error Handling**: Clear error messages for invalid inputs or server errors
- **Success Feedback**: Confirmation messages for successful operations
- **Responsive Design**: Optimized for all screen sizes
- **Session Management**: Secure user sessions with automatic redirects

## Machine Learning Models

### Models Used
- **Logistic Regression**: Linear model for mental health severity classification
- **Random Forest**: Ensemble model for robust prediction with feature importance
- **Ensemble Prediction**: Combined prediction from both models for higher accuracy

### Features Analyzed (10 total)
1. **Sleep Hours**: Average hours of sleep per night
2. **Screen Time**: Daily screen time exposure
3. **Diet Quality**: Nutritional habits assessment
4. **Exercise Frequency**: Physical activity levels
5. **Social Activity**: Social interaction patterns
6. **Mood Level**: General emotional state
7. **Anxiety Symptoms**: Two anxiety-related questions
8. **Depression Symptoms**: Two depression-related questions

### Wellness Factors Analyzed (8 total)
- **Stress Level**: High % = Bad (Red), Low % = Good (Green)
- **Social Support**: High % = Good (Green), Low % = Bad (Red)
- **Work-Life Balance**: High % = Good (Green), Low % = Bad (Red)
- **Physical Health**: High % = Good (Green), Low % = Bad (Red)
- **Sleep Quality**: High % = Good (Green), Low % = Bad (Red)
- **Digital Wellness**: High % = Good (Green), Low % = Bad (Red)
- **Emotional Stability**: High % = Good (Green), Low % = Bad (Red)
- **Lifestyle Balance**: High % = Good (Green), Low % = Bad (Red)

### Color-Coding System
- **Green (Good Wellness)**: Optimal levels for mental health
- **Orange (Moderate Wellness)**: Areas that need attention
- **Red (Poor Wellness)**: Critical areas requiring immediate focus

### Severity Levels
- **Minimal**: Low risk, healthy habits recommended
- **Mild**: Some concerns, self-care suggested
- **Moderate**: Professional help recommended
- **High**: Immediate professional intervention advised

## Database Schema

The application uses MongoDB Atlas with the following collection structure:

```javascript
// users collection
{
  "_id": ObjectId,
  "name": String,
  "email": String (unique),
  "password": String (hashed),
  "created_at": Date,
  "user_id": String (UUID)
}
```

## Security Features

- **Password Hashing**: Passwords are hashed using Werkzeug's security functions
- **Session Management**: Secure session handling with Flask sessions
- **Input Validation**: Server-side and client-side validation
- **CORS Support**: Cross-origin resource sharing enabled
- **Error Handling**: Graceful error handling without exposing sensitive information

## Customization

### Styling
- Modify `static/css/style.css` to change colors, fonts, and layout
- The design uses CSS Grid and Flexbox for responsive layouts
- Color scheme can be easily changed by updating CSS variables

### Functionality
- Add new routes in `app.py` for additional features
- Extend the dashboard with new mental health tracking features
- Implement additional validation rules in `static/js/auth.js`

## Troubleshooting

### Common Issues

1. **Port already in use**:
   - Change the port in `app.py` line 89: `app.run(debug=True, host='0.0.0.0', port=5001)`

2. **MongoDB connection error**:
   - Verify your MongoDB Atlas connection string
   - Check network connectivity
   - Ensure your IP is whitelisted in MongoDB Atlas

3. **Module not found errors**:
   - Run `pip install -r requirements.txt` to install all dependencies
   - Ensure you're using Python 3.7+
