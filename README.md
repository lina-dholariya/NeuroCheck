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
