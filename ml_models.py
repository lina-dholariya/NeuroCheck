import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os

class MentalHealthPredictor:
    def __init__(self):
        self.logistic_model = LogisticRegression(random_state=42, max_iter=1000)
        self.random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature names for risk analysis
        self.feature_names = [
            'Sleep Hours', 'Screen Time', 'Diet Quality', 'Exercise Frequency',
            'Social Activity', 'Mood Level', 'Anxiety Level 1', 'Anxiety Level 2',
            'Depression Level 1', 'Depression Level 2'
        ]
        
        # Risk factor categories
        self.risk_factors = [
            'Stress Level', 'Social Support', 'Work-Life Balance', 'Physical Health',
            'Sleep Quality', 'Digital Wellness', 'Emotional Stability', 'Lifestyle Balance'
        ]
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for mental health prediction"""
        np.random.seed(42)
        
        # Generate features
        sleep_hours = np.random.normal(7, 2, n_samples).clip(3, 12)
        screen_time = np.random.normal(6, 3, n_samples).clip(0, 16)
        diet_quality = np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.3, 0.2])
        exercise_freq = np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.3, 0.25, 0.15])
        social_activity = np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.4, 0.4])
        mood_level = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
        anxiety1 = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        anxiety2 = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        depression1 = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        depression2 = np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        
        # Create feature matrix
        X = np.column_stack([
            sleep_hours, screen_time, diet_quality, exercise_freq, social_activity,
            mood_level, anxiety1, anxiety2, depression1, depression2
        ])
        
        # Generate target labels based on mental health severity
        # Calculate a composite score
        composite_score = (
            (8 - sleep_hours) * 0.5 +  # Less sleep = higher risk
            screen_time * 0.3 +        # More screen time = higher risk
            (3 - diet_quality) * 0.4 + # Poorer diet = higher risk
            (3 - exercise_freq) * 0.4 + # Less exercise = higher risk
            (2 - social_activity) * 0.5 + # Less social = higher risk
            (2 - mood_level) * 0.8 +   # Poor mood = higher risk
            anxiety1 * 0.6 +           # Anxiety symptoms
            anxiety2 * 0.6 +           # Anxiety symptoms
            depression1 * 0.7 +        # Depression symptoms
            depression2 * 0.7          # Depression symptoms
        )
        
        # Assign severity levels
        y = np.zeros(n_samples, dtype=int)
        y[composite_score < 5] = 0    # Minimal
        y[(composite_score >= 5) & (composite_score < 10)] = 1  # Mild
        y[(composite_score >= 10) & (composite_score < 15)] = 2  # Moderate
        y[composite_score >= 15] = 3  # High
        
        return X, y
    
    def train_models(self):
        """Train both Logistic Regression and Random Forest models"""
        # Generate training data
        X, y = self.generate_synthetic_data(n_samples=2000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Logistic Regression
        self.logistic_model.fit(X_train_scaled, y_train)
        lr_pred = self.logistic_model.predict(X_test_scaled)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        # Train Random Forest
        self.random_forest_model.fit(X_train, y_train)
        rf_pred = self.random_forest_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        self.is_trained = True
        
        print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")
        print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
        
        return {
            'logistic_accuracy': lr_accuracy,
            'random_forest_accuracy': rf_accuracy
        }
    
    def predict_mental_health(self, user_data):
        """Predict mental health severity using both models"""
        if not self.is_trained:
            self.train_models()
        
        # Extract features from user data
        features = np.array([
            float(user_data.get('sleep_hours', 7)),
            float(user_data.get('screen_time', 6)),
            int(user_data.get('diet', 1)),
            int(user_data.get('exercise', 1)),
            int(user_data.get('social_activity', 1)),
            int(user_data.get('mood', 1)),
            int(user_data.get('anxiety1', 0)),
            int(user_data.get('anxiety2', 0)),
            int(user_data.get('depression1', 0)),
            int(user_data.get('depression2', 0))
        ]).reshape(1, -1)
        
        # Scale features for logistic regression
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from both models
        lr_pred = self.logistic_model.predict(features_scaled)[0]
        rf_pred = self.random_forest_model.predict(features)[0]
        
        # Get probabilities
        lr_proba = self.logistic_model.predict_proba(features_scaled)[0]
        rf_proba = self.random_forest_model.predict_proba(features)[0]
        
        # Ensemble prediction (average of both models)
        ensemble_pred = int(round((lr_pred + rf_pred) / 2))
        
        # Calculate confidence score
        lr_confidence = max(lr_proba) * 100
        rf_confidence = max(rf_proba) * 100
        ensemble_confidence = (lr_confidence + rf_confidence) / 2
        
        # Severity levels
        severity_levels = ['Minimal', 'Mild', 'Moderate', 'High']
        
        return {
            'logistic_prediction': lr_pred,
            'random_forest_prediction': rf_pred,
            'ensemble_prediction': ensemble_pred,
            'severity_level': severity_levels[ensemble_pred],
            'logistic_confidence': lr_confidence,
            'random_forest_confidence': rf_confidence,
            'ensemble_confidence': ensemble_confidence,
            'logistic_probabilities': lr_proba.tolist(),
            'random_forest_probabilities': rf_proba.tolist(),
            'features': features[0].tolist()
        }
    
    def analyze_risk_factors(self, user_data):
        """Analyze risk factors based on user responses with proper wellness logic"""
        # Calculate risk scores for different factors
        sleep_hours = float(user_data.get('sleep_hours', 7))
        screen_time = float(user_data.get('screen_time', 6))
        diet_quality = int(user_data.get('diet', 1))
        exercise_freq = int(user_data.get('exercise', 1))
        social_activity = int(user_data.get('social_activity', 1))
        mood_level = int(user_data.get('mood', 1))
        anxiety1 = int(user_data.get('anxiety1', 0))
        anxiety2 = int(user_data.get('anxiety2', 0))
        depression1 = int(user_data.get('depression1', 0))
        depression2 = int(user_data.get('depression2', 0))
        
        # Calculate wellness percentages based on ideal levels
        # Stress Level: HIGH % = BAD (Red), LOW % = GOOD (Green)
        stress_level = min(100, (anxiety1 + anxiety2) * 25 + (depression1 + depression2) * 20)
        
        # Social Support: HIGH % = GOOD (Green), LOW % = BAD (Red)
        social_support = max(0, 100 - (2 - social_activity) * 40)
        
        # Work-Life Balance: HIGH % = GOOD (Green), LOW % = BAD (Red)
        work_life_balance = max(0, 100 - screen_time * 5 - (3 - exercise_freq) * 15)
        
        # Physical Health: HIGH % = GOOD (Green), LOW % = BAD (Red)
        physical_health = max(0, 100 - (3 - diet_quality) * 25 - (3 - exercise_freq) * 20)
        
        # Sleep Quality: HIGH % = GOOD (Green), LOW % = BAD (Red)
        sleep_quality = max(0, 100 - abs(7 - sleep_hours) * 10)
        
        # Digital Wellness: HIGH % = GOOD (Green), LOW % = BAD (Red)
        digital_wellness = max(0, 100 - screen_time * 6)
        
        # Emotional Stability: HIGH % = GOOD (Green), LOW % = BAD (Red)
        emotional_stability = max(0, 100 - (2 - mood_level) * 40 - (anxiety1 + anxiety2) * 15)
        
        # Lifestyle Balance: HIGH % = GOOD (Green), LOW % = BAD (Red)
        lifestyle_balance = max(0, 100 - abs(7 - sleep_hours) * 8 - screen_time * 4 - (3 - exercise_freq) * 12)
        
        risk_factors = {
            'Stress Level': stress_level,
            'Social Support': social_support,
            'Work-Life Balance': work_life_balance,
            'Physical Health': physical_health,
            'Sleep Quality': sleep_quality,
            'Digital Wellness': digital_wellness,
            'Emotional Stability': emotional_stability,
            'Lifestyle Balance': lifestyle_balance
        }
        
        return risk_factors
    
    def get_treatment_recommendation(self, severity_level, confidence_score):
        """Generate treatment recommendations based on severity and confidence"""
        if severity_level in ['Moderate', 'High']:
            treatment_needed = True
            treatment_confidence = confidence_score
        elif severity_level == 'Mild':
            treatment_needed = confidence_score > 60
            treatment_confidence = confidence_score
        else:  # Minimal
            treatment_needed = False
            treatment_confidence = 100 - confidence_score
        
        return {
            'treatment_needed': treatment_needed,
            'confidence': treatment_confidence,
            'recommendation': self._get_recommendation_text(severity_level, treatment_needed)
        }
    
    def _get_recommendation_text(self, severity_level, treatment_needed):
        """Get personalized recommendation text"""
        recommendations = {
            'Minimal': {
                True: "Continue maintaining your healthy habits and consider regular check-ins.",
                False: "Excellent! Keep up with your current wellness routine."
            },
            'Mild': {
                True: "Consider talking to a friend, family member, or mental health professional. Practice self-care techniques.",
                False: "Monitor your mood and practice stress management techniques."
            },
            'Moderate': {
                True: "Strongly recommend consulting with a mental health professional. Consider therapy or counseling.",
                False: "Consider seeking professional help and implementing stress reduction strategies."
            },
            'High': {
                True: "Immediate professional help is highly recommended. Contact a mental health professional or crisis hotline.",
                False: "Urgent: Please seek professional mental health support immediately."
            }
        }
        
        return recommendations[severity_level][treatment_needed]
    
    def create_risk_factor_chart(self, risk_factors):
        """Create risk factor analysis bar chart with proper wellness color-coding"""
        plt.figure(figsize=(16, 10))
        plt.style.use('seaborn-v0_8')
        
        factors = list(risk_factors.keys())
        values = list(risk_factors.values())
        
        # Define color-coding logic based on wellness guidelines
        def get_wellness_color(factor, value):
            """Get color based on factor and value according to wellness logic"""
            # Stress Level: HIGH % = BAD (Red), LOW % = GOOD (Green)
            if factor == 'Stress Level':
                if value < 30:
                    return '#2E8B57'  # Green (Good)
                elif value < 60:
                    return '#FFA500'  # Orange (Moderate)
                else:
                    return '#FF4500'  # Red (Bad)
            # All other factors: HIGH % = GOOD (Green), LOW % = BAD (Red)
            else:
                if value >= 70:
                    return '#2E8B57'  # Green (Good)
                elif value >= 40:
                    return '#FFA500'  # Orange (Moderate)
                else:
                    return '#FF4500'  # Red (Bad)
        
        # Create colors based on wellness logic
        colors = [get_wellness_color(factor, value) for factor, value in zip(factors, values)]
        
        bars = plt.bar(factors, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.0f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Wellness Factor Analysis', fontsize=20, fontweight='bold', pad=25)
        plt.xlabel('Wellness Factors', fontsize=14, fontweight='bold')
        plt.ylabel('Wellness Level (%)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        
        # Add legend with wellness context
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2E8B57', label='Good Wellness (Green)'),
            Patch(facecolor='#FFA500', label='Moderate Wellness (Orange)'),
            Patch(facecolor='#FF4500', label='Poor Wellness (Red)')
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.tight_layout()
        
        # Save to base64 string with higher DPI for better quality
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=400, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str
    
    def create_treatment_chart(self, treatment_data):
        """Create treatment recommendation doughnut chart"""
        plt.figure(figsize=(14, 10))
        plt.style.use('seaborn-v0_8')
        
        labels = ['Treatment Recommended', 'No Treatment Needed']
        sizes = [treatment_data['confidence'], 100 - treatment_data['confidence']]
        colors = ['#FF6B6B', '#4ECDC4']
        
        # Create doughnut chart
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90,
                                          pctdistance=0.85, labeldistance=1.1)
        
        # Create center circle for doughnut effect
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        # Add confidence score in center
        plt.text(0, 0, f'Confidence\n{treatment_data["confidence"]:.1f}%', 
                ha='center', va='center', fontsize=18, fontweight='bold')
        
        plt.title('Treatment Recommendation', fontsize=20, fontweight='bold', pad=25)
        
        # Save to base64 string with higher DPI for better quality
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=400, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_str

# Global instance
mental_health_predictor = MentalHealthPredictor()
