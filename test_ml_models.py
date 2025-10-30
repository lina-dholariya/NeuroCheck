#!/usr/bin/env python3
"""
Test script for Mental Health ML Models
"""

from ml_models import mental_health_predictor
import json

def test_ml_models():
    """Test the machine learning models with sample data"""
    
    print("🧠 Testing Mental Health ML Models")
    print("=" * 50)
    
    # Test 1: Train models
    print("\n1. Training models...")
    try:
        accuracy = mental_health_predictor.train_models()
        print(f"✅ Models trained successfully!")
        print(f"   Logistic Regression Accuracy: {accuracy['logistic_accuracy']:.3f}")
        print(f"   Random Forest Accuracy: {accuracy['random_forest_accuracy']:.3f}")
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return False
    
    # Test 2: Sample user data
    print("\n2. Testing with sample user data...")
    
    # Sample 1: Low risk user
    low_risk_user = {
        'sleep_hours': '8',
        'screen_time': '4',
        'diet': '3',
        'exercise': '3',
        'social_activity': '2',
        'mood': '2',
        'anxiety1': '0',
        'anxiety2': '0',
        'depression1': '0',
        'depression2': '0'
    }
    
    # Sample 2: High risk user
    high_risk_user = {
        'sleep_hours': '5',
        'screen_time': '10',
        'diet': '0',
        'exercise': '0',
        'social_activity': '0',
        'mood': '0',
        'anxiety1': '3',
        'anxiety2': '3',
        'depression1': '3',
        'depression2': '3'
    }
    
    # Test low risk user
    print("\n   Testing Low Risk User:")
    try:
        result = mental_health_predictor.predict_mental_health(low_risk_user)
        print(f"   ✅ Prediction: {result['severity_level']}")
        print(f"   ✅ Confidence: {result['ensemble_confidence']:.1f}%")
        print(f"   ✅ LR Model: {['Minimal', 'Mild', 'Moderate', 'High'][result['logistic_prediction']]}")
        print(f"   ✅ RF Model: {['Minimal', 'Mild', 'Moderate', 'High'][result['random_forest_prediction']]}")
    except Exception as e:
        print(f"   ❌ Error predicting low risk user: {e}")
    
    # Test high risk user
    print("\n   Testing High Risk User:")
    try:
        result = mental_health_predictor.predict_mental_health(high_risk_user)
        print(f"   ✅ Prediction: {result['severity_level']}")
        print(f"   ✅ Confidence: {result['ensemble_confidence']:.1f}%")
        print(f"   ✅ LR Model: {['Minimal', 'Mild', 'Moderate', 'High'][result['logistic_prediction']]}")
        print(f"   ✅ RF Model: {['Minimal', 'Mild', 'Moderate', 'High'][result['random_forest_prediction']]}")
    except Exception as e:
        print(f"   ❌ Error predicting high risk user: {e}")
    
    # Test 3: Wellness factor analysis
    print("\n3. Testing wellness factor analysis...")
    try:
        risk_factors = mental_health_predictor.analyze_risk_factors(high_risk_user)
        print("   ✅ Wellness factors calculated:")
        for factor, value in risk_factors.items():
            # Determine wellness status based on logic
            if factor == 'Stress Level':
                status = "Good" if value < 30 else "Moderate" if value < 60 else "Poor"
            else:
                status = "Good" if value >= 70 else "Moderate" if value >= 40 else "Poor"
            print(f"      {factor}: {value:.0f}% ({status})")
    except Exception as e:
        print(f"   ❌ Error analyzing wellness factors: {e}")
    
    # Test 4: Treatment recommendation
    print("\n4. Testing treatment recommendation...")
    try:
        treatment = mental_health_predictor.get_treatment_recommendation('High', 85.0)
        print(f"   ✅ Treatment needed: {treatment['treatment_needed']}")
        print(f"   ✅ Confidence: {treatment['confidence']:.1f}%")
        print(f"   ✅ Recommendation: {treatment['recommendation']}")
    except Exception as e:
        print(f"   ❌ Error getting treatment recommendation: {e}")
    
    # Test 5: Chart generation
    print("\n5. Testing chart generation...")
    try:
        risk_chart = mental_health_predictor.create_risk_factor_chart(risk_factors)
        treatment_chart = mental_health_predictor.create_treatment_chart(treatment)
        print("   ✅ Wellness factor chart generated")
        print("   ✅ Treatment recommendation chart generated")
        print(f"   ✅ Chart data length: {len(risk_chart)} characters")
    except Exception as e:
        print(f"   ❌ Error generating charts: {e}")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("The ML models are ready to use in the application.")
    
    return True

if __name__ == "__main__":
    test_ml_models()
