#!/usr/bin/env python3
"""
Demonstration of Wellness Color-Coding System
Shows how different factors are color-coded based on wellness logic
"""

from ml_models import mental_health_predictor

def demo_wellness_colors():
    """Demonstrate the wellness color-coding system with different scenarios"""
    
    print("🧠 Wellness Color-Coding System Demonstration")
    print("=" * 60)
    
    # Train models first
    mental_health_predictor.train_models()
    
    # Scenario 1: Optimal Wellness User
    print("\n📊 Scenario 1: Optimal Wellness User")
    print("-" * 40)
    optimal_user = {
        'sleep_hours': '8',
        'screen_time': '3',
        'diet': '3',
        'exercise': '3',
        'social_activity': '2',
        'mood': '2',
        'anxiety1': '0',
        'anxiety2': '0',
        'depression1': '0',
        'depression2': '0'
    }
    
    factors = mental_health_predictor.analyze_risk_factors(optimal_user)
    print_wellness_analysis(factors, "Optimal")
    
    # Scenario 2: Moderate Wellness User
    print("\n📊 Scenario 2: Moderate Wellness User")
    print("-" * 40)
    moderate_user = {
        'sleep_hours': '6',
        'screen_time': '6',
        'diet': '2',
        'exercise': '2',
        'social_activity': '1',
        'mood': '1',
        'anxiety1': '1',
        'anxiety2': '1',
        'depression1': '1',
        'depression2': '1'
    }
    
    factors = mental_health_predictor.analyze_risk_factors(moderate_user)
    print_wellness_analysis(factors, "Moderate")
    
    # Scenario 3: Poor Wellness User
    print("\n📊 Scenario 3: Poor Wellness User")
    print("-" * 40)
    poor_user = {
        'sleep_hours': '4',
        'screen_time': '12',
        'diet': '0',
        'exercise': '0',
        'social_activity': '0',
        'mood': '0',
        'anxiety1': '3',
        'anxiety2': '3',
        'depression1': '3',
        'depression2': '3'
    }
    
    factors = mental_health_predictor.analyze_risk_factors(poor_user)
    print_wellness_analysis(factors, "Poor")
    
    print("\n" + "=" * 60)
    print("✅ Color-Coding Logic Summary:")
    print("🟢 Green (Good): Optimal wellness levels")
    print("🟠 Orange (Moderate): Areas needing attention")
    print("🔴 Red (Poor): Critical areas requiring focus")
    print("\n📝 Note: Stress Level is inverted - LOW % = GOOD, HIGH % = BAD")

def print_wellness_analysis(factors, scenario_name):
    """Print wellness analysis with color indicators"""
    print(f"Results for {scenario_name} Wellness User:")
    
    for factor, value in factors.items():
        # Determine color and status based on wellness logic
        if factor == 'Stress Level':
            if value < 30:
                color = "🟢"
                status = "Good"
            elif value < 60:
                color = "🟠"
                status = "Moderate"
            else:
                color = "🔴"
                status = "Poor"
        else:
            if value >= 70:
                color = "🟢"
                status = "Good"
            elif value >= 40:
                color = "🟠"
                status = "Moderate"
            else:
                color = "🔴"
                status = "Poor"
        
        print(f"  {color} {factor}: {value:.0f}% ({status})")

if __name__ == "__main__":
    demo_wellness_colors()
