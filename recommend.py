# recommend.py
# Generates health recommendations based on risk scores

def get_physical_recommendations(risk_score, user_data):
    """
    risk_score: float between 0 and 1 (probability of diabetes)
    user_data: dict of user inputs
    """
    recommendations = []

    if risk_score >= 0.7:
        level = "HIGH"
        recommendations += [
            "🚨 Consult a doctor as soon as possible",
            "🩸 Get a proper blood glucose test done",
            "🚫 Strictly avoid sugar and processed foods",
            "💊 Discuss medication options with your doctor",
        ]
    elif risk_score >= 0.4:
        level = "MODERATE"
        recommendations += [
            "⚠️ Monitor your blood sugar regularly",
            "🥗 Switch to a low glycemic index diet",
            "🏃 Exercise at least 30 minutes daily",
            "💧 Drink at least 2-3 litres of water daily",
        ]
    else:
        level = "LOW"
        recommendations += [
            "✅ Keep maintaining your healthy lifestyle",
            "🥦 Continue eating balanced meals",
            "🏋️ Stay physically active",
            "📅 Get a routine checkup once a year",
        ]

    # Add specific recommendations based on their actual inputs
    if user_data.get("Glucose", 0) > 140:
        recommendations.append("🍬 Your glucose is high — cut out sugary drinks immediately")
    if user_data.get("BMI", 0) > 30:
        recommendations.append("⚖️ Your BMI suggests obesity — consider a structured diet plan")
    if user_data.get("BloodPressure", 0) > 90:
        recommendations.append("❤️ High blood pressure detected — reduce salt intake")
    if user_data.get("Age", 0) > 45:
        recommendations.append("📋 Age is a risk factor — annual diabetes screening is important")

    return level, recommendations


def get_mental_recommendations(risk_score, user_data):
    """
    risk_score: float between 0 and 1 (probability of needing mental health support)
    user_data: dict of user inputs
    """
    recommendations = []

    if risk_score >= 0.7:
        level = "HIGH"
        recommendations += [
            "🧠 Please consider speaking to a mental health professional",
            "💬 Talk to someone you trust about how you're feeling",
            "📵 Take regular breaks from work and screens",
            "🛌 Prioritise getting 7-8 hours of sleep",
        ]
    elif risk_score >= 0.4:
        level = "MODERATE"
        recommendations += [
            "🧘 Try 10 minutes of meditation or deep breathing daily",
            "📓 Maintain a journal to track your emotions",
            "🚶 A short daily walk significantly reduces stress",
            "👥 Stay socially connected with friends and family",
        ]
    else:
        level = "LOW"
        recommendations += [
            "✅ Your mental wellness looks good — keep it up",
            "😊 Continue activities that bring you joy",
            "🌱 Practice gratitude — write 3 things daily",
            "⚖️ Maintain your work-life balance",
        ]

    # Specific inputs based recommendations
    if user_data.get("family_history") == "Yes":
        recommendations.append("🧬 Family history of mental illness — stay extra mindful of your mental health")
    if user_data.get("work_interfere") in ["Often", "Sometimes"]:
        recommendations.append("💼 Work is affecting your mental health — consider speaking to your manager")
    if user_data.get("seek_help") == "No":
        recommendations.append("🆘 Don't hesitate to seek professional help — it's a sign of strength")

    return level, recommendations


def get_combined_health_index(physical_score, mental_score):
    """
    Combines both scores into one overall health index (0-100)
    Lower is better
    """
    # Convert to percentage risk
    combined = (physical_score * 0.5 + mental_score * 0.5) * 100

    if combined >= 70:
        status = "Needs Immediate Attention"
        color  = "red"
        emoji  = "🔴"
    elif combined >= 40:
        status = "Needs Some Attention"
        color  = "orange"
        emoji  = "🟡"
    else:
        status = "Looking Good"
        color  = "green"
        emoji  = "🟢"

    return round(combined, 1), status, color, emoji


def get_weekly_action_plan(physical_level, mental_level):
    """
    Returns a 7 day action plan based on risk levels
    """
    plan = {
        "Monday":    "🏃 30 min morning walk + drink 8 glasses of water",
        "Tuesday":   "🥗 Eat a salad for lunch + 10 min breathing exercise",
        "Wednesday": "💤 Sleep by 10:30pm + no screens 1 hour before bed",
        "Thursday":  "🧘 15 min yoga or stretching + journal your mood",
        "Friday":    "🚴 Physical activity of your choice + call a friend",
        "Saturday":  "🍎 Cook a healthy meal + spend time in nature",
        "Sunday":    "📋 Review your week + set 3 health goals for next week",
    }

    # Add urgency if high risk
    if physical_level == "HIGH":
        plan["Monday"]    = "🚨 " + plan["Monday"] + " + Book a doctor's appointment"
        plan["Wednesday"] = "🩸 Get blood glucose tested today"

    if mental_level == "HIGH":
        plan["Tuesday"]   = "💬 Talk to someone you trust today"
        plan["Thursday"]  = "🧠 Research mental health support options in your area"

    return plan