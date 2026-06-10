# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
from recommend import (
    get_physical_recommendations,
    get_mental_recommendations,
    get_combined_health_index,
    get_weekly_action_plan
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MindBody Health AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #667eea44;
        text-align: center;
    }
    .risk-high   { color: #ff4b4b; font-size: 1.5rem; font-weight: 700; }
    .risk-medium { color: #ffa500; font-size: 1.5rem; font-weight: 700; }
    .risk-low    { color: #00cc44; font-size: 1.5rem; font-weight: 700; }
    .recommendation-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
    }
    .chat-message-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-message-ai {
        background: #f0f2f6;
        color: #333;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .plan-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #eee;
        margin: 0.3rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD API KEY (from Streamlit secrets or .env)
# ─────────────────────────────────────────────────────────────────────────────

def get_api_key():
    # First try Streamlit secrets (for deployed app)
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass

    # Then try environment variables (for local dev)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return None

api_key = get_api_key()

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    physical_model = joblib.load("models/physical_model.pkl")
    mental_model   = joblib.load("models/mental_model.pkl")
    return physical_model, mental_model

physical_model, mental_model = load_models()

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

if "history"       not in st.session_state: st.session_state.history       = []
if "last_results"  not in st.session_state: st.session_state.last_results  = None
if "chat_messages" not in st.session_state: st.session_state.chat_messages = []

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧬 MindBody Health AI")
    st.markdown("*Your personal AI health assistant*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home",
         "🔬 Health Assessment",
         "📊 Health Dashboard",
         "🤖 AI Chat Assistant",
         "🎯 Action Plan"],
        label_visibility="hidden"
    )
    st.markdown("---")
    st.markdown("### 📋 SDG 3: Good Health")
    st.markdown("*Early detection and mental wellness awareness*")
    st.markdown("---")
    st.caption("⚠️ Not a medical diagnosis. Always consult a doctor.")
    st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — HOME
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown('<p class="main-header">🧬 MindBody Health AI</p>', unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;color:#888'>AI-powered physical and mental health risk assistant</h4>",
                unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    for col, emoji, title, desc in zip(
        [col1, col2, col3, col4],
        ["🔬", "📊", "🤖", "🎯"],
        ["Health Assessment", "Visual Dashboard", "AI Chat", "Action Plan"],
        ["AI-powered physical and mental risk analysis",
         "Charts and insights from your health data",
         "Ask health questions, get personalised answers",
         "Your personalised 7-day health plan"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <h2>{emoji}</h2>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How It Works")
    col1, col2, col3, col4 = st.columns(4)
    for col, step, desc in zip(
        [col1, col2, col3, col4],
        ["Step 1", "Step 2", "Step 3", "Step 4"],
        ["Fill the health assessment form",
         "AI analyses your physical and mental risk",
         "View your dashboard and recommendations",
         "Follow your personalised action plan"]
    ):
        with col:
            st.markdown(f"**{step}**\n\n{desc}")

    st.markdown("---")
    if st.session_state.last_results:
        st.success("✅ You have a recent assessment! Head to 📊 Health Dashboard to view it.")
    else:
        st.info("👉 Start by clicking **🔬 Health Assessment** in the sidebar")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — HEALTH ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔬 Health Assessment":
    st.markdown("## 🔬 Health Assessment")
    st.markdown("Fill in the form below. Takes about 2 minutes.")
    st.warning("⚠️ This tool is for awareness only — not a substitute for professional medical advice.")
    st.markdown("---")

    # ── PHYSICAL HEALTH ──────────────────────────────────────────────────────
    st.markdown("### 🫀 Physical Health")
    col1, col2 = st.columns(2)

    with col1:
        age            = st.slider("Age", 15, 80, 25)
        glucose        = st.slider("Glucose Level (mg/dL)", 50, 250, 100,
                                   help="Normal fasting glucose: 70–100 mg/dL")
        bmi            = st.slider("BMI", 10.0, 60.0, 22.0, step=0.1,
                                   help="Normal BMI: 18.5–24.9")
        insulin        = st.slider("Insulin Level (mu U/ml)", 0, 900, 80)

    with col2:
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 40, 130, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        dpf            = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01,
                                   help="Measures genetic influence. Average ~0.47")
        pregnancies    = st.slider("Number of Pregnancies", 0, 17, 0)

    physical_input = {
        "Pregnancies":              pregnancies,
        "Glucose":                  glucose,
        "BloodPressure":            blood_pressure,
        "SkinThickness":            skin_thickness,
        "Insulin":                  insulin,
        "BMI":                      bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age":                      age
    }

    st.markdown("---")

    # ── MENTAL HEALTH ────────────────────────────────────────────────────────
    st.markdown("### 🧠 Mental Wellness")
    col1, col2 = st.columns(2)

    with col1:
        gender          = st.selectbox("Gender", ["Male", "Female"])
        occupation      = st.selectbox("Occupation",
                                       ["Student", "Corporate", "Business",
                                        "Housewife", "Others"])
        self_employed   = st.selectbox("Are you self-employed?", ["No", "Yes"])
        family_history  = st.selectbox("Family history of mental illness?", ["No", "Yes"])
        days_indoors    = st.selectbox("How long do you stay indoors?",
                                       ["Go out Every day", "1-14 days",
                                        "15-30 days", "31-60 days",
                                        "More than 2 months"])
        growing_stress  = st.selectbox("Are you experiencing growing stress?",
                                       ["No", "Maybe", "Yes"])
        changes_habits  = st.selectbox("Have you noticed changes in your habits?",
                                       ["No", "Maybe", "Yes"])

    with col2:
        mental_history  = st.selectbox("Do you have a mental health history?",
                                       ["No", "Maybe", "Yes"])
        mood_swings     = st.selectbox("How are your mood swings?",
                                       ["Low", "Medium", "High"])
        coping          = st.selectbox("Do you struggle with coping?", ["No", "Yes"])
        work_interest   = st.selectbox("Do you have interest in work/studies?",
                                       ["Yes", "Maybe", "No"])
        social_weakness = st.selectbox("Do you feel socially weak?",
                                       ["No", "Maybe", "Yes"])
        mh_interview    = st.selectbox("Would you discuss mental health in an interview?",
                                       ["No", "Maybe", "Yes"])
        care_options    = st.selectbox("Do you have access to mental health care options?",
                                       ["No", "Not sure", "Yes"])

    # Encode
    binary_map  = {"No": 0, "Yes": 1}
    three_map   = {"No": 0, "Maybe": 1, "Yes": 2}
    gender_map  = {"Male": 1, "Female": 0}
    occ_map     = {"Corporate": 0, "Student": 1, "Business": 2,
                   "Housewife": 3, "Others": 4}
    days_map    = {"go out every day": 0, "1-14 days": 1, "15-30 days": 2,
                   "31-60 days": 3, "more than 2 months": 4,
                   "Go out Every day": 0, "More than 2 months": 4}
    mood_map    = {"Low": 0, "Medium": 1, "High": 2}
    care_map    = {"No": 0, "Not sure": 1, "Yes": 2}
    wi_map      = {"Yes": 0, "Maybe": 1, "No": 2}

    mental_input_encoded = {
        "Gender":                 gender_map.get(gender, 0),
        "Occupation":             occ_map.get(occupation, 0),
        "self_employed":          binary_map.get(self_employed, 0),
        "family_history":         binary_map.get(family_history, 0),
        "Days_Indoors":           days_map.get(days_indoors, 0),
        "Growing_Stress":         three_map.get(growing_stress, 0),
        "Changes_Habits":         three_map.get(changes_habits, 0),
        "Mental_Health_History":  three_map.get(mental_history, 0),
        "Mood_Swings":            mood_map.get(mood_swings, 0),
        "Coping_Struggles":       binary_map.get(coping, 0),
        "Work_Interest":          wi_map.get(work_interest, 0),
        "Social_Weakness":        three_map.get(social_weakness, 0),
        "mental_health_interview": three_map.get(mh_interview, 0),
        "care_options":           care_map.get(care_options, 0),
    }

    mental_input_raw = {
        "family_history":  family_history,
        "work_interfere":  growing_stress,
        "seek_help":       care_options,
        "mood_swings":     mood_swings,
        "coping":          coping,
        "occupation":      occupation,
    }

    st.markdown("---")

    # ── SUBMIT ───────────────────────────────────────────────────────────────
    if st.button("🔍 Analyse My Health", type="primary", use_container_width=True):
        with st.spinner("Running AI analysis..."):

            # Physical prediction
            physical_df    = pd.DataFrame([physical_input])
            physical_proba = physical_model.predict_proba(physical_df)[0][1]

            # Mental prediction
            mental_cols = [
                "Gender", "Occupation", "self_employed", "family_history",
                "Days_Indoors", "Growing_Stress", "Changes_Habits",
                "Mental_Health_History", "Mood_Swings", "Coping_Struggles",
                "Work_Interest", "Social_Weakness", "mental_health_interview",
                "care_options"
            ]
            mental_df    = pd.DataFrame([mental_input_encoded])[mental_cols]
            mental_proba = mental_model.predict_proba(mental_df)[0][1]

            # Recommendations
            physical_level, physical_recs = get_physical_recommendations(
                physical_proba, physical_input)
            mental_level, mental_recs     = get_mental_recommendations(
                mental_proba, mental_input_raw)

            # Combined
            combined_score, combined_status, combined_color, combined_emoji = \
                get_combined_health_index(physical_proba, mental_proba)

            # Save
            results = {
                "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M"),
                "physical_score":   round(physical_proba * 100, 1),
                "mental_score":     round(mental_proba * 100, 1),
                "combined_score":   combined_score,
                "physical_level":   physical_level,
                "mental_level":     mental_level,
                "physical_recs":    physical_recs,
                "mental_recs":      mental_recs,
                "combined_status":  combined_status,
                "combined_emoji":   combined_emoji,
                "physical_input":   physical_input,
                "mental_input_raw": mental_input_raw,
            }
            st.session_state.last_results = results
            st.session_state.history.append({
                "timestamp":      results["timestamp"],
                "physical_score": results["physical_score"],
                "mental_score":   results["mental_score"],
                "combined_score": results["combined_score"],
            })

        # Results
        st.markdown("---")
        st.markdown("## 📋 Your Results")

        col1, col2, col3 = st.columns(3)
        for col, label, score, level in zip(
            [col1, col2, col3],
            ["🫀 Physical Risk", "🧠 Mental Risk", "💚 Overall Health Index"],
            [physical_proba*100, mental_proba*100, combined_score],
            [physical_level, mental_level, combined_status]
        ):
            risk_class = ("risk-high"   if score >= 70 else
                          "risk-medium" if score >= 40 else "risk-low")
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <h4>{label}</h4>
                    <p class='{risk_class}'>{score:.1f}%</p>
                    <p>{level}</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🫀 Physical Recommendations")
            for rec in physical_recs:
                st.markdown(f"<div class='recommendation-box'>{rec}</div>",
                            unsafe_allow_html=True)
        with col2:
            st.markdown("#### 🧠 Mental Recommendations")
            for rec in mental_recs:
                st.markdown(f"<div class='recommendation-box'>{rec}</div>",
                            unsafe_allow_html=True)

        st.markdown("---")
        st.success("✅ Results saved! Head to 📊 Health Dashboard or 🎯 Action Plan.")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — HEALTH DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Health Dashboard":
    st.markdown("## 📊 Health Dashboard")

    if not st.session_state.last_results:
        st.info("👉 Complete the Health Assessment first.")
        st.stop()

    r = st.session_state.last_results

    # Gauge charts
    col1, col2 = st.columns(2)
    for col, value, title, color in zip(
        [col1, col2],
        [r["physical_score"], r["mental_score"]],
        ["Physical Risk %", "Mental Risk %"],
        ["#667eea", "#764ba2"]
    ):
        with col:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = value,
                title = {"text": title},
                gauge = {
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": color},
                    "steps": [
                        {"range": [0,  40],  "color": "rgba(0, 204, 68, 0.15)"},
                        {"range": [40, 70],  "color": "rgba(255, 165, 0, 0.15)"},
                        {"range": [70, 100], "color": "rgba(255, 75, 75, 0.15)"},
                    ],
                    "threshold": {
                        "line":  {"color": "red", "width": 4},
                        "value": 70
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=0))
            st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("### 🕸️ Health Profile Radar")
    pi         = r["physical_input"]
    categories = ["Glucose Control", "Blood Pressure",
                  "BMI Health", "Age Factor", "Mental Wellness"]
    values = [
        min(max((pi["Glucose"] - 70) / 180 * 100, 0), 100),
        min(max((pi["BloodPressure"] - 60) / 70 * 100, 0), 100),
        min(max((pi["BMI"] - 18) / 32 * 100, 0), 100),
        min(pi["Age"] / 80 * 100, 100),
        r["mental_score"]
    ]
    fig = go.Figure(go.Scatterpolar(
        r         = values + [values[0]],
        theta     = categories + [categories[0]],
        fill      = "toself",
        line      = {"color": "#667eea"},
        fillcolor = "rgba(102,126,234,0.2)"
    ))
    fig.update_layout(
        polar  = {"radialaxis": {"range": [0, 100]}},
        height = 400,
        margin = dict(t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # History chart
    if len(st.session_state.history) > 1:
        st.markdown("### 📈 Your Score History")
        hist_df = pd.DataFrame(st.session_state.history)
        fig = px.line(
            hist_df, x="timestamp",
            y=["physical_score", "mental_score", "combined_score"],
            title="Health Risk Over Time",
            color_discrete_map={
                "physical_score": "#667eea",
                "mental_score":   "#764ba2",
                "combined_score": "#f97316"
            }
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📈 Take more assessments over time to see your trend here.")

    # Key metrics
    st.markdown("### 📌 Key Physical Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Glucose",        f"{pi['Glucose']} mg/dL",
                delta="High" if pi['Glucose'] > 140 else "Normal",
                delta_color="inverse")
    col2.metric("BMI",            f"{pi['BMI']}",
                delta="High" if pi['BMI'] > 25 else "Normal",
                delta_color="inverse")
    col3.metric("Blood Pressure", f"{pi['BloodPressure']} mmHg",
                delta="High" if pi['BloodPressure'] > 90 else "Normal",
                delta_color="inverse")
    col4.metric("Age",            f"{pi['Age']} yrs")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — AI CHAT ASSISTANT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 AI Chat Assistant":
    st.markdown("## 🤖 AI Health Chat Assistant")
    st.markdown("Ask me anything about your health results or general wellness.")

    if not api_key:
        st.error("⚠️ AI Chat is not configured. Gemini API key missing.")
        st.info("Set GEMINI_API_KEY in a .env file or add it to Streamlit secrets.")
        st.stop()

    # System prompt with user context
    if st.session_state.last_results:
        r       = st.session_state.last_results
        context = f"""
        The user has completed a health assessment:
        - Physical Risk: {r['physical_score']}% ({r['physical_level']} risk)
        - Mental Risk: {r['mental_score']}% ({r['mental_level']} risk)
        - Overall Health Index: {r['combined_score']}% — {r['combined_status']}
        - Glucose: {r['physical_input']['Glucose']} mg/dL
        - BMI: {r['physical_input']['BMI']}
        - Blood Pressure: {r['physical_input']['BloodPressure']} mmHg
        - Age: {r['physical_input']['Age']}
        """
    else:
        context = "The user has not completed a health assessment yet."

    system_prompt = f"""You are a compassionate AI health assistant built into 
    MindBody Health AI, an SDG 3 (Good Health) project.

    {context}

    Your role:
    - Answer health questions clearly and compassionately
    - Reference the user's actual results when relevant
    - Always remind users you are not a doctor
    - Keep answers concise (3-5 sentences)
    - Be warm, supportive and encouraging
    """

    # Display chat history
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-message-user'>👤 {msg['content']}</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-message-ai'>🤖 {msg['content']}</div>",
                        unsafe_allow_html=True)

    user_input = st.text_input("Ask a health question...", key="ai_health_input")
    send = st.button("Send")

    if send and user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="models/gemini-2.5-flash",
                system_instruction=system_prompt
            )

            # Convert history to Gemini format with correct role names
            history = []
            for msg in st.session_state.chat_messages[:-1]:
                role = "USER" if msg["role"] == "user" else "MODEL"
                history.append({"role": role, "parts": [{"text": msg["content"]}]})
            
            chat = model.start_chat(history=history)
            response = chat.send_message(user_input)

            reply = response.text.strip()

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": reply}
            )
        except Exception as e:
            st.error(f"Error: {str(e)}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5 — ACTION PLAN
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🎯 Action Plan":
    st.markdown("## 🎯 Your Personalised Action Plan")

    if not st.session_state.last_results:
        st.info("👉 Complete the Health Assessment first.")
        st.stop()

    r    = st.session_state.last_results
    plan = get_weekly_action_plan(r["physical_level"], r["mental_level"])

    st.markdown(f"""
    ### {r['combined_emoji']} Overall Status: {r['combined_status']}
    *Physical: {r['physical_level']} | Mental: {r['mental_level']}*
    """)

    st.markdown("---")
    st.markdown("### 📅 Your 7-Day Health Plan")

    day_emojis = {
        "Monday": "1️⃣", "Tuesday": "2️⃣", "Wednesday": "3️⃣",
        "Thursday": "4️⃣", "Friday": "5️⃣", "Saturday": "6️⃣", "Sunday": "7️⃣"
    }

    for day, activity in plan.items():
        st.markdown(f"""
        <div class='plan-card'>
            <strong>{day_emojis[day]} {day}</strong><br>{activity}
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Top Priorities This Week")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🫀 Physical")
        for rec in r["physical_recs"][:3]:
            st.markdown(f"- {rec}")
    with col2:
        st.markdown("#### 🧠 Mental")
        for rec in r["mental_recs"][:3]:
            st.markdown(f"- {rec}")

    st.markdown("---")
    st.info("💬 Have questions? Visit the **🤖 AI Chat Assistant** page.")