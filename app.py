import streamlit as st
import joblib
from src.treatment import calculate_risk_score

# 🔹 Page Config
st.set_page_config(page_title="Diabetes AI System", layout="wide")

# 🔹 Load model
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# 🔹 Title
st.markdown(
    "<h1 style='text-align: center;'>🩺 Diabetes Prediction & Treatment System</h1>",
    unsafe_allow_html=True
)

# 🔹 Sidebar for inputs
st.sidebar.header("Enter Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0)
glucose = st.sidebar.number_input("Glucose", 0.0)
bp = st.sidebar.number_input("Blood Pressure", 0.0)
skin = st.sidebar.number_input("Skin Thickness", 0.0)
insulin = st.sidebar.number_input("Insulin", 0.0)
bmi = st.sidebar.number_input("BMI", 0.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0)
age = st.sidebar.number_input("Age", 0)

# 🔹 Predict button
if st.sidebar.button("🚀 Predict"):

    user_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]
    user_scaled = scaler.transform(user_data)

    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    risk_score = calculate_risk_score(user_data[0])

    # 🔥 Layout with columns
    col1, col2, col3 = st.columns(3)

    # 🔹 Prediction Card
    with col1:
        st.subheader("🔍 Prediction")

        if prediction == 1:
            st.error("Diabetic (1)")
        else:
            st.success("Not Diabetic (0)")

        st.write(f"Confidence: {probability:.2f}")

    # 🔹 Risk Score Card
    with col2:
        st.subheader("📊 Risk Score")

        st.progress(risk_score / 100)
        st.write(f"{risk_score}/100")

        if risk_score < 30:
            st.success("Low Risk")
        elif risk_score < 60:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

    # 🔹 Health Insights Card
    with col3:
        st.subheader("💡 Insights")

        if glucose > 140:
            st.write("⚠️ High Glucose")

        if bmi > 25:
            st.write("⚠️ High BMI")

        if bp > 80:
            st.write("⚠️ High Blood Pressure")

        st.write("✔ Stay active")
        st.write("✔ Monitor health")

    # 🔥 Diet Plan Section
    st.markdown("---")
    st.subheader("🥗 Personalized Diet Plan")

    if risk_score < 30:
        st.success("🥗 Balanced Diet: Fruits, vegetables, whole grains")
        st.success("Lunch: Rice + vegetables + protein")
        st.success("Dinner: Light meal (salad/soup)")
        st.success(" Snacks: Fruits, nuts")

    elif risk_score < 60:
        st.warning("Controlled Diet: Low sugar, high protein")
        st.warning("Breakfast: Oats / boiled eggs (no sugar)")
        st.warning("Lunch: Chapati + vegetables + protein")
        st.warning("Dinner: Salad + grilled food")
        st.warning("Snacks: Almonds, walnuts")
        st.warning("Avoid sugar, bakery, soft drinks")

    else:
        st.error("visit doctor urgently")
        st.error("Strict Diabetic Diet: Low-carb, high fiber")
        st.error("Breakfast: Low-carb (eggs, sprouts)")
        st.error("Lunch: High fiber (vegetables, legumes)")
        st.error("Dinner: Soup + salad (very light)")
        st.error("Snacks: Seeds, nuts")
        st.error("STRICTLY avoid sugar & refined carbs")

    st.write("💧 Drink 2–3L water daily")
    st.write("🏃 Exercise regularly")     