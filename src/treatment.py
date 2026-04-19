import joblib


# 🔹 Risk Score Calculation (multi-factor based)
def calculate_risk_score(input_data):
    glucose = input_data[1]
    bp = input_data[2]
    bmi = input_data[5]
    age = input_data[7]

    score = 0

    # Glucose contribution
    if glucose >= 200:
        score += 40
    elif glucose >= 140:
        score += 30
    elif glucose >= 100:
        score += 15

    # BMI contribution
    if bmi >= 30:
        score += 25
    elif bmi >= 25:
        score += 15

    # Blood Pressure
    if bp >= 90:
        score += 15
    elif bp >= 80:
        score += 10

    # Age
    if age >= 50:
        score += 15
    elif age >= 35:
        score += 10

    return min(score, 100)


# 🔹 Treatment Recommendation
def recommend_treatment(prediction, input_data):
    score = calculate_risk_score(input_data)

    print("\n---  Treatment Recommendation ---")
    print(f"Risk Score: {score}/100")

    if score < 30:
        level = "🟢 Low Risk"
    elif score < 60:
        level = "🟡 Moderate Risk"
    else:
        level = "🔴 High Risk"

    print(level)

    # Personalized insights
    print("\n Personalized Insights:")

    glucose = input_data[1]
    bmi = input_data[5]
    bp = input_data[2]

    if glucose > 140:
        print("• High glucose → reduce sugar & refined carbs")

    if bmi > 25:
        print("• High BMI → focus on weight loss & exercise")

    if bp > 80:
        print("• Elevated BP → reduce salt intake & stress")

    print("• Maintain regular physical activity")
    print("• Monitor health metrics regularly")

    return score  # 🔥 return score for diet plan


# 🔹 Diet Plan Generator
def generate_diet_plan(input_data, risk_score):
    glucose = input_data[1]
    bmi = input_data[5]

    print("\n--- Personalized Diet Plan ---") 

    # Low Risk
    if risk_score < 30:
        print("🥗 Balanced Diet:") 
        print("• Breakfast: Oats / fruits / whole grains")
        print("• Lunch: Rice + vegetables + protein")
        print("• Dinner: Light meal (salad/soup)")
        print("• Snacks: Fruits, nuts")

    # Moderate Risk
    elif risk_score < 60:
        print("🥦 Controlled Diet:")
        print("• Breakfast: Oats / boiled eggs (no sugar)")
        print("• Lunch: Chapati + vegetables + protein")
        print("• Dinner: Salad + grilled food")
        print("• Snacks: Almonds, walnuts")
        print("• Avoid sugar, bakery, soft drinks")

    # High Risk
    else:
        print("⚠️ Strict Diabetic Diet:")
        print(" Visit The Doctor Urgently")  
        print("• Breakfast: Low-carb (eggs, sprouts)")
        print("• Lunch: High fiber (vegetables, legumes)")
        print("• Dinner: Soup + salad (very light)")
        print("• Snacks: Seeds, nuts")
        print("• STRICTLY avoid sugar & refined carbs")

    # Personalized adjustments
    print("\n📌 Personalized Adjustments:")

    if glucose > 140:
        print("• Avoid high glycemic foods (white rice, sweets)")

    if bmi > 25:
        print("• Follow calorie deficit for weight loss")

    print("• Drink 2–3 liters of water daily")
    print("• Eat meals at regular intervals")


# 🔹 MAIN FUNCTION (User Input + Prediction + Treatment + Diet)
def run_treatment_system():
    print("\n=== Diabetes Risk & Treatment System ===")

    # User input
    pregnancies = int(input("Pregnancies: "))
    glucose = float(input("Glucose: "))
    bp = float(input("Blood Pressure: "))
    skin = float(input("Skin Thickness: "))
    insulin = float(input("Insulin: "))
    bmi = float(input("BMI: "))
    dpf = float(input("Diabetes Pedigree Function: "))
    age = int(input("Age: "))

    user_data = [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]]

    # Load model & scaler
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("xgb_model.pkl")

    # Scale input
    user_scaled = scaler.transform(user_data)

    # Prediction
    prediction = model.predict(user_scaled)[0]
    probability = model.predict_proba(user_scaled)[0][1]

    print("\n--- Prediction Result ---")
    print(f"Prediction (0 = Not Diabetic, 1 = Diabetic): {int(prediction)}")
    print(f"Risk Probability: {probability:.2f}")

    # Treatment
    risk_score = recommend_treatment(int(prediction), user_data[0])

    # Diet Plan 🔥
    generate_diet_plan(user_data[0], risk_score)


# 🔹 Run
if __name__ == "__main__":
    run_treatment_system()    