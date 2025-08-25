import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.title("Personalized Diet & Meal Planner")
st.markdown("click in arrow to Get a customized  diet plan based on your profile, fitness goal, and dietary preferences.")
@st.cache_data
def load_data():
    df = pd.read_csv("diet0011BB.csv")
    df.columns = df.columns.str.strip()  # Clean column names

    # Clean numeric columns
    numeric_cols = ['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)', 'Fiber (g)',
                    'Sodium (mg)', 'Cholesterol (mg)', 'Glycemic Index']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Clean text columns
    df['Type'] = df['Type'].astype(str).str.title().str.strip()
    df['Cuisine'] = df['Cuisine'].astype(str).str.strip()
    df['Allergens'] = df['Allergens'].fillna('None').astype(str).str.title().str.strip()
    df['Goal'] = df['Goal'].fillna('Weight Loss').astype(str).str.title().str.strip()
    df['Meal Time'] = df['Meal Time'].astype(str).str.strip()
    return df

df = load_data()

# ------------------------ NUTRITION 
def calculate_bmr(weight, height, age, gender):
    return 10*weight + 6.25*height - 5*age + 5 if gender == "Male" else 10*weight + 6.25*height - 5*age - 161

def calculate_tdee(bmr, activity_factor):
    return bmr * activity_factor

def calculate_macros(tdee, goal, weight):
    if goal == "Weight Loss":
        calories = max(1200, tdee - 500)
        protein = min(250, weight * 2.0)
    elif goal == "Weight Gain":
        calories = tdee + 500
        protein = weight * 1.8
    else:  # Bodybuilding
        calories = tdee + 300
        protein = min(300, weight * 2.4)
    return {"calories": round(calories), "protein": round(protein)}

# ------------- ML ----------------
def recommend_dish(candidates, target_cal, target_protein, used_dishes):
    if len(candidates) == 0:
        return None
    candidates = candidates[~candidates["Dish Name"].isin(used_dishes)]
    if len(candidates) == 0:
        return df.sample(1).iloc[0]

    X = candidates[['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)', 'Fiber (g)']].values
    target = [target_cal, target_protein, target_cal*0.45/4, target_cal*0.25/9, 5]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaled = scaler.transform([target])

    model = NearestNeighbors(n_neighbors=1)
    model.fit(X_scaled)
    _, idx = model.kneighbors(target_scaled)
    return candidates.iloc[idx[0][0]]

# ------- USER INPUTS ------------------------
st.sidebar.header("Your Profile")
age = st.sidebar.slider("Age", 18, 80, 28)
gender = st.sidebar.radio("Gender", ["Male", "Female"])
weight = st.sidebar.number_input("Weight (kg)", 30, 200, 70)
height = st.sidebar.number_input("Height (cm)", 100, 220, 175)
activity = st.sidebar.selectbox("Activity", [
    "Sedentary", "Light", "Moderate", "Active", "Very Active"])
goal = st.sidebar.radio("Goal", ["Weight Loss", "Weight Gain", "Bodybuilding"])
diet = st.sidebar.radio("Diet", ["Veg", "Non-Veg"])
allergies = st.sidebar.text_input("Allergies", "None")
region = st.sidebar.selectbox("Region", ["Any", "North Indian", "South Indian", "Gujarati", "Punjabi", "Bengali"])

# Activity map
activity_map = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}

# ------------------------ GENERATE PLAN ------------------------
if st.sidebar.button("Generate 15-Day Plan"):
    with st.spinner("Generating meal plan..."):
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity_map[activity])
        macros = calculate_macros(tdee, goal, weight)

        # Filter dataset
        filtered_df = df.copy()
        if diet == "Veg":
            filtered_df = filtered_df[filtered_df["Type"] == "Veg"]
        else:
            filtered_df = filtered_df[filtered_df["Type"] != "Veg"]

        if allergies != "None":
            allergy_list = [a.strip().lower() for a in allergies.split(",")]
            filtered_df = filtered_df[
                ~filtered_df["Allergens"].str.lower().apply(
                    lambda x: any(a in x.lower() for a in allergy_list)
                )
            ]

        if region != "Any":
            filtered_df = filtered_df[
                filtered_df["Cuisine"].str.contains(region.split()[0], case=False, na=False)
            ]

        # Group by meal time
        meals = {}
        for mt in ["Breakfast", "Lunch", "Snack", "Dinner"]:
            meals[mt] = filtered_df[
                filtered_df["Meal Time"].str.contains(mt, case=False, na=False)
            ]

        used_dishes = set()
        plan = []

        for day in range(1, 16):
            day_plan = {"Day": day}
            daily_cal = 0
            daily_protein = 0

            for meal_type in ["Breakfast", "Lunch", "Snack", "Dinner"]:
                target_cal = macros["calories"] * (0.25 if meal_type == "Breakfast" else 0.35 if meal_type == "Lunch" else 0.15 if meal_type == "Snack" else 0.25)
                target_protein = macros["protein"] * (0.25 if meal_type == "Breakfast" else 0.35 if meal_type == "Lunch" else 0.15 if meal_type == "Snack" else 0.25)
                dish = recommend_dish(meals[meal_type], target_cal, target_protein, used_dishes)

                if dish is not None:
                    day_plan[meal_type] = dish["Dish Name"]
                    used_dishes.add(dish["Dish Name"])
                    daily_cal += dish["Calories"]
                    daily_protein += dish["Protein (g)"]
                else:
                    day_plan[meal_type] = "Oats with Nuts"

            day_plan["Total Calories"] = round(daily_cal)
            day_plan["Protein (g)"] = round(daily_protein)
            plan.append(day_plan)

        # Show results
        st.success(" Plan Generated")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BMR", f"{bmr:.0f} kcal")
        col2.metric("TDEE", f"{tdee:.0f} kcal")
        col3.metric("Target", f"{macros['calories']} kcal")
        col4.metric("Protein", f"{macros['protein']}g")

        # Display plan
        for day_plan in plan:
            with st.expander(f"Day {day_plan['Day']} — {day_plan['Total Calories']} kcal | {day_plan['Protein (g)']}g Protein"):
                c1, c2, c3, c4 = st.columns(4)
                c1.write("**Breakfast**")
                c1.write(day_plan['Breakfast'])
                c2.write("**Lunch**")
                c2.write(day_plan['Lunch'])
                c3.write("**Snack**")
                c3.write(day_plan['Snack'])
                c4.write("**Dinner**")
                c4.write(day_plan['Dinner'])

        # Download
        plan_df = pd.DataFrame(plan)
        csv = plan_df.to_csv(index=False)
        st.download_button("Download Plan", data=csv, file_name="diet-plan.csv", mime="text/csv")
