import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

new_data = []
def load_data():
    return pd.read_csv("students_data.csv")

def assess_risk(student):
    if student["CGPA"] < 6.0 or student["Carry Overs"] > 0 or student["Attendance"] < 60 or student["Behavior"] == "Poor":
        return "At Risk"
    return "Safe"

def train_model(data):
    X = data[["Marks", "CGPA", "Carry Overs", "Attendance"]]
    y = data["Carry Overs"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, scaler

students_df = load_data()
students_df["Risk Assessment"] = students_df.apply(assess_risk, axis=1)
model, scaler = train_model(students_df)

st.set_page_config(page_title="College Examination System", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard", "Student Details", "Predict Carry Over"])

if page == "Home":
    st.image("college_logo.jpg", width=700)
    st.title("🏛️ Dronacharya Group of Institutions")
    st.write("## Location: Gautam Budh Nagar, Greater Noida")
    st.write(
        "Dronacharya Group of Institutions is the best in academic excellence and innovation. This system helps students track their acadeamic progress and predict their performance."
    )
    st.write("Features:")
    st.markdown("📊 Interactive Dashboard")
    st.markdown("📝 Student Details Management")
    st.markdown("🔮 Carry Over Prediction Model")
    st.markdown("📈 CGPA Analysis")


elif page == "Dashboard":
    st.title("📊 College Examination System Dashboard")
    st.write("Overview of Student Performance")
    st.dataframe(students_df)

elif page == "Student Details":
    st.title("📝 Fill Student Details")
    with st.form("student_form"):
        name = st.text_input("Name")
        roll_number = st.number_input("Roll Number", min_value=1, step=1)
        course = "B.Tech"
        branch = st.selectbox("Branch", ["CSE", "ECE", "ME", "Civil"])
        year = st.selectbox("Year", [1, 2, 3, 4])
        semester = st.selectbox("Semester", [1, 2])
        marks = st.number_input("Marks", min_value=0, max_value=100)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, format="%.2f")
        carry_overs = st.number_input("Carry Overs", min_value=0, step=1)
        attendance = st.number_input("Attendance %", min_value=0, max_value=100)
        behavior = st.selectbox("Behavior", ["Good", "Average", "Poor"])
        submit = st.form_submit_button("Submit")
        if submit:
            new_data.append((name, roll_number, course, branch, year, semester, marks, cgpa, carry_overs, attendance, behavior))
            new_student = pd.DataFrame(new_data, columns=[
        "Name", "Roll Number", "Course", "Branch", "Year", "Semester",
        "Marks", "CGPA", "Carry Overs", "Attendance", "Behavior"
    ])
            students_df = pd.concat([students_df, new_student], ignore_index=True)
            students_df.to_csv("students_data.csv", index=False)
            st.success("Student details added successfully!")

elif page == "Predict Carry Over":
    st.title("📈 Predict Carry Over")
    with st.form("prediction_form"):
        prev_marks = st.number_input("Previous Semester Marks", min_value=0, max_value=100)
        prev_cgpa = st.number_input("Previous CGPA", min_value=0.0, max_value=10.0, format="%.2f")
        prev_carry_overs = st.number_input("Previous Carry Overs", min_value=0, step=1)
        attendance = st.number_input("Attendance %", min_value=0, max_value=100)
        year = st.selectbox("Year", [1, 2, 3, 4])
        semester = st.selectbox("Semester", [1, 2])
        predict = st.form_submit_button("Predict")

        if predict:
            if year == 1 and semester == 1:
                st.warning("Prediction cannot be done for 1st Year, 1st Semester students.")
            else:
                input_data = np.array([[prev_marks, prev_cgpa, prev_carry_overs, attendance]])
                input_data = scaler.transform(input_data)
                predicted_carry_overs = model.predict(input_data)[0]
                st.write(f"Predicted Carry Overs for Next Semester: {predicted_carry_overs}")