import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Import your existing logic
from src.data.preprocessing import (
    add_student_id,
    drop_unused_columns,
    encode_ordinal_features,
    encode_binary_features
)
from src.models.exam_score_model import train_exam_score_model

def create_joblib_files():
    # 1. Load Data
    data_path = "notebooks/Student_data.csv"
    if not os.path.exists(data_path):
        print("Error: Student_data.csv not found in notebooks folder.")
        return
    
    df = pd.read_csv(data_path)

    # 2. Preprocess
    df = add_student_id(df)
    df = drop_unused_columns(df)
    df = encode_ordinal_features(df)
    df = encode_binary_features(df)
    
    # 3. Define the exact features used by the dashboard
    feature_cols = [
        "Student_ID","Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", 
        "Tutoring_Sessions", "Physical_Activity", "Internet_Access", 
        "Extracurricular_Activities", "Learning_Disabilities", "Gender", 
        "School_Type_Public", "Parental_Involvement", "Access_to_Resources", 
        "Motivation_Level", "Family_Income", "Peer_Influence"
    ]
    
    X = df[feature_cols]
    y = df["Exam_Score"]

    # 4. Scale and Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Using your existing function from exam_score_model.py
    model = train_exam_score_model(X_scaled, y)

    # 5. Save Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/exam_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    
    print("✅ Success! Created models/exam_model.joblib and models/scaler.joblib")

if __name__ == "__main__":
    create_joblib_files()
