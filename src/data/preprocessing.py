import pandas as pd


def add_student_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a unique Student_ID in the format STUD0001, STUD0002, etc.
    """
    df = df.copy()
    # Create IDs based on the row index
    df['Student_ID'] = [f"STUD{i+1:04d}" for i in range(len(df))]
    
    # Move Student_ID to the first column position
    cols = ['Student_ID'] + [col for col in df.columns if col != 'Student_ID']
    return df[cols]
def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[
            "Teacher_Quality",
            "Parental_Education_Level",
            "Distance_from_Home",
        ]
    )


def encode_ordinal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ordinal_maps = {
        "Parental_Involvement": {"Low": 0, "Medium": 1, "High": 2},
        "Access_to_Resources": {"Low": 0, "Medium": 1, "High": 2},
        "Motivation_Level": {"Low": 0, "Medium": 1, "High": 2},
        "Family_Income": {"Low": 0, "Medium": 1, "High": 2},
        "Peer_Influence": {"Negative": 0, "Neutral": 1, "Positive": 2},
    }

    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping)

    return df


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    binary_maps = {
        "Internet_Access": {"No": 0, "Yes": 1},
        "Extracurricular_Activities": {"No": 0, "Yes": 1},
        "Learning_Disabilities": {"No": 0, "Yes": 1},
        "Gender": {"Female": 0, "Male": 1},
        "School_Type": {"Private": 0, "Public": 1}
    }

    for col, mapping in binary_maps.items():
        df[col] = df[col].map(mapping)

    df = df.rename(columns={"School_Type": "School_Type_Public"})
    return df

def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Exam_Score",
):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y