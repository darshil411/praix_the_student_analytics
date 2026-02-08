# src/genai/build_payload.py

def assign_risk_level(gap: float) -> str:
    if gap <= -1.0:
        return "High"
    elif gap <= -0.5:
        return "Medium"
    else:
        return "Low"


def build_genai_payload(row: dict) -> dict:
    """
    Converts computed ML outputs into a strict GenAI input contract.
    Assumes ALL features are already computed.
    """

    payload = {
        "persona_label": row["persona_label"],
        "risk_level": assign_risk_level(row["effort_outcome_gap"]),
        "effort_outcome_gap": round(row["effort_outcome_gap"], 2),
        "primary_lever": row["primary_lever"],
        "key_drivers": row[["Sleep_Hours", "Attendance","Hours_Studied"]].values.tolist(),  # list like ["Sleep_Hours", "Attendance"]

        "student_context": {
         
            "School_Type_Public": row["School_Type_Public"],
            "learning_disabilities": row["Learning_Disabilities"]
        }
    }

    return payload