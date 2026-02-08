# 🎓 PARIX: Student Intervention Analytics

PARIX is an AI-powered Learning Analytics system designed to help educators identify at-risk students and generate data-driven intervention playbooks.

## 🚀 Key Features
- **Effort-Outcome Gap Analysis**: Identifies students underperforming relative to their predicted potential.
- **Persona Clustering**: Categorizes students into behavioral archetypes (e.g., "Overworked Struggler").
- **GenAI Playbooks**: Generates personalized intervention strategies and parent communication drafts using OpenRouter (GLM-4.5).
- **Interactive Visuals**: Real-time risk distribution and performance radar charts.

## 🛠 Setup
1. Clone the repo: `git clone <your-repo-url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `OPENROUTER_API_KEY` to a `.env` file.
4. Run the app: `streamlit run ui/teacher_dashboard.py`

## 📊 Methodology
The system uses **Linear Regression** for score prediction and **K-Means Clustering** for behavioral segmentation, grounded in 16+ socio-economic and academic features.