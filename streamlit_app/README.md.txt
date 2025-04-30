OnTrust AI
==========

OnTrust AI is a simple app I built to help companies assess supplier risk using machine learning and document verification. 
The goal is to make vendor onboarding faster and safer by using predictive models and some LLM-based checks.

What it does
------------

- Lets you upload supplier data (manually or using a CSV)
- Predicts the risk level (High, Medium, Low, or Needs Review) using pre-trained ML models
- Verifies supplier documents (like PDFs or zipped folders) using LLMs
- Explains why a supplier was flagged using SHAP values

Tech used
---------

- Streamlit (frontend)
- XGBoost + scikit-learn (ML models)
- SHAP (for explainability)
- Some basic LLM logic for document checks
- Everything is in Python

To run locally
--------------

git clone https://github.com/bharath2197/ontrust-ai-platform.git
cd ontrust-ai-platform/app/streamlit_app
pip install -r requirements.txt
streamlit run ontrustai_mvp.py

What's next
-----------

- Improve the LLM verification logic
- Add an API version
- Build a dashboard for post-onboarding monitoring
- Clean up the UI


