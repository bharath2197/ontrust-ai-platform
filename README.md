# OnTrust AI

OnTrust AI helps companies assess supplier risk using machine learning and LLM-based document verification — making vendor onboarding faster, safer, and more explainable.

---

## What it does

- Upload supplier data manually or via CSV
- Predicts risk level (**High / Medium / Low / Needs Review**) using pre-trained ML models
- Verifies supplier documents (PDFs, zip folders) using LLM-based checks
- Explains why a supplier was flagged using SHAP values for full transparency

---

## Tech used

| Tool | Purpose |
|------|---------|
| Python | Core logic and backend |
| Streamlit | Interactive frontend |
| XGBoost + scikit-learn | ML risk prediction models |
| SHAP | Model explainability |
| LLM integration | Document verification logic |

---

## To run locally

```bash
git clone https://github.com/bharath2197/ontrust-ai-platform.git
cd ontrust-ai-platform/app/streamlit_app
pip install -r requirements.txt
streamlit run ontrustai_mvp.py
```

---

## What's next

- Improve LLM verification logic
- Add REST API version
- Build post-onboarding monitoring dashboard
- UI polish and cleanup
