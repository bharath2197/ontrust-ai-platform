import streamlit as st
import pandas as pd
import pickle
import pytesseract
from PIL import Image
import ollama
import os
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import zipfile
import io

# --- Load Models ---
model_supplier = None
model_restaurant = None

if os.path.exists("models/ontrust_xgb_model.pkl"):
    with open("models/ontrust_xgb_model.pkl", "rb") as f:
        model_supplier = pickle.load(f)

if os.path.exists("models/restaurant_xgb_model.pkl"):
    with open("models/restaurant_xgb_model.pkl", "rb") as f:
        model_restaurant = pickle.load(f)

# --- SHAP Explainers ---
supplier_explainer = None
restaurant_explainer = None

if model_supplier:
    import shap
    supplier_explainer = shap.TreeExplainer(model_supplier)

if model_restaurant:
    import shap
    restaurant_explainer = shap.TreeExplainer(model_restaurant)

# --- Label Mappings ---
compliance_map = {"Yes": 1, "No": 0}
country_map = {"USA": 3, "India": 2, "China": 0, "Germany": 1}
ecomm_category_map = {"Lighting": 1, "Furniture": 2, "Decor": 3, "Electronics": 4, "Apparel": 5, "Home Improvement": 6, "Sports Equipment": 7, "Beauty & Personal Care": 8, "Automotive Parts": 9, "Industrial Supplies": 10, "Other": 11}
restaurant_category_map = {"Fine Dining": 0, "Fast Food": 1, "Bakery": 2, "Café": 3, "Other": 4}
risk_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# --- Utility Functions ---
def auto_rotate_image(image):
    try:
        osd = pytesseract.image_to_osd(image)
        rotate_angle = 0
        if "Rotate: 90" in osd:
            rotate_angle = 270
        elif "Rotate: 180" in osd:
            rotate_angle = 180
        elif "Rotate: 270" in osd:
            rotate_angle = 90
        if rotate_angle != 0:
            image = image.rotate(rotate_angle, expand=True)
        return image
    except:
        return image

def clean_image_for_ocr(pil_img):
    img = np.array(pil_img)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(clean)

# --- Streamlit App Setup ---
st.set_page_config(page_title="OnTrust AI - Vendor Risk & Verification Platform", layout="wide")
st.title("OnTrust AI - Unified Vendor Risk & Verification Platform")

# --- Tabs Setup ---
tab1, tab2, tab3, tab4 = st.tabs(["Vendor Info Upload", "Risk Scoring", "Document Verification", "Bulk Upload Center"])

# --- Session State Init ---
for key in ["vendor_info", "risk_score_label", "risk_reasons", "verification_result", "document_reasons"]:
    if key not in st.session_state:
        st.session_state[key] = {} if "info" in key or "reasons" in key else None

# --- Tab 1: Vendor Info Upload ---
with tab1:
    st.header("Vendor Information Upload")
    st.write("📋 Please fill in the vendor's details below:")

    vendor_type = st.radio("Select Vendor Type", ["E-commerce Supplier", "Restaurant / Food Vendor"])
    vendor_name = st.text_input("Vendor Name")
    country = st.selectbox("Country", list(country_map.keys()))
    if vendor_type == "E-commerce Supplier":
        category = st.selectbox("E-commerce Category", list(ecomm_category_map.keys()))
    else:
        category = st.selectbox("Restaurant Category", list(restaurant_category_map.keys()))

    if st.button("Save Vendor Info"):
        st.session_state['vendor_info'] = {
            'vendor_type': vendor_type,
            'vendor_name': vendor_name,
            'country': country,
            'category': category
        }
        st.success("✅ Vendor Info Saved! Please move to Risk Scoring tab ➡️")

# --- Tab 2: Risk Scoring (updated with SHAP reasons) ---
with tab2:
    st.header("Vendor Risk Scoring")

    if st.session_state['vendor_info']:
        vendor = st.session_state['vendor_info']

        risk_reasons = []
        if vendor['vendor_type'] == "Restaurant / Food Vendor":
            health_rating = st.slider("Health Rating", 1.0, 5.0, 4.0)
            food_compliance_docs = st.radio("Food Compliance Docs Provided?", ["Yes", "No"])
            years_operating = st.slider("Years Operating", 0, 30, 2)
            delivery_timeliness = st.slider("Delivery Timeliness (%)", 50, 100, 85)
            customer_complaints = st.slider("Customer Complaints", 0, 10, 0)
            violations = st.slider("Health Violations", 0, 10, 0)

            if model_restaurant and restaurant_explainer:
                X = pd.DataFrame([[health_rating, compliance_map.get(food_compliance_docs, 0), years_operating, delivery_timeliness, customer_complaints, violations, 1]],
                                 columns=["health_rating", "food_compliance_docs", "years_operating", "delivery_timeliness", "customer_complaints", "violations", "category"])
                risk_score = model_restaurant.predict(X)[0]
                st.session_state['risk_score_label'] = risk_map[risk_score]

                # SHAP explanation
                shap_values = restaurant_explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_for_class = shap_values[risk_score][0]
                else:
                    shap_for_class = shap_values[0]
                top_features_idx = np.argsort(-np.abs(shap_for_class))[0][:3]
                risk_reasons = [str(X.columns[i]) for i in top_features_idx]

        else:
            delivery_rate = st.slider("Delivery Rate (%)", 0, 100, 70)
            avg_rating = st.slider("Average Rating", 1.0, 5.0, 3.0)
            compliance_docs = st.radio("Compliance Documents Provided?", ["Yes", "No"])
            business_age = st.slider("Business Age (Years)", 0, 30, 2)
            past_incidents = st.slider("Past Incidents", 0, 10, 0)

            if model_supplier and supplier_explainer:
                X = pd.DataFrame([[delivery_rate, avg_rating, compliance_map.get(compliance_docs, 0), business_age, past_incidents, country_map.get(vendor['country'], 3), ecomm_category_map.get(vendor['category'], 11)]],
                                 columns=["delivery_rate", "avg_rating", "compliance_docs", "business_age", "past_incidents", "country", "product_category"])
                risk_score = model_supplier.predict(X)[0]
                st.session_state['risk_score_label'] = risk_map[risk_score]

                # SHAP explanation
                shap_values = supplier_explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_for_class = shap_values[risk_score][0]
                else:
                    shap_for_class = shap_values[0]
                top_features_idx = np.argsort(-np.abs(shap_for_class))[0][:3]
                risk_reasons = [str(X.columns[i]) for i in top_features_idx]

        st.metric("Predicted Risk Tier", st.session_state['risk_score_label'])

        if risk_reasons:
            st.success(f"Top Risk Reason(s): {'; '.join(risk_reasons)}")

        st.session_state['risk_reasons'] = risk_reasons

    else:
        st.info("📋 Please complete Vendor Info Upload first.")

# --- Tab 3: Document Verification ---
with tab3:
    st.header("Vendor Document Verification")
    uploaded_doc = st.file_uploader("Upload Business License / Tax Certificate", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_doc:
        mime_type = uploaded_doc.type

        if mime_type == "application/pdf":
            images = convert_from_bytes(uploaded_doc.read(), poppler_path="/usr/bin/")
            image = images[0]
        elif mime_type.startswith("image/"):
            image = Image.open(uploaded_doc)
        else:
            st.error("Unsupported file type.")
            st.stop()

        image = auto_rotate_image(image)
        image = clean_image_for_ocr(image)

        extracted_text = pytesseract.image_to_string(image)
        st.text_area("Extracted Text", extracted_text, height=200)

        if extracted_text.strip() != "":
            llm_prompt = f"""
You are verifying a vendor's business license or tax certificate. Please answer:

- Is the document valid (contains business name, license/registration number, authority name)?
- If most key fields (>70%) are present, respond: "Valid document: Yes"
- If major fields missing, respond: "Valid document: No"

Document text:
{extracted_text}
"""
            llm_output = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': llm_prompt}])
            result = llm_output['message']['content']
            st.text_area("LLM Validation Result", result, height=200)

            if "Valid document: Yes" in result:
                st.session_state['verification_result'] = "✅ Verified"
            else:
                st.session_state['verification_result'] = "❌ Needs Review"
                if "missing" in result.lower():
                    st.session_state['document_reasons'].append("Missing Key Document Details")

            st.metric("Document Verification", st.session_state['verification_result'])
    else:
        st.info("📄 Please upload a document for verification.")

# --- Tab 4: Bulk Upload Center ---
with tab4:
    st.header("Bulk Upload Center - ML Risk + Document Verification")

    vendor_type = st.radio("Select Vendor Type:", ["E-commerce Supplier", "Restaurant / Food Vendor", "Other"])

    vendor_csv = st.file_uploader("Upload Vendor Details CSV", type=["csv"], key="csv_upload")
    doc_zip = st.file_uploader("Upload Documents Zip", type=["zip"], key="zip_upload")

    if vendor_csv and doc_zip:
        vendor_df = pd.read_csv(vendor_csv)

        # --- Vendor Type Validation ---
        uploaded_vendor_types = vendor_df['vendor_type'].unique()

        if len(uploaded_vendor_types) > 1:
            st.error("⚠️ Error: Mixed vendor types found in uploaded CSV. Please upload only Suppliers or only Restaurants.")
            st.stop()
        elif uploaded_vendor_types[0] != vendor_type:
            st.error(f"❌ Error: Uploaded vendor type '{uploaded_vendor_types[0]}' does not match selected '{vendor_type}'. Please upload the correct file.")
            st.stop()
        else:
            st.success("✅ Vendor type matches selection. Proceeding...")

        # --- Process Documents ---
        docs = {}
        with zipfile.ZipFile(doc_zip, "r") as z:
            for filename in z.namelist():
                base_filename = os.path.splitext(os.path.basename(filename))[0].lower().strip()
                with z.open(filename) as f:
                    docs[base_filename] = f.read()

        # --- ZIP Document Validation ---
        vendor_names = vendor_df['vendor_name'].str.lower().str.strip().tolist()
        matched_docs = [name for name in vendor_names if name in docs.keys()]
        match_rate = len(matched_docs) / len(vendor_names)

        if match_rate < 0.8:  # Less than 80% vendors have documents
            st.error(f"❌ Error: Only {int(match_rate*100)}% of vendor documents found in uploaded ZIP. Please upload the correct document zip.")
            st.stop()
        else:
            st.success(f"✅ {int(match_rate*100)}% vendor documents matched. Proceeding...")

        results = []

        for idx, row in vendor_df.iterrows():
            vendor_name = row['vendor_name']
            matched_doc = None
            for key in docs.keys():
                if key in vendor_name.lower():
                    matched_doc = docs[key]
                    break

            # --- Model Selection Based on Vendor Type ---
            risk_score = None
            reason_list = []

            try:
                if vendor_type == "E-commerce Supplier" and model_supplier:
                    X_row = pd.DataFrame([[
                        row['delivery_rate'],
                        row['avg_rating'],
                        1 if row['compliance_docs'] == "Yes" else 0,
                        row['business_age'],
                        row['past_incidents']
                    ]], columns=["delivery_rate", "avg_rating", "compliance_docs", "business_age", "past_incidents"])
                    risk_score = model_supplier.predict(X_row)[0]

                    # --- SHAP for Supplier ---
                    explainer = shap.TreeExplainer(model_supplier)
                    shap_values = explainer.shap_values(X_row)

                    if isinstance(shap_values, list):
                        shap_for_class = shap_values[risk_score][0]
                    else:
                        shap_for_class = shap_values[0]

                    important_features = X_row.columns[abs(shap_for_class) > 0.1]
                    reason_list = important_features.tolist()

                elif vendor_type == "Restaurant / Food Vendor" and model_restaurant:
                    X_row = pd.DataFrame([[
                        row['health_rating'],
                        1 if row['food_compliance_docs'] == "Yes" else 0,
                        row['years_operating'],
                        row['delivery_timeliness'],
                        row['customer_complaints'],
                        row['violations']
                    ]], columns=["health_rating", "food_compliance_docs", "years_operating", "delivery_timeliness", "customer_complaints", "violations"])
                    risk_score = model_restaurant.predict(X_row)[0]

                    # --- SHAP for Restaurant ---
                    explainer = shap.TreeExplainer(model_restaurant)
                    shap_values = explainer.shap_values(X_row)

                    if isinstance(shap_values, list):
                        shap_for_class = shap_values[risk_score][0]
                    else:
                        shap_for_class = shap_values[0]

                    important_features = X_row.columns[abs(shap_for_class) > 0.1]
                    reason_list = important_features.tolist()

                else:
                    risk_score = 1  # Default Medium if something wrong

                risk_label = risk_map.get(risk_score, "Medium Risk")

            except Exception as e:
                risk_label = "Medium Risk"
                reason_list = ["Prediction Error"]

            # --- Document Verification ---
            doc_valid = False
            if matched_doc:
                try:
                    mime_type = "application/pdf" if matched_doc[:4] == b'%PDF' else "image/jpeg"
                    if mime_type == "application/pdf":
                        images = convert_from_bytes(matched_doc, poppler_path="/usr/bin/")
                        image = images[0]
                    else:
                        image = Image.open(io.BytesIO(matched_doc))

                    image = auto_rotate_image(image)
                    image = clean_image_for_ocr(image)
                    extracted_text = pytesseract.image_to_string(image)

                    llm_prompt = f"""
You are verifying a vendor's business license or tax certificate. Please answer:

- Is the document valid (contains business name, license/registration number, authority name)?
- If most key fields (>70%) are present, respond: "Valid document: Yes"
- If major fields missing, respond: "Valid document: No"

Document text:
{extracted_text}
"""
                    llm_output = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': llm_prompt}])
                    result = llm_output['message']['content']

                    doc_valid = "Valid document: Yes" in result

                except Exception as e:
                    doc_valid = False

            document_status = "Verified" if doc_valid else "Needs Review"

            # --- Final Decision ---
            if risk_label == "Low Risk" and document_status == "Verified":
                final_decision = "Fast-Tracked"
            elif risk_label == "High Risk" and document_status == "Needs Review":
                final_decision = "Rejected"
            else:
                final_decision = "Needs Manual Review"

            results.append({
                'Vendor Name': vendor_name,
                'Vendor Type': vendor_type,
                'Risk Tier': risk_label,
                'Document Verification': document_status,
                'Final Decision': final_decision,
                'Risk Reason(s)': "; ".join(reason_list)
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Result CSV", data=csv, file_name="bulk_upload_results.csv", mime="text/csv")

    else:
        st.info("Please upload both Vendor CSV and Document Zip.")



