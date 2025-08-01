import streamlit as st
import pandas as pd
import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from io import BytesIO
from prediction_pipeline import predict_pipeline

st.set_page_config(layout="wide")

st.title("🧱 GenAI Data Firewall: Real-Time Drift Detection & Auto-Fix")

st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 40px;">
        <img 
            src="https://i.postimg.cc/yYwC3ymm/genai-firewall-huggingface-refined.png" 
            width="220px" 
            style="
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                border-radius: 8px;
                max-width: 100%;
                height: auto;
            "
        />
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("<h4 style='text-align: center;'>🤖 Behind the Scenes: Mistral LLM</h4>", unsafe_allow_html=True)
st.sidebar.markdown("""
Mistral-7B is a **powerful open-source language model** optimized for instruction-following.

We use it to **auto-fix bad or missing data** based on schema rules:

- Trained on diverse data  
- Outputs JSON-friendly fixes  
- CPU-compatible, fast, and reliable  

💡 Let the GenAI handle noisy inputs while you focus on insights!
""")

# Cleaned-up "How It Works" panel
st.sidebar.markdown("<h4 style='text-align: center;'>🤖 Inside the GenAI Firewall</h4>", unsafe_allow_html=True)

st.sidebar.markdown("""
**1. Training Data**  
`[{product_id: 'X120', warehouse: 'WH_22', holiday_week: 'Y', demand: 91, delivery_time: 35}, ...]`

**2. Incoming Record**  
`{product_id: 'X120', warehouse: 'WH_22', holiday_week: 'Y', demand: null, delivery_time: '?'}`

**3. Detection**  
🚨 Missing `demand`  
🚨 Invalid `delivery_time`

**4. GenAI Repair**  
🔧 Prompt to LLM with schema + examples  
✅ Fixed: `{demand: 92, delivery_time: 36}`

**5. Model Prediction**  
📈 `{expected_demand_next_week: 101.4, confidence: 0.89}`
""")

st.sidebar.markdown("""
**6. Logging**  
📝 All inputs, fixes, and predictions are tracked for monitoring and analysis.
""")


# Upload files
data_file = st.file_uploader("📄 Upload CSV data file", type="csv")
schema_file = st.file_uploader("📜 Upload JSON schema file", type="json")

if data_file and schema_file:
    df = pd.read_csv(data_file)
    schema = json.load(schema_file)

    st.subheader("Step 1: 🔍 Schema Column Check")
    expected_columns = set(schema.keys())
    actual_columns = set(df.columns)
    missing = list(expected_columns - actual_columns)
    unexpected = list(actual_columns - expected_columns)

    st.json({"missing_columns": missing, "unexpected_columns": unexpected})

    st.subheader("Step 2: 🚨 Detect Row Violations")
    available_schema = {col: rules for col, rules in schema.items() if col in df.columns}
    violations = []

    for idx, row in df.iterrows():
        row_issues = {}
        for col, rules in available_schema.items():
            value = row.get(col, None)
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                row_issues[col] = {"issue": "Missing/blank value"}
            elif rules["type"] == "categorical":
                if not isinstance(value, str) or value not in rules.get("valid_values", []):
                    row_issues[col] = {"issue": "Invalid category", "value": value}
            elif rules["type"] == "numeric":
                try:
                    val = float(value)
                    if val < rules["min_estimate"] or val > rules["max_estimate"]:
                        row_issues[col] = {"issue": "Out of range", "value": val}
                except:
                    row_issues[col] = {"issue": "Not numeric", "value": value}
            elif rules["type"] == "int64":
                try:
                    if not float(value).is_integer():
                        raise ValueError
                except:
                    row_issues[col] = {"issue": "Not integer", "value": value}
        if row_issues:
            violations.append({"row": int(idx), "problems": row_issues})

    st.write(f"Total Violations: {len(violations)}")
    st.json(violations[:3])

    if st.button("🛠️ Run GenAI Auto-Fix"):
        st.subheader("Step 3: 🤖 GenAI Repair")

        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu")
        model.eval()

        def extract_json(text):
            try:
                if "[/INST]" in text:
                    text = text.split("[/INST]", 1)[1].strip()
                return json.loads(text[text.find("{"): text.rfind("}")+1])
            except:
                return None

        fixes = {}
        for v in violations:
            row_id = v["row"]
            fixes[str(row_id)] = {}
            for col, issue in v["problems"].items():
                context = df.iloc[row_id].drop(col).to_dict()
                prompt = f"""[INST]
Fix column '{col}' based on schema and row context.
Row: {json.dumps(context)}
Column: {col}
Rule: {schema[col]}
Format: { {row_id: {col: VALUE}} }
[/INST]"""
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
                    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    parsed = extract_json(result)
                    if parsed and str(row_id) in parsed and col in parsed[str(row_id)]:
                        fixes[str(row_id)][col] = parsed[str(row_id)][col]
                except:
                    pass
                gc.collect()

        for row_id, colfix in fixes.items():
            for col, val in colfix.items():
                df.at[int(row_id), col] = val

        st.success("✅ Fixes Applied")
        st.json(fixes)

        st.subheader("Step 4: 📊 Model Predictions")
        try:
            preds, probs = predict_pipeline(df.copy())
            df["Prediction"] = preds
            df["Prediction_Prob"] = probs
            st.dataframe(df.head())

            pred_csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download CSV", pred_csv, "firewall_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.info("👆 Please upload both a CSV and JSON schema file to proceed.")