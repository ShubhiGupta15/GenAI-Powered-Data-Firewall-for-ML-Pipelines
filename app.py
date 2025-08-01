import streamlit as st
import pandas as pd
import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import joblib
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




uploaded_file = st.file_uploader("Upload Drifted JSON", type="json")

if uploaded_file:
    file_contents = uploaded_file.read()
    input_json = json.loads(file_contents)

    st.subheader("📤 Drifted Input")
    st.json(input_json)

    if st.button("Run Firewall"):
        with st.spinner("🔧 GenAI is fixing the data... hang tight!"):
            fixed_json = predict_pipeline(input_json)
            st.subheader("✅ Fixed Output")
            st.json(fixed_json)

            # Provide download button
            fixed_json_str = json.dumps(fixed_json, indent=4)
            b = BytesIO()
            b.write(fixed_json_str.encode())
            b.seek(0)
            st.download_button(label="Download Fixed JSON",
                               data=b,
                               file_name="fixed_output.json",
                               mime="application/json")
            gc.collect()

# Upload files
data_file = st.file_uploader("📄 Upload CSV data file", type="csv")
schema_file = st.file_uploader("📜 Upload JSON schema file", type="json")

if data_file and schema_file:
    df = pd.read_csv(data_file)
    schema = json.load(schema_file)

    st.subheader("Step 1: 🔍 Schema Column Check")
    expected_columns = set(schema.keys())
    actual_columns = set(df.columns)

    missing_columns = list(expected_columns - actual_columns)
    unexpected_columns = list(actual_columns - expected_columns)

    column_issues = {
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns
    }

    st.json(column_issues)

    st.subheader("Step 2: 🚨 Row-Level Validation")

    available_schema = {col: rules for col, rules in schema.items() if col in df.columns}
    violations = []

    for index, row in df.iterrows():
        row_issues = {}
        for col, rules in available_schema.items():
            value = row.get(col, None)

            # Missing
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                row_issues[col] = {"issue": "Missing/blank value"}
            elif rules["type"] == "categorical":
                if not isinstance(value, str):
                    row_issues[col] = {
                        "issue": "Wrong data type for categorical",
                        "value": value,
                        "expected_type": "string"
                    }
                elif value not in rules["valid_values"]:
                    row_issues[col] = {
                        "issue": "Invalid category",
                        "value": value,
                        "expected": rules["valid_values"]
                    }
            elif rules["type"] == "numeric":
                try:
                    val = float(value)
                    if val < rules["min_estimate"] or val > rules["max_estimate"]:
                        row_issues[col] = {
                            "issue": "Out of expected range",
                            "value": val,
                            "expected_range": [rules["min_estimate"], rules["max_estimate"]]
                        }
                except:
                    row_issues[col] = {
                        "issue": "Wrong data type",
                        "value": value,
                        "expected_type": "numeric"
                    }
            elif rules["type"] == "datetime":
                try:
                    pd.to_datetime(value)
                except:
                    row_issues[col] = {
                        "issue": "Invalid datetime",
                        "value": value
                    }
            elif rules["type"] == "int64":
                try:
                    val = float(value)
                    if not val.is_integer():
                        raise ValueError
                except:
                    row_issues[col] = {
                        "issue": "Expected integer",
                        "value": value
                    }

        if row_issues:
            violations.append({
                "row": int(index),
                "problems": row_issues
            })

    st.write(f"Total Violations Found: {len(violations)}")
    st.json(violations[:3])  # Show top 3

    if st.button("🛠️ Run GenAI Auto-Fix"):
        st.subheader("Step 3: 🤖 GenAI Fixing")

        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu")
        model.eval()

        def extract_json(text):
            try:
                if "[/INST]" in text:
                    text = text.split("[/INST]", 1)[1].strip()
                json_start = text.find("{")
                json_end = text.rfind("}") + 1
                return json.loads(text[json_start:json_end])
            except Exception:
                return None

        fixes = {}
        progress = st.progress(0)
        total = len(violations)

        for i, violation in enumerate(violations):
            row_id = violation["row"]
            problems = violation["problems"]
            fixes.setdefault(str(row_id), {})

            for col, issue in problems.items():
                rule = schema.get(col)
                if not rule:
                    continue

                try:
                    row_data = df.iloc[row_id].to_dict()
                    context_json = json.dumps({k: v for k, v in row_data.items() if k != col})

                    # Generate rule_str more safely with fallbacks for NaN/missing info
                    if rule["type"] == "numeric":
                        min_val = rule.get("min_estimate")
                        max_val = rule.get("max_estimate")
                        if pd.notna(min_val) and pd.notna(max_val):
                            rule_str = f"Value must be a number between {min_val} and {max_val}."
                        else:
                            rule_str = "Value must be a valid numeric type (e.g., float or int)."

                    elif rule["type"] == "int64":
                        rule_str = "Value must be a whole number (integer)."

                    elif rule["type"] == "categorical":
                        valid_values = rule.get("valid_values") or rule.get("unique_values") or []
                        if isinstance(valid_values, list) and valid_values:
                            rule_str = f"Value must be one of the following: {', '.join(map(str, valid_values))}."
                        else:
                            rule_str = "Value must be a valid categorical option."

                    elif rule["type"] == "datetime":
                        rule_str = "Value must be a valid date or timestamp in the format YYYY-MM-DD or similar."

                    else:
                        rule_str = "Value must follow schema rules."


                    format_hint = f'{{"{row_id}": {{"{col}": VALUE}}}}'

                    prompt = f"""[INST]
You are a data cleaner. Fix column '{col}' using row context and schema rule.

Column: {col}
Rule: {rule_str}

Row context:
{context_json}

Output ONLY a valid JSON in this format:
{format_hint}
[/INST]"""

                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=128,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    fixed_value = extract_json(response)

                    if fixed_value and str(row_id) in fixed_value and col in fixed_value[str(row_id)]:
                        fixes[str(row_id)][col] = fixed_value[str(row_id)][col]
                except Exception as e:
                    st.warning(f"Failed to fix row {row_id}, col {col}: {e}")
                finally:
                    gc.collect()

            progress.progress((i + 1) / total)

        # Apply fixes to df
        for row_id, cols in fixes.items():
            for col, val in cols.items():
                df.at[int(row_id), col] = val

        st.success("✅ Fixes applied!")
        st.json(fixes)



          # Save fixed CSV to buffer
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        # Download fixed file
        st.subheader("⬇️ Download Fixed CSV")
        st.download_button("Download CSV", csv_data, "data_drift_fixed.csv", "text/csv")

        # Save to disk automatically
        fixed_csv_path = "data_drift_fixed_saved.csv"
        with open(fixed_csv_path, "wb") as f:
            f.write(csv_data)
        st.success(f"Fixed CSV saved to {fixed_csv_path}")

        # --- MODEL PREDICTION ---
        st.subheader("📊 Model Prediction on Cleaned Data")

        try:
            preds, probs = predict_pipeline(df.copy())
            df["Prediction"] = preds
            df["Prediction_Prob"] = probs

            st.success("✅ Predictions completed!")
            st.dataframe(df.head())

            pred_csv = df.to_csv(index=False).encode()
            st.download_button("📥 Download Predictions CSV", pred_csv, "data_drift_with_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Prediction failed: {e}")


else:
    st.info("👆 Please upload both a CSV data file and a JSON schema file.")



#   // "Product_Description": {
#   //   "type": "numeric",
#   //   "mean": NaN,
#   //   "std": NaN,
#   //   "min_estimate": NaN,
#   //   "max_estimate": NaN
#   // },