
🧱 GenAI-Powered Data Firewall for ML Pipelines
===============================================================================

Protect your machine learning models from silent failures caused by data drift,
schema mismatches, missing values, and unexpected input — all in real time 
using Generative AI.

“Smart middleware between your raw data and ML model that catches and fixes 
issues before they break your pipeline.”


🚀 Overview
===============================================================================

This project introduces a Generative AI-powered Firewall built to:

- Detect schema changes, drift, and malformed input in structured data
- Auto-fix issues using LLMs based on context + schema rules
- Seamlessly integrate into your ML inference pipeline
- Provide optional human override for business-specific fixes
- Deploy in a few clicks via Streamlit


🧠 Why This Project?
===============================================================================

During real-world ML deployments, pipelines often break not because of bugs — 
but due to *unseen data problems*.

We faced this ourselves during production rollouts and wanted to build a tool that can:

✔️ Detect harmful input early  
✔️ Auto-repair intelligently using context  
✔️ Preserve model reliability  
✔️ Explain what changed (transparently)


📦 Features
===============================================================================

- 🔍 Schema Validator: Detect missing, extra, or mismatched columns
- 🚨 Row-Level Checker: Spot missing values, out-of-range data, type mismatches
- 🤖 GenAI Auto-Fix: Use an LLM to fix data based on context and schema rules
- 🖼️ SHAP-Based Explanations:Model transparency for predictions
- 🖥️ Streamlit UI: Upload files, view violations, fix and download in seconds


🧪 Use Case
===============================================================================

This firewall was built on top of an ML model that predicts **late delivery risk** 
in logistics. It ensures:

- Data quality before inference  
- Robust and drift-resistant predictions  
- Real-time repair of schema issues  
- Flexibility for human decision overrides


🧰 Tech Stack
===============================================================================

- Python  
- Streamlit  
- Hugging Face Transformers (Mistral-7B-Instruct-v0.2)  
- Pandas, Torch  
- XGBoost  
- SHAP


📄 Inputs
===============================================================================

CSV File:
- Your structured input data file

JSON Schema:
- Defines expected columns, types, allowed ranges or valid categories

Example schema format:
{
  "Benefit_per_order": {
    "type": "numeric",
    "min_estimate": 0,
    "max_estimate": 5000
  },
  
  "Shipping_Mode": {
    "type": "categorical",
    "valid_values": ["Air", "Ground", "Ship"]
  }
}


⚙️ GenAI Model
===============================================================================

The firewall uses a Generative AI model (Mistral-7B) to suggest context-aware fixes.  
Prompts include:
- Column name  
- Schema rule  
- Row context (excluding the broken column)

Model output:
{
  "12": {
    "Benefit_per_order": 140.5
  }
}


🧠 Predict Using Your Cleaned Data
===============================================================================

Once repaired, the data is passed through a trained ML model to generate predictions.

- Outputs include predicted labels and probabilities
- Cleaned + labeled data is downloadable as CSV


📥 Try It Out
===============================================================================

Try it, fork it, break it:

GitHub: https://github.com/ShubhiGupta15/GenAI-Powered-Data-Firewall-for-ML-Pipelines.git

Streamlit App: https://genai-powered-data-firewall-for-ml-pipelines-e5xjyojzbicwfrfpf.streamlit.app/

Sample Files: `data.csv`, `schema.json`

#GenAI #DataDrift #Streamlit #MachineLearning #MLOps #XGBoost #LLM #SHAP #SmartMiddleware #DataFirewall #PythonAI #OpenSourceAI
