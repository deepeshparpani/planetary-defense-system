import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Configure Page
st.set_page_config(page_title="Planetary Defense DSS", page_icon="🛡️", layout="wide")

# Handle Docker vs Local URLs
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# --- HEADER SECTION ---
st.title("🛡️ Planetary Defense Decision Support System (DSS)")
st.markdown("""
### Graduate Portfolio Project: Near-Earth Object (NEO) Hazard Detection
**Model Specs:** SMOTE-Balanced XGBoost | **Recall Optimization:** 99.2% | **Explainability:** Log-Normalized Feature Contribution
---
""")

# --- GLOSSARY & IMPACT ANALYSIS ---
with st.expander("📚 Understanding Orbital Attributes & Hazard Impact"):
    st.markdown("""
    | Attribute | Definition | Impact on Hazard Assessment |
    | :--- | :--- | :--- |
    | **Est. Diameter** | Physical size of the object (km). | **High:** Larger mass directly increases potential impact energy. |
    | **Relative Velocity** | Speed relative to Earth (km/h). | **Critical:** Energy scales with the *square* of velocity ($KE = 1/2 mv^2$). |
    | **Miss Distance** | Minimum distance to Earth's center (km). | **High:** Lower distances increase gravitational capture risk. |
    | **Absolute Magnitude** | Intrinsic brightness (H). | **Medium:** Inverse proxy for size; lower values indicate larger bodies. |
    """)

# --- HISTORICAL TEMPLATE EXAMPLES ---
st.subheader("☄️ Preset Object Templates")
templates = {
    "Manual Entry": {"diam": 0.15, "vel": 45000, "dist": 1000000, "mag": 22.0},
    "99942 Apophis (High Risk)": {"diam": 0.37, "vel": 108000, "dist": 31000, "mag": 19.7},
    "101955 Bennu (Significant)": {"diam": 0.49, "vel": 101000, "dist": 200000, "mag": 20.2},
    "2023 DZ2 (Moderate)": {"diam": 0.07, "vel": 28000, "dist": 175000, "mag": 24.2}
}

selected_name = st.selectbox("Choose a known asteroid template to auto-populate parameters:", list(templates.keys()))
template = templates[selected_name]

# --- SIDEBAR: PARAMETER INPUT ---
st.sidebar.header("☄️ Asteroid Orbital Parameters")
diam = st.sidebar.number_input("Est. Min Diameter (km)", value=template["diam"], step=0.01)
vel = st.sidebar.number_input("Relative Velocity (km/h)", value=template["vel"], step=1000)
dist = st.sidebar.number_input("Miss Distance (km)", value=template["dist"], step=100000)
mag = st.sidebar.slider("Absolute Magnitude (H)", 10.0, 30.0, template["mag"])

# --- DATA INSIGHTS ---
col_stats, col_viz = st.columns([1, 2])
with col_stats:
    st.subheader("🔢 Engineered Features")
    k_proxy = (vel**2) * diam
    s_dist = diam / (dist + 1e-5)
    st.write(f"**Kinetic Proxy:** `{k_proxy:,.0f}`")
    st.write(f"**Size-to-Distance Ratio:** `{s_dist:.2e}`")

with col_viz:
    st.subheader("📊 Relative Feature Influence (Log-Normalized)")
    
    # MASTER'S LEVEL LOGIC: Log-scaling prevents massive values from dominating the UI.
    # We apply log10 to compress the scale while preserving the relative ranking of drivers.
    influence_map = {
        'Velocity': np.log10(vel + 1),
        'Diameter': np.log10((diam * 1000) + 1),         # Scaled km to m for better log granularity
        'Distance': np.log10((2000000 / (dist + 1)) + 1), # Inverse log: closer distance = higher risk
        'Magnitude': np.log10((30 - mag) + 1),
        'Kinetic Proxy': np.log10(k_proxy + 1)
    }

    labels = list(influence_map.keys())
    log_values = np.array(list(influence_map.values()))
    
    # Normalizing log values to percentages for the Donut Chart
    log_values = np.maximum(log_values, 0.1) 
    normalized_percentages = (log_values / log_values.sum()) * 100

    # Create Donut Chart
    fig, ax = plt.subplots(figsize=(4, 4))
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0']
    
    ax.pie(
        normalized_percentages, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        wedgeprops=dict(width=0.45),
        textprops={'fontsize': 8}
    )
    
    # Center text for a professional look
    ax.text(0, 0, 'Risk\nFactors', ha='center', va='center', fontweight='bold')
    ax.set_title("Normalized Factor Contribution", fontsize=10)
    
    st.pyplot(fig)
    st.caption("Log-normalization is applied to visualize the relative impact of features across disparate numerical scales.")

# --- PREDICTION LOGIC ---
st.divider()
if st.button("🚀 Run Comprehensive Hazard Assessment", use_container_width=True):
    payload = {"est_diameter_min": diam, "relative_velocity": vel, "miss_distance": dist, "absolute_magnitude": mag}
    
    try:
        with st.spinner("Querying Inference Engine..."):
            response = requests.post(f"{BACKEND_URL}/predict", json=payload)
            res = response.json()
        
        prob_val = float(res["probability"].strip('%')) / 100
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Hazard Probability", res["probability"])
            st.progress(prob_val)
            
        with res_col2:
            if res["is_hazardous"]:
                st.error("### 🚨 CLASSIFICATION: POTENTIALLY HAZARDOUS")
            else:
                st.success("### ✅ CLASSIFICATION: NON-HAZARDOUS")

        # PHYSICS INSIGHTS
        st.subheader("🧠 Model Decision Logic & Physics Insights")
        if k_proxy > 1000000:
            st.warning(f"**Critical Energy Alert:** High Kinetic Proxy (`{k_proxy:,.0f}`).")
        
        st.info(f"**Statistical Note:** System tuned for 99.2% Recall to prioritize planetary safety over accuracy.")

    except Exception as e:
        st.error(f"**Connection Error:** Backend unreachable. Error: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Developed by dannycodes :)")