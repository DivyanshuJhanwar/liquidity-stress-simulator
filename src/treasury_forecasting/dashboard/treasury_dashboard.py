import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
FLAGGED_DATA_PATH = "reports/flagged_risky_banks.csv"
TOP_20_PLOT = "reports/top_risk_banks.png"
SHOCK_PLOT = "reports/borrowings_shock_impact.png"
CHARTER_PLOT = "reports/charter_liquidity_boxplot.png"
ASSET_TIER_PLOT = "reports/asset_tier_liquidity_boxplot.png"

# Set page config
st.set_page_config(page_title="Treasury Liquidity Dashboard", layout="wide")

# Sidebar
st.sidebar.header("Simulation Controls")
threshold = st.sidebar.slider("Liquidity Threshold (%)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)

show_shock = st.sidebar.checkbox("Show Shock Impact Histogram", value=True)
show_charter = st.sidebar.checkbox("Show Charter Boxplot", value=True)
show_asset_tier = st.sidebar.checkbox("Show Asset Tier Boxplot", value=True)
show_risky_table = st.sidebar.checkbox("Show Top Risky Banks Table", value=True)

# Title and context
st.title("Treasury Liquidity Stress Simulator")
st.markdown("""
### 💡 About this Project

This portfolio project explores how borrowing shocks affect post-crisis bank liquidity using a real-world dataset of over 18,000 US institutions. It simulates stress scenarios using a Random Forest model, flags vulnerable banks below threshold levels, and segments resilience patterns by charter type and asset tier.

Built as part of my career development journey, this tool integrates regulatory concepts with financial analytics and showcases my Python proficiency in end-to-end dashboard development.

---

### 📊 Model Performance

- **R² Score**: `0.8863`  
- **Mean Absolute Error (MAE)**: `2.03`

These results indicate strong predictive accuracy in post-shock liquidity forecasting, confirming the reliability of the simulator across diverse institutions.
""")


# Summary Insight
st.subheader("Key Insight")
st.info(
    "Post-shock liquidity resilience appears concentrated in high-asset institutions, "
    "with over 90% of flagged banks belonging to the ‘Over $1B’ tier. Distribution analysis reveals "
    "median liquidity ratios hovering near 0.10, while select outliers maintain excess cash buffers exceeding 0.7—"
    "highlighting stratified stress absorption capabilities."
)

# Load flagged data
try:
    df = pd.read_csv(FLAGGED_DATA_PATH)
    risky_df = df[df["liquidity_post_shock"] < threshold]
except Exception as e:
    st.error(f"Failed to load flagged data: {e}")
    st.stop()

# Show histogram
if show_shock:
    st.subheader("Liquidity Change Distribution")
    st.image(SHOCK_PLOT, use_column_width=True)

# Show charter segmentation
col1, col2 = st.columns(2)

with col1:
    if show_charter:
        st.subheader("Charter Type Segmentation")
        st.image(CHARTER_PLOT, use_column_width=True)

with col2:
    if show_asset_tier:
        st.subheader("Asset Tier Segmentation")
        st.image(ASSET_TIER_PLOT, use_column_width=True)

# Show table of risky banks
if show_risky_table:
    st.subheader(f"Top Risky Banks Below {threshold}% Liquidity")
    preview = risky_df.sort_values("liquidity_post_shock").head(20)[
        ["cert", "total_assets", "borrowings", "liquidity_post_shock"]
    ]
    st.dataframe(preview, use_container_width=True)

# Download button
st.subheader("Download Flagged Banks")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Full CSV",
    data=csv,
    file_name="flagged_risky_banks.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.caption("Developed by Divyanshu • Liquidity Stress Simulator • Streamlit Portfolio Edition")
