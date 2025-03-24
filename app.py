#import all nec. libraries for data scripting
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(layout="wide")

# Load data
# @st.cache_data
# def load_data():
#     return pd.read_csv("cleaned_promotional_data.csv")

# df = load_data()
@st.cache_data
def load_data():
    # Replace YOUR_FILE_ID with the actual ID from Google Drive
    file_id = "1znwRKZIg0gcIVUuWg3OdLmNpfdn1nVE_"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    return pd.read_csv(url)

df = load_data()

uploaded_file = st.file_uploader("Upload the promotional dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(" Awesome! File uploaded successfully!")
else:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()


# --- Color Palette Selector ---


custom_themes = {
    "Tropical Breeze": ["#00C9A7", "#FFD93D", "#FF6B6B", "#6A0572", "#245953"],
    "Ocean Deep": ["#023E8A", "#0077B6", "#0096C7", "#00B4D8", "#48CAE4","#FFB200"
],
    "Sunset Heat": ["#ff7e5f", "#feb47b", "#ff6b81", "#c44536", "#2b2d42","#EB5B00"],
    "Minimal Light": ["#F4F1DE", "#E07A5F", "#3D405B", "#81B29A", "#F2CC8F","#D91656"],
    "Earth Tones": ["#9C6644", "#DDB892", "#FFE6A7", "#6B4226", "#FFE156","#640D5F"]
}

selected_palette = st.sidebar.selectbox("Select Color Theme", list(custom_themes.keys()))
custom_color = custom_themes[selected_palette]

# --- Sidebar Filters ---
if 'brand' in df.columns:
    brand_options = df['brand'].dropna().unique().tolist()
    selected_brand = st.sidebar.selectbox("Select a Brand", ["All"] + brand_options)
    if selected_brand != "All":
        df = df[df['brand'] == selected_brand]

if 'sub_category' in df.columns:
    subcat_options = df['sub_category'].dropna().unique().tolist()
    selected_subcat = st.sidebar.selectbox("Select a Subcategory", ["All"] + subcat_options)
    if selected_subcat != "All":
        df = df[df['sub_category'] == selected_subcat]


summary_tab,overview_tab, promo_tab, modeling_tab, advanced_tab,download_tab = st.tabs([
    "Summary","Overview", "Promotions", "Modeling", "Advanced Metrics", "Download"])

# --- Summary Tab ---
with summary_tab:
    st.markdown("""
## Executive Summary: Brand Pricing & Promotions

This dashboard presents an in-depth analysis of how brands position themselves through pricing strategies, promotional efforts, and market performance. Our goal is to uncover actionable insights that help stakeholders better understand customer behavior, competitive dynamics, and promotional effectiveness.

### 1. Brand Competitive Reactions
When one brand launches a promotion, others often react quickly. Brands B and E were the most aggressive with steep discounts, while A and C maintained premium price points.
**Insight:** Competitive monitoring is key to staying relevant in fast-moving markets.

### 2. Market Leaders
Based on total unit sales and revenue, Brands A and C lead the pack. D and E are emerging but haven’t broken through.
**Insight:** Leadership depends on strategic pricing, reach, and retention.

### 3. Promo Effectiveness
Not all promotions yield equal results. Brand B saw the highest sales lift per discount dollar, while Brand C struggled despite heavy promo spend.
**Insight:** Promotions need timing and targeting, not just budget.

### 4. Customer Loyalty
Brand E showed consistent sales—indicating a loyal base. Brand A had more volatile sales, suggesting deal-driven behavior.
**Insight:** Consistency = loyalty. Volatility = promo dependence.

### 5. Price Sensitivity
Price Elasticity of Demand (PED) revealed that Brands D and B lose customers quickly when prices rise, while A and C are more resilient.
**Insight:** High elasticity means pricing needs to be carefully managed.

### 6. Seasonal Patterns
Promotions peak in March–April, while prices rise from January–May.
**Insight:** Promotional windows matter. March–April is the battleground.

### 7. Cross-Brand Promotion Impact
When Brand A discounts heavily, Brand B sees reduced sales.
**Insight:** Brand actions affect the entire market, not just themselves.

### 8. Store-Level Reach
Brands A and C had the widest store presence for promotions, while B and D focused selectively.
**Insight:** Exposure must be weighed against cost and return.

### 9. Premium vs Budget Positioning
Premium: A, B, C, E | Budget: D
**Insight:** Premium brands command loyalty; budget brands drive volume.

---

## Strategic Recommendations
- Balance pricing power with retention tactics
- Avoid promo fatigue by rewarding loyal customers
- Focus March–April for major promo pushes
- Monitor competitor pricing and react strategically
- Reinforce brand identity: premium ≠ discount

---

## Glossary of Abbreviations
- **avg_unit_price** – Average Unit Price
- **base_unit_price** – Original (non-discounted) Unit Price
- **any_promo_unit_price** – Unit Price during any kind of Promotion
- **any_promo_unit_price_%_disc** – Percentage Discount applied to Promo Price
- **tdp** – Total Display Price (estimated label price seen by consumers)
- **PED** – Price Elasticity of Demand
- **price_bin** – Grouped price levels created from average price (quantile binning)

---

**Note:** This summary provides context to interpret dashboard visuals and guide promotional strategy decisions.
""")

# --- Overview Tab ---
with overview_tab:
    st.markdown("<h3 style='font-size:24px;'>Brand & Store Overview</h3>", unsafe_allow_html=True)

    if 'brand' in df.columns and 'units' in df.columns:
        brand_agg = df.groupby("brand")["units"].sum().reset_index().sort_values(by="units", ascending=False)
        fig = px.bar(brand_agg, x="brand", y="units", title="Brand Aggregation and Market Share", color_discrete_sequence=custom_color)
        st.plotly_chart(fig, use_container_width=True)

    if 'week' in df.columns and 'any_promo_unit_price' in df.columns:
        seasonality = df.groupby("week")["any_promo_unit_price"].mean().reset_index()
        fig_seasonal = px.line(seasonality, x="week", y="any_promo_unit_price", title="Seasonal Trends in Pricing Promotions", color_discrete_sequence=custom_color)
        st.plotly_chart(fig_seasonal, use_container_width=True)

# --- Promotions Tab ---
with promo_tab:
    st.markdown("<h3 style='font-size:24px;'>Promotions Insights</h3>", unsafe_allow_html=True)

    if 'brand' in df.columns and 'any_promo_unit_price_%_disc' in df.columns:
        promo_depth = df.groupby("brand")["any_promo_unit_price_%_disc"].mean().reset_index()
        fig_promo = px.bar(promo_depth, x="brand", y="any_promo_unit_price_%_disc", title="Average % Discount by Brand", color_discrete_sequence=custom_color)
        st.plotly_chart(fig_promo, use_container_width=True)

    if 'sub_category' in df.columns and 'units' in df.columns:
        low_flavors = df.groupby("sub_category")["units"].sum().reset_index().sort_values(by="units").head(10)
        fig_low = px.bar(low_flavors, x="sub_category", y="units", title="Lowest Performing Flavors", color_discrete_sequence=custom_color)
        st.plotly_chart(fig_low, use_container_width=True)

    if 'brand' in df.columns and 'any_promo_units' in df.columns and 'units' in df.columns:
        promo_effect = df.groupby("brand").agg({"units": "sum", "any_promo_units": "sum"}).reset_index()
        promo_effect["Effectiveness"] = promo_effect["any_promo_units"] / promo_effect["units"]
        fig_eff = px.bar(promo_effect, x="brand", y="Effectiveness", title="Promotional Effectiveness by Brand", color_discrete_sequence=custom_color)
        st.plotly_chart(fig_eff, use_container_width=True)

    if 'week' in df.columns and 'brand' in df.columns and 'any_promo_units' in df.columns:
        cross_impact = df.groupby(['week', 'brand'])["any_promo_units"].sum().reset_index()
        fig_cross = px.line(cross_impact, x="week", y="any_promo_units", color="brand", title="Cross-Brand Promotion Impact Over Time", color_discrete_sequence=custom_color)
        st.plotly_chart(fig_cross, use_container_width=True)

# --- Modeling Tab ---
with modeling_tab:
    st.subheader("Revenue Prediction Models")
    numeric_cols = ["avg_unit_price", "base_unit_price", "any_promo_unit_price", "any_promo_unit_price_%_disc", "tdp"]
    required_cols = numeric_cols + ["$"]
    model_choice = st.selectbox("Select a model", ["Linear Regression", "Ridge Regression", "Random Forest", "XGBoost"])

    if all(col in df.columns for col in required_cols):
        model_df = df[required_cols].dropna()
        X = model_df[numeric_cols]
        y = model_df["$"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Ridge Regression":
            model = Ridge(alpha=1.0)
        elif model_choice == "XGBoost":
            model = XGBRegressor(objective='reg:squarederror', random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            X_train_scaled = X_train
            X_test_scaled = X_test

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.markdown(f"**Model:** {model_choice}")
        st.markdown(f"**R² Score:** {r2:.4f}")
        st.markdown(f"**RMSE:** {rmse:.2f}")

        fig6 = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Revenue', 'y': 'Predicted Revenue'}, title=f"Actual vs Predicted Revenue - {model_choice}", color_discrete_sequence=custom_color)
        st.plotly_chart(fig6, use_container_width=True)
# --- Seasonal Trends Tab ---
# --- Seasonal Trends Tab ---
# with seasonal_tab:
#     st.subheader("Seasonal Trends in Pricing & Promotions")

#     if 'date' in df.columns:
#         df['date'] = pd.to_datetime(df['date'], errors='coerce')
#         df = df.dropna(subset=['date'])  # Drop rows with invalid/missing dates
#         df['month_num'] = df['date'].dt.month
#         df['month'] = df['month_num'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))

#         seasonal_avg = df.groupby(['month_num', 'month'])["avg_unit_price"].mean().reset_index().sort_values('month_num')
#         seasonal_promo = df.groupby(['month_num', 'month'])["any_promo_unit_price"].mean().reset_index().sort_values('month_num')
#         seasonal_discount = df.groupby(['month_num', 'month'])["any_promo_unit_price_%_disc"].mean().reset_index().sort_values('month_num')

#         fig_price = px.line(seasonal_avg, x="month", y="avg_unit_price", title="Average Unit Price by Month", markers=True, color_discrete_sequence=custom_color)
#         fig_promo = px.line(seasonal_promo, x="month", y="any_promo_unit_price", title="Promo Price by Month", markers=True, color_discrete_sequence=custom_color)
#         fig_disc = px.line(seasonal_discount, x="month", y="any_promo_unit_price_%_disc", title="% Discount by Month", markers=True, color_discrete_sequence=custom_color)

#         st.plotly_chart(fig_price, use_container_width=True)
#         st.plotly_chart(fig_promo, use_container_width=True)
#         st.plotly_chart(fig_disc, use_container_width=True)
#     else:
#         st.warning("The 'date' column is missing or not in a valid format.")

# --- Advanced Tab ---
with advanced_tab:
    st.subheader("Product Segmentation & Price Sensitivity")

    if all(col in df.columns for col in ["avg_unit_price", "any_promo_unit_price", "tdp"]):
        clustering_data = df[["avg_unit_price", "any_promo_unit_price", "tdp"]].dropna()
        kmeans = KMeans(n_clusters=3, random_state=0).fit(clustering_data)
        clustering_data["Segment"] = kmeans.labels_
        fig7 = px.scatter(clustering_data, x="avg_unit_price", y="tdp", color="Segment", title="Product Segmentation", color_discrete_sequence=custom_color)
        st.plotly_chart(fig7, use_container_width=True)

    if all(col in df.columns for col in ["avg_unit_price", "units"]):
        sensitivity = df[["avg_unit_price", "units"]].dropna().copy()
        sensitivity["price_bin"] = pd.qcut(sensitivity["avg_unit_price"], q=5).astype(str)
        grouped = sensitivity.groupby("price_bin")["units"].mean().reset_index()
        fig8 = px.line(grouped, x="price_bin", y="units", title="Price Sensitivity Analysis", markers=True, color_discrete_sequence=custom_color)
        st.plotly_chart(fig8, use_container_width=True)

    if 'date' in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.strftime('%B')
        seasonal_avg = df.groupby("month")["avg_unit_price"].mean().reset_index()
        seasonal_promo = df.groupby("month")["any_promo_unit_price"].mean().reset_index()
        seasonal_discount = df.groupby("month")["any_promo_unit_price_%_disc"].mean().reset_index()

        fig_price = px.line(seasonal_avg, x="month", y="avg_unit_price", title="Average Unit Price by Month", markers=True, color_discrete_sequence=custom_color)
        fig_promo = px.line(seasonal_promo, x="month", y="any_promo_unit_price", title="Promo Price by Month", markers=True, color_discrete_sequence=custom_color)
        fig_disc = px.line(seasonal_discount, x="month", y="any_promo_unit_price_%_disc", title="% Discount by Month", markers=True, color_discrete_sequence=custom_color)

        st.plotly_chart(fig_price, use_container_width=True)
        st.plotly_chart(fig_promo, use_container_width=True)
        st.plotly_chart(fig_disc, use_container_width=True)


# --- Download Tab ---
with download_tab:
    st.subheader("Download Filtered Dataset")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")
    st.dataframe(df.head(100))

# Global style update
st.markdown("""
    <style>
        .stPlotlyChart svg text {
            font-size: 16px !important;
        }
        .stMarkdown h3 {
            font-size: 24px !important;
        }
        .css-1d391kg { font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
---
<div style='text-align:center; padding-top: 20px; font-size: 18px;'>
    Created and Published by <strong>Capstone Team C</strong>
</div>
""", unsafe_allow_html=True)
