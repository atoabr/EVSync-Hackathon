import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="EVSync Platform", layout="wide")

# --- DATA PREPARATION ---
# Mock Data for Market Comparison
data = {
    'Year': [2026, 2028, 2030, 2032, 2035],
    'Canada_Adoption_Pct': [20, 34, 60, 83, 100],
    'Pakistan_Adoption_Pct': [5, 15, 30, 40, 50]
}
df = pd.DataFrame(data)

# Historical data for the AI prediction model
historical_years = [2020, 2021, 2022, 2023, 2024, 2025]
pakistan_historical = [1, 1.5, 2.1, 3.5, 4.2, 5.0]

# Machine Learning Function
def forecast_adoption(years, adoption_rates, future_years):
    X = np.array(years).reshape(-1, 1)
    y = np.array(adoption_rates)
    X_future = np.array(future_years).reshape(-1, 1)

    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    X_future_poly = poly.transform(X_future)

    model = LinearRegression()
    model.fit(X_poly, y)

    predictions = model.predict(X_future_poly)
    predictions = np.clip(predictions, 0, 100) # Prevents predicting over 100%
    return predictions

# --- UI LAYOUT ---
st.sidebar.title("⚡ EVSync Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "Market Comparison", "Predictive Insights", "Supply Chain"])

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Abrar Hussain | BQF Digitals")

if page == "Overview":
    st.title("EVSync: Global EV Intelligence")
    st.markdown("Bridging the data gap in the global EV supply chain.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Global EV Market Value", value="$500B+", delta="15% YoY")
    col2.metric(label="Canada 2035 Target", value="100% ZEV", delta="On Track")
    col3.metric(label="Pakistan 2030 Target", value="30% EV", delta="Emerging")

elif page == "Market Comparison":
    st.title("Canada vs. Pakistan: 2026-2035 Outlook")
    fig = px.line(df, x='Year', y=['Canada_Adoption_Pct', 'Pakistan_Adoption_Pct'], 
                  labels={'value': 'Adoption Target (%)', 'variable': 'Country'},
                  title="EV Adoption Targets (2026-2035)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("**Canada:** Mandating 100% Zero-Emission Vehicles by 2035.")
    st.write("**Pakistan:** Targeting 30% EV adoption by 2030, driven by 2-wheelers and 3-wheelers.")

elif page == "Predictive Insights":
    st.title("AI Forecast: EV Adoption Trajectory")
    st.write("Using Scikit-Learn Polynomial Regression to forecast market growth.")
    
    st.sidebar.subheader("Forecast Parameters")
    target_year = st.sidebar.slider("Forecast until year:", 2026, 2040, 2035)

    # Run the model
    future_years = list(range(2026, target_year + 1))
    predicted_rates = forecast_adoption(historical_years, pakistan_historical, future_years)

    # Combine data
    all_years = historical_years + future_years
    all_rates = pakistan_historical + list(predicted_rates)
    
    df_forecast = pd.DataFrame({
        "Year": all_years,
        "Adoption %": all_rates,
        "Type": ["Historical"] * len(historical_years) + ["Forecast"] * len(future_years)
    })

    # Plot
    fig = px.scatter(df_forecast, x="Year", y="Adoption %", color="Type", 
                     title="Pakistan EV Adoption Forecast (Scikit-Learn)")
    st.plotly_chart(fig, use_container_width=True)

    # Insight generation
    parity_year = df_forecast[df_forecast['Adoption %'] >= 30]['Year'].min()
    if pd.notna(parity_year):
        st.success(f"🤖 Model Prediction: Pakistan is on track to hit its 30% mandate by **{int(parity_year)}**.")

elif page == "Supply Chain":
    st.title("Global Supply Chain Flow")
    st.write("Tracking critical EV materials from raw mining to final vehicle assembly.")
    
    # Create an interactive Sankey Diagram
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = ["Lithium (Chile/Australia)", "Nickel (Canada/Indonesia)", "Cobalt (DRC)", 
                   "Chemical Refining (Asia/EU)", "Battery Cell Production (Gigafactories)", 
                   "Canada Assembly (4-Wheelers)", "Pakistan Assembly (2/3-Wheelers)"],
          color = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7B731", "#5F27CD", "#FF9F43", "#10AC84"]
        ),
        link = dict(
          source = [0, 1, 2, 3, 4, 4], # The origin nodes
          target = [3, 3, 3, 4, 5, 6], # The destination nodes
          value =  [40, 35, 25, 100, 70, 30] # The volume of flow
        ))])
    
    fig.update_layout(height=600, font_size=12)
    st.plotly_chart(fig, use_container_width=True)
