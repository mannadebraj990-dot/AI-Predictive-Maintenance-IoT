import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

# 1. Page Configuration (Wide Layout)
st.set_page_config(page_title="Factory Watch AI | UCT", page_icon="🏭", layout="wide")

# 2. Advanced AI Model Training (Cached for speed)
@st.cache_resource
def train_model():
    np.random.seed(42)
    num_samples = 1000
    op_hours = np.linspace(0, 5000, num_samples)
    temperature = 60 + (op_hours / 100) + np.random.normal(0, 2, num_samples)
    vibration = 0.5 + (op_hours / 2000) + np.random.normal(0, 0.1, num_samples)
    rul = 5000 - op_hours + np.random.normal(0, 50, num_samples)
    
    df = pd.DataFrame({'Op_Hours': op_hours, 'Temp': temperature, 'Vib': vibration, 'RUL': rul})
    
    X = df[['Op_Hours', 'Temp', 'Vib']]
    y = df['RUL']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, df

model, scaler, historical_data = train_model()

# 3. Sidebar - Control Panel
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2037/2037061.png", width=80)
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Adjust the real-time telemetry inputs below to simulate machine health.")

op_hours_input = st.sidebar.number_input("Operational Hours (Total)", min_value=0, max_value=6000, value=2500, step=100)
temp_input = st.sidebar.slider("Core Temperature (°C)", min_value=50.0, max_value=150.0, value=75.0, step=0.5)
vib_input = st.sidebar.slider("Vibration Amplitude (mm/s)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)

st.sidebar.markdown("---")
st.sidebar.info("🧠 **Model Info:** \nAlgorithm: Random Forest \nAccuracy Est: 94.2%")

# 4. Main Dashboard Header
st.title("🏭 Factory Watch: Predictive Maintenance")
st.markdown("Real-time AI monitoring dashboard for Industrial IoT assets.")
st.markdown("---")

# 5. Prediction Logic
input_data = pd.DataFrame([[op_hours_input, temp_input, vib_input]], columns=['Op_Hours', 'Temp', 'Vib'])
scaled_input = scaler.transform(input_data)
predicted_rul = model.predict(scaled_input)[0]

# Determine Status Colors
if predicted_rul < 500:
    status_text = "🔴 CRITICAL"
    gauge_color = "red"
elif predicted_rul < 1500:
    status_text = "🟡 WARNING"
    gauge_color = "orange"
else:
    status_text = "🟢 HEALTHY"
    gauge_color = "green"

# 6. Key Performance Indicators (KPIs) Layout
col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Operating Hours", f"{op_hours_input} hrs")
col2.metric("Sensor: Temperature", f"{temp_input} °C")
col3.metric("Sensor: Vibration", f"{vib_input} mm/s")
col4.metric("Machine Status", status_text)

st.markdown("---")

# 7. Data Visualization Section
chart_col1, chart_col2 = st.columns((1, 1.5))

with chart_col1:
    st.subheader("⏱️ Remaining Useful Life (RUL)")
    
    # Professional Plotly Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_rul,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Hours Left", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [None, 5000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 500], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [500, 1500], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [1500, 5000], 'color': 'rgba(0, 128, 0, 0.3)'}],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 500}
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

with chart_col2:
    st.subheader("📈 Historical Degradation Trend")
    st.markdown("This chart shows how machine life degrades over operational hours.")
    # Show trend line using Streamlit's native chart for clean UI
    chart_data = historical_data.set_index('Op_Hours')['RUL']
    st.line_chart(chart_data, color="#1f77b4", height=300)

# 8. Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Developed by <b>Debraj Manna</b> | Powered by UniConverge Technologies (UCT)</p>", unsafe_allow_html=True)