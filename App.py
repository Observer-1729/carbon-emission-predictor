import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb

st.set_page_config(page_title="Carbon Emission Forecasting", layout="wide")
st.title("🌍 Carbon Emission Forecasting")

uploaded_file = st.file_uploader("📁 Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    try:
        # Load dataset
        
        df = pd.read_csv(uploaded_file)

        # Keep numeric columns only
        df = df.select_dtypes(include=[np.number])
        if df.shape[1] < 2:
            st.error("Dataset must have at least one feature column and one target column.")
            st.stop()

        # Features & target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Add Hour column
        X['Hour'] = np.tile(np.arange(24), len(X) // 24 + 1)[:len(X)]

        # Lag & rolling features
        X['y_lag1'] = y.shift(1)
        X['y_lag2'] = y.shift(2)
        X['y_rolling_mean3'] = y.rolling(window=3).mean()

        X = X.dropna()
        y = y.iloc[-len(X):]

        # Train-test split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train model
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate
        X_train_eval = scaler.transform(X_train)
        y_pred_train = best_model.predict(X_train_eval)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        r2 = r2_score(y_train, y_pred_train)

        st.success("✅ Model trained successfully!")
        st.write(f"**📉 RMSE (Training):** {rmse:.2f}")
        st.write(f"**🎯 R² Score (Training Accuracy):** {r2:.2f}")

        # User input
        st.subheader("🛠️ Input Operating Conditions")
        with st.form("manual_input_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                fuel_usage = st.number_input("Fuel Usage", value=float(X.iloc[-1]['fuel_usage']))
                air_flow = st.number_input("Air Flow", value=float(X.iloc[-1]['air_flow']))
            with col2:
                kiln_temp = st.number_input("Kiln Temperature", value=float(X.iloc[-1]['kiln_temperature']))
                raw_feed = st.number_input("Raw Feed Rate", value=float(X.iloc[-1]['raw_feed_rate']))
            with col3:
                prod_rate = st.number_input("Production Rate", value=float(X.iloc[-1]['production_rate']))
            submitted = st.form_submit_button("Predict 24 Hour Forecast")

        if submitted:
            st.info("Generating 24-hour forecast...")

            # Forecast logic
            last_y_values = y.values[-3:].tolist()
            base_features = np.array([fuel_usage, kiln_temp, air_flow, raw_feed, prod_rate])

            predictions = []
            hours = list(range(24))

            for hour in hours:
                y_lag1 = last_y_values[-1]
                y_lag2 = last_y_values[-2]
                y_roll = np.mean(last_y_values[-3:])
                full_features = np.append(base_features, [hour, y_lag1, y_lag2, y_roll])
                scaled = scaler.transform([full_features])
                pred = best_model.predict(scaled)[0]
                predictions.append(pred)
                last_y_values.append(pred)

            # Table
            forecast_df = pd.DataFrame({
                "Hour": hours,
                "Predicted CO₂ Emission (tons/hr)": predictions
            })
            total_emission = sum(predictions)

            st.subheader("📋 Forecasted CO₂ Emission (Hourly)")
            st.dataframe(forecast_df.style.format({"Predicted CO₂ Emission (tons/hr)": "{:.2f}"}))

            # Download
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forecast (CSV)",
                data=csv,
                file_name="24_hour_forecast.csv",
                mime="text/csv"
            )

            # Plot
            st.subheader("📈 CO₂ Emission Forecast")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(forecast_df["Hour"], forecast_df["Predicted CO₂ Emission (tons/hr)"], marker='o')
            ax.set_xticks(hours)
            ax.set_title("Hourly CO₂ Emission Forecast")
            ax.set_xlabel("Hour")
            ax.set_ylabel("CO₂ Emissions (tons/hr)")
            ax.grid(True)
            st.pyplot(fig)

            # Total
            st.success(f"🌍 Total Predicted CO₂ Emission (24h): `{total_emission:.2f} tons`")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
