import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import xgboost as xgb

st.title("Carbon Emission Forecasting")

uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])

if uploaded_file:
    try:
        # Load CSV or Excel
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        st.subheader("Raw Dataset Preview")
        st.write(df.head())

        # Drop datetime columns if any
        df = df.select_dtypes(exclude=['datetime', 'object'])
        if df.shape[1] < 2:
            st.error("Your dataset must contain at least one feature column and one target column (numeric).")
            st.stop()

        # Split features and target (last column is target)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Add engineered features (if applicable)
        y_shifted_1 = y.shift(1)
        y_shifted_2 = y.shift(2)
        X['y_lag1'] = y_shifted_1
        X['y_lag2'] = y_shifted_2
        X['y_rolling_mean3'] = y.rolling(window=3).mean()
        X = X.dropna()
        y = y.iloc[-len(X):]
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train XGBoost with Grid Search
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }

        model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=0)
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Model trained successfully!")
        st.write(f"**Root Mean Square Error (RMSE):** {rmse:.2f}")
        st.write(f"**R² Score (Accuracy):** {r2:.2f}")

        # Show predicted values
        st.subheader("🔍 Prediction Results (Top 10)")
        results_df = pd.DataFrame({
            "Actual CO₂ Emission": y_test.values,
            "Predicted CO₂ Emission": y_pred
        }).reset_index(drop=True)

        st.dataframe(results_df.head(10).style.format("{:.2f}"))

        # Highlight most recent prediction
        latest_actual = results_df.iloc[-1]['Actual CO₂ Emission']
        latest_pred = results_df.iloc[-1]['Predicted CO₂ Emission']

        st.markdown(f"### 🔔 Most Recent Prediction")
        st.write(f"**Predicted CO₂ Emission:** `{latest_pred:.2f} tons/hour`")
        st.write(f"**Actual CO₂ Emission (for comparison):** `{latest_actual:.2f} tons/hour`")

        # Optional download
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Prediction Data", csv, "predictions.csv", "text/csv")


        # Plot
        st.subheader("📈 Actual vs Predicted CO₂ Emissions")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(y_test.values, label='Actual')
        ax.plot(y_pred, label='Predicted')
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("CO₂ Emissions (Ton/hr)")
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred: {e}")
