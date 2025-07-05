# 🏭 Carbon Emission Forecasting in Sinter Plants and Kilns

This project is a machine learning-powered CO₂ emission forecasting system designed to help industries monitor, predict, and plan their emission outputs using historical and operational data. The app is built using **Python**, **Streamlit**, and **XGBoost**, with support for CSV and Excel datasets.

---

## 🚀 Features

- 📤 Upload `.csv` dataset with emission data
- 🧠 Uses lag features, rolling averages, and real-time input scaling
- 🔍 Predicts CO₂ emissions based on historical trends
- 📈 Visualizes actual vs. predicted emissions over time
- 📊 Shows RMSE and R² Score for accuracy evaluation
- 📥 Download predictions as `.csv`
- ✅ Clean and interactive **Streamlit** interface

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/carbon-emission-predictor.git
cd carbon-emission-predictor
2️⃣ (Optional) Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
Activate on Windows:

bash
Copy
Edit
venv\Scripts\activate
Activate on Mac/Linux:

bash
Copy
Edit
source venv/bin/activate
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the Streamlit App
bash
Copy
Edit
streamlit run App.py
The app will launch in your browser at http://localhost:8501.

⚙️ How It Works
User Uploads Dataset: Supports CSV and XLSX files.

Preprocessing:

Converts Excel to CSV if needed

Creates lag features and rolling averages

Scales input using StandardScaler

Model Training:

XGBoost Regressor with GridSearchCV optimization

Automatically detects the last column as the target

Prediction & Evaluation:

Predicts CO₂ emissions

Calculates RMSE and R²

Shows plots and lets users download results

📁 Input Format
The input file should:

Contain timestamped or sequential data

Have numerical columns with the last column as the target (CO₂ emissions)

📊 Example Metrics
Metric	Value (Synthetic Data)
RMSE	~2.21
R² Score	~0.93

🧪 Tech Stack
Frontend: Streamlit

Backend/ML: Python, XGBoost, scikit-learn, pandas, numpy

Visualization: Matplotlib

Input/Output: CSV files

📡 Deployment
✅ Run Locally or on Streamlit Cloud
Push the repo to GitHub

Go to Streamlit Cloud

Click New App, select the repo and App.py

Click Deploy 🚀

🎯 Use Cases
🌱 Environmental Engineers

🏭 Industrial Emissions Monitoring

🔍 Regulatory Compliance & Audits

📊 Academic Projects on Carbon Forecasting

💼 Sustainable Manufacturing Initiatives

📦 Requirements
Install with:

bash
Copy
Edit
pip install -r requirements.txt
Main Libraries:

streamlit

pandas

numpy

scikit-learn

xgboost

matplotlib


📈 Future Improvements
✅ Real industrial dataset support
✅ Live dashboard with time-series updates
✅ Anomaly detection for emission spikes
✅ Integration with sensor/IoT data
✅ Export results to PDF reports

🔚 Conclusion
This forecasting app gives industries a practical, AI-driven way to monitor and predict CO₂ emissions using familiar data inputs. With real-time model training, feature engineering, and powerful visualizations, it helps companies take the first step toward data-driven emission reduction strategies. 🌍♻️
