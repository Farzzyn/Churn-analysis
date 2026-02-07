# ⚓ Anchor - Customer Churn Analytics

<div align="center">

![Anchor Logo](assets/logo.png)

**Implies stability and keeps customers grounded in your service.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-analysis-anchor.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🚀 **Live Demo**](https://churn-analysis-anchor.streamlit.app/) • [📖 Documentation](#features) • [🐛 Report Bug](https://github.com/Farzzyn/Churn-analysis/issues)

</div>

---

## 🎯 Overview

**Anchor** is a powerful, AI-driven customer churn prediction and analytics platform. Built with a sleek Spotify-inspired dark theme, it transforms raw customer data into actionable retention strategies.

Upload your customer data and let Anchor:
- 🧹 **Clean & engineer** your data automatically
- 🤖 **Predict churn probability** using Machine Learning
- 📊 **Visualize insights** with interactive dashboards
- 💡 **Generate AI strategies** for customer retention

---

## ✨ Features

### 📥 Smart Data Ingestion
- Supports **CSV** and **Excel** (.xlsx) file uploads
- Automatic header standardization (snake_case)
- Intelligent duplicate detection and removal
- Auto-detection of date columns

### 🛡️ Data Guard (Cleaning & Validation)
- **Numeric Imputation**: Fills missing values with median
- **Categorical Handling**: Replaces nulls with 'Unknown'
- **Type Detection**: Automatically converts date strings to datetime

### ⚙️ Feature Engineering Engine
- **Tenure Calculation**: Computes customer tenure from signup/activity dates
- **LTV Estimation**: Calculates Estimated Lifetime Value
- **Smart Aggregations**: Creates meaningful derived metrics

### 🧠 Machine Learning Pipeline
- **Random Forest Classifier** with balanced class weights
- Automatic one-hot encoding for categorical variables
- Train/test split with 70/30 ratio
- Feature importance analysis

### 📊 Interactive Visual Intelligence
| Visualization | Description |
|---------------|-------------|
| **Confusion Matrix** | Model performance at a glance |
| **Risk Distribution** | Churn probability histogram |
| **Feature Importance** | Top 10 churn drivers |
| **Retention Heatmap** | Cohort analysis by tenure & segments |
| **Correlation Matrix** | Variable relationships |

### 🎯 AI Strategic Insights
- **Automated analysis** of top churn drivers
- **Personalized recommendations** based on your data
- **Executive summary** with actionable next steps

### 📤 Export Intelligence
- Download scored data with churn probabilities as CSV
- Ready for CRM integration or further analysis

---

## 🚀 Live Demo

Experience Anchor in action:

### 👉 [**Launch App →**](https://churn-analysis-anchor.streamlit.app/)

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Farzzyn/Churn-analysis.git
   cd Churn-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run churn_app.py
   ```

5. **Open your browser** at `http://localhost:8501`

---

## 📦 Dependencies

```txt
streamlit
pandas
numpy
plotly
scikit-learn
openpyxl
```

---

## 📋 Usage Guide

### Step 1: Upload Your Data
- Click the file uploader in the sidebar
- Select a CSV or Excel file containing customer data

### Step 2: Configure Target Column
- Anchor auto-detects columns containing "churn" or "target"
- Manually select if needed

### Step 3: Run Analysis
- Click the **"Run Analysis"** button
- Wait for the ML model to train

### Step 4: Explore Insights
- View the Executive Dashboard with key metrics
- Explore interactive visualizations
- Read AI-generated retention strategies

### Step 5: Export Results
- Download the scored data with churn probabilities
- Use for targeted marketing campaigns

---

## 📊 Sample Data Format

Your input data should include customer-level records. Anchor works best with columns like:

| Column Type | Examples |
|-------------|----------|
| **Customer ID** | `customer_id`, `user_id` |
| **Dates** | `signup_date`, `last_activity` |
| **Monetary** | `monthly_charges`, `total_spend` |
| **Categorical** | `contract_type`, `payment_method` |
| **Target** | `churn`, `churned`, `is_active` |

---

## 🎨 Design Philosophy

Anchor features a **Spotify-inspired dark theme** with:
- 🌑 Rich dark backgrounds (`#121212`, `#181818`)
- 💚 Vibrant green accents (`#1DB954`)
- 🔘 Pill-shaped interactive buttons
- ✨ Smooth hover transitions
- 📱 Responsive layout

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Farzzyn**

- GitHub: [@Farzzyn](https://github.com/Farzzyn)

---

<div align="center">

**⚓ Keep your customers grounded with Anchor**

Made with ❤️ and Python

</div>
