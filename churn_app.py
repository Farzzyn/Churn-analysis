import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import io

# --- MODULE 1: UI LAYOUT & STATE MANAGEMENT ---
def setup_page():
    st.set_page_config(
        page_title="Anchor - Stability Analytics",
        page_icon="⚓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Spotify-inspired Dark Theme
    st.markdown("""
        <style>
        /* Main Background */
        .stApp {
            background-color: #121212;
            color: #FFFFFF;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Circular', 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #FFFFFF;
            font-weight: 700;
        }
        
        /* Cards / Containers (Like Playlists) */
        .css-1r6slb0, .css-12oz5g7, .stExpander, div[data-testid="stMetric"] {
            background-color: #181818; 
            border-radius: 8px;
            padding: 16px; 
            transition: background-color 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            background-color: #282828;
        }
        
        /* Metric Values */
        div[data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            font-weight: 700;
            color: #1DB954 !important; /* Spotify Green */
        }
        div[data-testid="stMetricLabel"] {
            font-size: 1rem;
            color: #B3B3B3; /* Light Gray text */
        }
        
        /* Buttons (Pill shaped, Green) */
        .stButton>button {
            border-radius: 500px;
            background-color: #1DB954;
            color: #000000;
            font-weight: bold;
            border: none;
            padding: 12px 32px;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: transform 0.1s;
        }
        .stButton>button:hover {
            background-color: #1ed760;
            transform: scale(1.04);
            color: #000000;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #000000;
        }
        
        /* Inputs/Selectboxes */
        .stSelectbox > div > div {
            background-color: #282828;
            color: white;
            border: none;
        }
        
        /* Custom Helper Classes */
        .highlight {
            color: #1DB954;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.image("assets/logo.png", width=80)
    with col_title:
        st.title("Anchor")
    
    st.markdown("<h4 style='color: #B3B3B3; margin-top: -20px; font-weight: 400;'>Implies stability and keeps customers grounded in your service.</h4>", unsafe_allow_html=True)

# --- MODULE 2: THE 'DATA GUARD' (CLEANING & VALIDATION) ---
@st.cache_data
def robust_cleaning(df):
    try:
        # Standardize headers to snake_case
        df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        
        # Deduplication using a potential ID column if exists, otherwise strictly duplicates
        possible_id_cols = [col for col in df.columns if 'id' in col or 'customer' in col]
        if possible_id_cols:
            df = df.drop_duplicates(subset=possible_id_cols[0])
        else:
            df = df.drop_duplicates()

        # Auto-Type Detection: Date conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col])
                except (ValueError, TypeError):
                    pass  # Keep as object if not date

        # Messy Data Handling: Imputation
        # Numeric
        num_cols = df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            imputer_num = SimpleImputer(strategy='median')
            df[num_cols] = imputer_num.fit_transform(df[num_cols])
        
        # Categorical
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if not cat_cols.empty:
            df[cat_cols] = df[cat_cols].fillna('Unknown')

        return df
    except Exception as e:
        st.error(f"Error in Data Guard: {e}")
        return df

# --- MODULE 3: FEATURE ENGINEERING ENGINE ---
@st.cache_data
def feature_engineering(df):
    try:
        # Tenure Calculation
        # Look for 'signup_date' and 'last_activity' or similar
        date_cols = df.select_dtypes(include=['datetime']).columns
        signup_col = next((c for c in date_cols if 'signup' in c or 'start' in c or 'join' in c), None)
        activity_col = next((c for c in date_cols if 'activity' in c or 'end' in c or 'last' in c), None)
        
        if signup_col and activity_col:
            df['tenure_days'] = (df[activity_col] - df[signup_col]).dt.days
            # Fill negative or NaN tenure with median
            df['tenure_days'] = df['tenure_days'].fillna(df['tenure_days'].median())
            df.loc[df['tenure_days'] < 0, 'tenure_days'] = 0

        # LTV Calculation (Tenure * Monthly Spend)
        # Look for 'monthly_spend', 'bill', 'charge'
        spend_col = next((c for c in df.select_dtypes(include=[np.number]).columns if 'spend' in c or 'bill' in c or 'charge' in c or 'amount' in c), None)
        
        if spend_col and 'tenure_days' in df.columns:
            # Approx months
            df['estimated_ltv'] = (df['tenure_days'] / 30) * df[spend_col]
        
        # Activity Velocity (Mock implementation logic as specific historical data might need complex structure)
        # If we had a usage history column, we'd calculate slope. 
        # Here we check if there's a numeric usage column and create a mock velocity if appropriate data is missing
        # For this script, we'll skip complex time-series slope calculation within a flat file unless columns suggest it.
        
        return df
    except Exception as e:
        st.warning(f"Feature Engineering Warning: {e}")
        return df

# --- MODULE 4: MACHINE LEARNING & PREDICTIVE ANALYTICS ---
@st.cache_resource
def train_model(df, target_col):
    try:
        # Preprocessing for ML
        # Drop date columns and ID columns for modeling
        X = df.drop(columns=[target_col])
        X = X.select_dtypes(exclude=['datetime'])
        
        # Drop ID-like columns if they have too many unique values (heuristic)
        for col in X.select_dtypes(include=['object']).columns:
            if X[col].nunique() > 0.9 * len(X):
                X = X.drop(columns=[col])

        # One-Hot Encoding for remaining categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        y = df[target_col]
        
        # Handle target if it's not numeric
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            # Assuming 1 is Churn (positive class). If "Yes"/"No", "Yes" usually becomes 1.

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest
        clf = RandomForestClassifier(class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature Importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': clf.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Predict on full dataset for export
        full_prob = clf.predict_proba(X)[:, 1]
        
        return clf, acc, cm, feature_importance, full_prob, X_test, y_test

    except Exception as e:
        st.error(f"Modeling Error: {e}")
        return None, None, None, None, None, None, None

# --- MODULE 6: AI STRATEGY & EXPORT ---
def generate_executive_summary(feature_importance, churn_rate, df, target_col):
    top_features = feature_importance.head(3)['Feature'].tolist()
    
    strategies = []
    
    # helper to get a friendly name
    def clean_name(name):
        return name.replace('_', ' ').title()

    # Introduction
    if churn_rate > 0.2:
        strategies.append(f"🔴 **Critical Attention Needed**: The current churn rate is **{churn_rate:.1%}**, which is above the healthy threshold. Immediate retention efforts are required.")
    else:
        strategies.append(f"🟢 **Healthy Status**: Churn rate is stable at **{churn_rate:.1%}**. Focus on sustaining customer satisfaction.")

    strategies.append("### 🧠 Smart Drivers Analysis")

    # Analyze Top Drivers
    for feature in top_features:
        fname = clean_name(feature)
        
        # Check if feature exists in df (it might be OHE, so we check original if possible, or just skip if not found directly)
        # Note: If OHE (e.g. 'Contract_Two year'), the logic below handles it as binary numeric usually.
        
        if feature in df.columns:
            # Numeric Analysis
            if pd.api.types.is_numeric_dtype(df[feature]):
                # Assume Churn is mapped to 1 in the probability or we look at the actual target col if it's numeric 0/1
                # We need to trust 'target_col' has 0/1 or similar. 
                # If target is object, we can't easily mean(), but we have Churn_Probability.
                
                # Let's use Churn_Probability as a proxy for the 'Churn Class' to split groups
                churn_mean = df[df['Churn_Probability'] > 0.5][feature].mean()
                stay_mean = df[df['Churn_Probability'] <= 0.5][feature].mean()
                
                if pd.isna(churn_mean) or pd.isna(stay_mean):
                    strategies.append(f"- **{fname}**: Significant impact detected, but data distribution is unclear.")
                    continue

                diff = churn_mean - stay_mean
                # Heuristic: If churned users have HIGHER avg, it's a "Pain Point"? Or "High Usage Risk"?
                # If churned users have LOWER avg, it's a "Stickiness Factor" (e.g. Tenure).
                
                if diff > 0:
                    strategies.append(f"- **{fname}**: Customers with **higher {fname}** ({churn_mean:.1f} vs {stay_mean:.1f}) are more likely to leave. *Potential friction point or pricing issue.*")
                else:
                    strategies.append(f"- **{fname}**: Customers with **lower {fname}** ({churn_mean:.1f} vs {stay_mean:.1f}) are at risk. *This appears to be a loyalty driver—encourage engagement here.*")
            else:
                 strategies.append(f"- **{fname}**: This category is a key differentiator in customer decisions.")
        
        # Handle One-Hot Encoded features implicitly (e.g. "Gender_Male")
        # We might not have the original column easily mapping back without complex logic, 
        # so we try to parse the name.
        else:
             strategies.append(f"- **{fname}**: This specific segment or attribute is highly influential. Investigate specific cohorts related to this.")

    strategies.append("### 🚀 Recommended Actions")
    strategies.append(f"1. **Targeted Campaigns**: focus on the '**{clean_name(top_features[0])}**' metric. Create an email or notification campaign addressing this specific behavior.")
    strategies.append(f"2. **Personalization**: Use the insights from **{clean_name(top_features[1])}** to customize the user dashboard or pricing tier.")
    
    return strategies

# --- MAIN APPLICATION LOGIC ---
def main():
    setup_page()
    
    # Sidebar
    st.sidebar.header("Data Configuration")
    uploaded_file = st.sidebar.file_uploader("Upload Customer Data (CSV/XLSX)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.toast("Data successfully loaded!", icon="✅")
            
            # 1. Clean Data
            df_clean = robust_cleaning(df)
            
            # 2. Feature Engineering
            df_eng = feature_engineering(df_clean)
            
            with st.expander("Data Preview (Cleaned & Engineered)", expanded=False):
                st.dataframe(df_eng.head())
            
            # Settings: Target Column
            cols = df_eng.columns.tolist()
            # Try to auto-detect target
            default_target = next((c for c in cols if 'churn' in c.lower() or 'target' in c.lower()), cols[-1])
            target_col = st.sidebar.selectbox("Select Target/Churn Column", cols, index=cols.index(default_target) if default_target in cols else 0)
            
            if st.sidebar.button("Run Analysis", type="primary"):
                with st.spinner("Training models and generating insights..."):
                    model, acc, cm, feats, probs, X_test, y_test = train_model(df_eng, target_col)
                
                if model:
                    # Append probabilities to original df (careful with alignment, here assuming index matches for simplicity or using full fit)
                    # For strict correctness in this single file demo, we calculated full_prob on the whole X
                    df_eng['Churn_Probability'] = probs
                    
                    # KPIs
                    churn_rate = df_eng[target_col].mean() if pd.api.types.is_numeric_dtype(df_eng[target_col]) else (df_eng[target_col].value_counts(normalize=True).iloc[0] if df_eng[target_col].value_counts(normalize=True).index[0] == 1 else df_eng[target_col].value_counts(normalize=True).get(1, 0))
                    # Handle boolean/string target for rate calc
                    if df_eng[target_col].dtype == 'object':
                         # Assuming 'Yes' or similar is Churn. We encoded earlier but that was local to train_model. 
                         # Let's do a quick check
                         is_churn_val = df_eng[target_col].mode()[0] # Fallback
                         # If we can't easily determine 'positive' class without looking at encoder, we rely on the encoded 'probs' 
                         # High probability means 'Positive' class.
                         churn_rate = (df_eng['Churn_Probability'] > 0.5).mean()

                    # Revenue at Risk (Sum of LTV for those predicted to churn)
                    rev_risk = 0
                    if 'estimated_ltv' in df_eng.columns:
                        rev_risk = df_eng[df_eng['Churn_Probability'] > 0.5]['estimated_ltv'].sum()
                    elif 'monthly_charges' in df_eng.columns: # fallback common name
                         rev_risk = df_eng[df_eng['Churn_Probability'] > 0.5]['monthly_charges'].sum() * 12

                    # --- MODULE 5: INTERACTIVE VISUAL INTELLIGENCE ---
                    st.divider()
                    st.header("Executive Dashboard")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted Churn Rate", f"{churn_rate:.1%}")
                    c2.metric("Revenue at Risk", f"${rev_risk:,.2f}")
                    c3.metric("Model Accuracy", f"{acc:.1%}")
                    
                    st.divider()
                    
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        st.subheader("Confusion Matrix")
                        if cm.shape == (2, 2):
                            cm_df = pd.DataFrame(cm, index=['Actual Stay', 'Actual Churn'], columns=['Pred Stay', 'Pred Churn'])
                        else:
                            # Fallback for multi-class targets (e.g. if user selected a column with >2 values)
                            cm_df = pd.DataFrame(cm, index=[f'Actual {i}' for i in range(cm.shape[0])], columns=[f'Pred {i}' for i in range(cm.shape[1])])
                            st.warning(f"⚠️ Multiclass Target Detected: The selected target has {cm.shape[0]} classes. Confusion Matrix adapted accordingly.")
                        
                        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Greens', title="Model Performance", template='plotly_dark')
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                    with col_viz2:
                        st.subheader("Churn Risk Distribution")
                        fig_hist = px.histogram(df_eng, x='Churn_Probability', nbins=20, title="Probability Density of Churn", color_discrete_sequence=['#1DB954'], template='plotly_dark')
                        st.plotly_chart(fig_hist, use_container_width=True)

                    st.subheader("Feature Importance")
                    fig_feat = px.bar(feats.head(10), x='Importance', y='Feature', orientation='h', title="Top Drivers of Churn", color='Importance', color_continuous_scale='Greens', template='plotly_dark')
                    st.plotly_chart(fig_feat, use_container_width=True)

                    # Retention Heatmap (Cohort-like analysis)
                    if 'tenure_days' in df_eng.columns:
                        st.subheader("Retention Heatmap")
                        # Bin tenure
                        df_eng['Tenure_Bin'] = pd.cut(df_eng['tenure_days'], bins=5, labels=['New', 'Early', 'Mid', 'Late', 'Loyal'])
                        
                        # Find a good categorical feature to split by (excluding ID/Target)
                        # We use the top feature from the model if it's categorical, else pick the first categorical column
                        top_cat_feature = None
                        for f in feats['Feature']:
                            if f in df_eng.columns and df_eng[f].dtype == 'object' or pd.api.types.is_categorical_dtype(df_eng[f]):
                                top_cat_feature = f
                                break
                        
                        if not top_cat_feature:
                            # Fallback to any categorical
                            cat_cols = df_eng.select_dtypes(include=['object', 'category']).columns.tolist()
                            top_cat_feature = cat_cols[0] if cat_cols else None
                            
                        if top_cat_feature:
                            # Calculate Retention Rate (1 - Churn Rate) per Bin and Feature Group
                            # We need to map target to 0/1 properly
                            # Assuming Churn_Probability > 0.5 is Churn (1)
                            df_eng['is_churned'] = df_eng['Churn_Probability'] > 0.5
                            retention_pivot = df_eng.groupby([top_cat_feature, 'Tenure_Bin'])['is_churned'].mean().unstack()
                            retention_pivot = 1 - retention_pivot # Convert Churn Rate to Retention Rate
                            
                            fig_ret = px.imshow(retention_pivot, text_auto='.0%', color_continuous_scale='Greens', title=f"Retention Rate by {top_cat_feature} & Tenure", template='plotly_dark')
                            st.plotly_chart(fig_ret, use_container_width=True)
                        else:
                            st.info("Not enough categorical data for detailed Retention Heatmap.")

                    # Correlation Matrix (Numerical only)
                    st.subheader("Correlation Matrix")
                    num_corr = df_eng.select_dtypes(include=[np.number]).corr()
                    fig_corr = px.imshow(num_corr, color_continuous_scale='Greens', title="Variable Correlations", template='plotly_dark')
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # --- MODULE 6: AI STRATEGY & EXPORT ---
                    st.divider()
                    st.header("AI Strategic Insights")
                    strategies = generate_executive_summary(feats, churn_rate, df_eng, target_col)
                    for s in strategies:
                        st.info(s, icon="💡")
                    
                    st.divider()
                    st.header("Export Intelligence")
                    csv = df_eng.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Scored Data (CSV)",
                        data=csv,
                        file_name='scored_churn_data.csv',
                        mime='text/csv',
                    )
                
        except Exception as e:
            st.error(f"Critical Error: {e}")
            st.exception(e)
    else:
        st.info("Please upload a CSV or XLSX file to begin the analysis.")

if __name__ == "__main__":
    main()
