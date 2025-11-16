# Streamlit for the user interface
import streamlit as st
import base64  # Added for logo handling

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning models
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Distance calculation
from geopy.distance import geodesic

# Visualization
import altair as alt
import plotly.express as px

# Deep learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Date handling
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="🚨 Health Insurance Fraud Risk Scoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fraud Scoring Logic
def calculate_fraud_scores(df):
    """
    Calculate various fraud risk scores for insurance claims
    
    Args:
        df (pd.DataFrame): Input dataframe with claim data
        
    Returns:
        pd.DataFrame: DataFrame with added fraud risk scores
    """
    # Treatment Consistency Score (TCS)
    tcs_model = IsolationForest(contamination=0.1, random_state=1)
    tcs_encoded = pd.get_dummies(df['Treatment_Type'].fillna("Unknown"))
    df['TCS'] = 1 - np.abs(tcs_model.fit_predict(tcs_encoded))

    # Claim Amount Deviation Score (CADS)
    mean_amt = df['Claim_Amount_KES'].mean()
    std_amt = df['Claim_Amount_KES'].std()
    df['CADS'] = 1 - np.minimum(np.abs(df['Claim_Amount_KES'] - mean_amt) / (std_amt * 3), 1)

    # Visit Frequency Score (VFS)
    visit_freq = df.groupby('Customer_ID')['Claim_ID'].count().reset_index(name='Visit_Count')
    df = df.merge(visit_freq, on='Customer_ID', how='left')
    df['VFS'] = 1 - np.minimum(df['Visit_Count'] / df['Visit_Count'].max(), 1)

    # Provider Risk Score (PRS)
    provider_std = df.groupby('Provider_Type')['Claim_Amount_KES'].std().fillna(0)
    df['PRS'] = df['Provider_Type'].map(provider_std)
    scaler = StandardScaler()
    df['PRS'] = 1 - np.minimum(scaler.fit_transform(df[['PRS']]), 1)

    # Geographic Distance Score (GDS)
    def calc_distance(row):
        try:
            return geodesic((row['Customer_Lat'], row['Customer_Lon']),
                          (row['Hospital_Lat'], row['Hospital_Lon'])).km
        except:
            return np.nan
    
    df['Distance_km'] = df.apply(calc_distance, axis=1)
    df['GDS'] = 1 - np.minimum(df['Distance_km'] / 50, 1)

    # Claim Similarity Score (CSS) - Optimized for large datasets
    try:
        # First try efficient TF-IDF approach
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['Claim_Description'].fillna(""))
        
        # Calculate average similarity per claim
        df['CSS'] = 1 - np.array([cosine_similarity(tfidf_matrix[i:i+1], tfidf_matrix).mean() 
                                 for i in range(tfidf_matrix.shape[0])])
    except Exception as e:
        st.warning(f"Using simplified Claim Similarity Score due to memory constraints. Error: {str(e)}")
        # Fallback to simpler text length-based similarity
        df['Claim_Length'] = df['Claim_Description'].str.len().fillna(0)
        mean_len = df['Claim_Length'].mean()
        std_len = df['Claim_Length'].std()
        df['CSS'] = 1 - np.minimum(np.abs(df['Claim_Length'] - mean_len) / (std_len + 1e-6), 1)

    # Policy Utilization Score (PUS)
    df['PUS'] = np.minimum(df['Claim_Amount_KES'] / df['Benefit_Limit_KES'], 1)

    # Time Submission Score (TSS)
    df['Submission_Hour'] = pd.to_datetime(df['Submission_Timestamp'], dayfirst=True).dt.hour
    df['TSS'] = df['Submission_Hour'].apply(lambda h: 0 if h in range(0, 6) else 1)

    # Claim Amount Score (CAS)
    df['CAS'] = np.minimum(df['Claim_Amount_KES'] / (df['Claim_Amount_KES'] + df['Co_Pay_Amount_KES']), 1)

    # Final Risk Score
    df['Final_Risk_Score'] = (
        0.15 * df['TCS'] + 0.15 * df['CADS'] + 0.15 * df['VFS'] +
        0.15 * df['PRS'] + 0.1 * df['GDS'] + 0.1 * df['CSS'] +
        0.1 * df['PUS'] + 0.05 * df['TSS'] + 0.05 * df['CAS']
    )
    
    return df

def detect_anomalies(df):
    """Predict anomalies using all fraud scores with Isolation Forest"""
    features = df[['TCS', 'CADS', 'VFS', 'PRS', 'GDS', 'CSS', 'PUS', 'TSS', 'CAS']]
    
    model = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly_Score'] = model.fit_predict(features)
    df['Anomaly_Score'] = 1 - (df['Anomaly_Score'] + 1) / 2  # Convert to 0-1 scale
    
    return df

def cluster_claims(df):
    """Group claims into risk clusters using K-Means"""
    features = df[['TCS', 'CADS', 'VFS', 'PRS', 'GDS', 'CSS', 'PUS', 'TSS', 'CAS']]
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Risk_Cluster'] = kmeans.fit_predict(features)
    
    # Label clusters (assuming cluster 0 is highest risk)
    cluster_labels = {0: 'High Risk', 1: 'Medium Risk', 2: 'Low Risk'}
    df['Risk_Label'] = df['Risk_Cluster'].map(cluster_labels)
    
    return df

def autoencoder_detection(df):
    """Detect anomalies using reconstruction error from Autoencoder"""
    features = df[['TCS', 'CADS', 'VFS', 'PRS', 'GDS', 'CSS', 'PUS', 'TSS', 'CAS']]
    
    # Normalize
    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)
    
    # Build autoencoder
    input_dim = X.shape[1]
    encoding_dim = 3
    
    autoencoder = Sequential([
        Dense(encoding_dim, activation="relu", input_shape=(input_dim,)),
        Dense(input_dim, activation="sigmoid")
    ])
    
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=50, batch_size=32, verbose=0)
    
    # Calculate reconstruction error
    reconstructions = autoencoder.predict(X, verbose=0)
    df['Reconstruction_Error'] = np.mean(np.square(X - reconstructions), axis=1)
    
    return df

def get_base64_logo(image_path):
    """Load the logo image as Base64 (Prevents broken image issues)"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        return None

def preprocess_client_data(df):
    """
    Preprocess client data to match our expected schema
    
    Args:
        df (pd.DataFrame): Raw client data
        
    Returns:
        pd.DataFrame: Processed data with matching columns
    """
    # Create a copy to avoid modifying the original
    processed = df.copy()
    
    # Basic column mappings
    column_mapping = {
        'MemberID': 'Customer_ID',
        'Claimnumber': 'Claim_ID',
        'TotalClaimed': 'Claim_Amount_KES',
        'Ailment': 'Claim_Description',
        'ProviderType': 'Provider_Type',
        'Benefit': 'Claim_Type',
        'BenefitSections': 'Treatment_Type',
        'SchemeName': 'Policy_Type'
    }
    
    # Apply column renaming
    processed = processed.rename(columns=column_mapping)
    
    # Handle missing columns with default values
    processed['Customer_Lat'] = -1.286389  # Default Nairobi coordinates
    processed['Customer_Lon'] = 36.817223
    processed['Hospital_Lat'] = -1.286389
    processed['Hospital_Lon'] = 36.817223
    processed['Benefit_Limit_KES'] = processed['Claim_Amount_KES'] * 10  # Estimate
    processed['Co_Pay_Amount_KES'] = processed['Claim_Amount_KES'] * 0.1  # 10% co-pay
    
    # Convert dates with dayfirst=True for European-style dates (DD/MM/YYYY)
    try:
        processed['Policy_Start_Date'] = pd.to_datetime(
            processed['DOB'], 
            dayfirst=True
        ).dt.date
        processed['Submission_Timestamp'] = pd.to_datetime(
            processed['ClaimDate'], 
            dayfirst=True
        ).dt.date
    except Exception as e:
        st.error(f"Error converting dates: {str(e)}")
        st.stop()
    
    # Calculate derived fields
    processed['Previous_Claims'] = processed.groupby('Customer_ID')['Claim_ID'].transform('count') - 1
    processed['Claim_Frequency'] = processed['Previous_Claims'] / 12  # Assuming 1 year history
    
    # Fill missing values
    processed['Claim_Description'] = processed['Claim_Description'].fillna('Unknown')
    processed['Treatment_Type'] = processed['Treatment_Type'].fillna('Unknown')
    
    return processed

def main():
    # Custom CSS for fixed header and marquee
    st.markdown("""
    <style>
        /* Fixed header styling */
        .fixed-header {
            position: sticky;
            top: 0;
            background-color: white;
            z-index: 100;
            padding: 1rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Marquee styling */
        .marquee {
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            box-sizing: border-box;
            padding: 0.5rem 0;
            color: #555;
        }
        
        .marquee span {
            display: inline-block;
            padding-left: 100%;
            animation: marquee 15s linear infinite;
        }
        
        @keyframes marquee {
            0%   { transform: translateX(0); }
            100% { transform: translateX(-100%); }
        }
        
        /* Sidebar default styles */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            box-shadow: 2px 0px 8px rgba(0, 0, 0, 0.2);
            overflow-y: auto;
            height: 100vh;
            width: 300px;
            position: fixed;
            left: 0;
            z-index: 1000;
        }

        /* Adjust Sidebar for Mobile */
        @media screen and (max-width: 768px) {
            [data-testid="stSidebar"] {
                width: 250px;
            }
        }

        /* Push content when sidebar is open */
        .main-content {
            margin-left: 300px;
            transition: margin-left 0.3s ease-in-out;
        }

        /* Adjust main content margin for mobile */
        @media screen and (max-width: 768px) {
            .main-content {
                margin-left: 250px;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Fixed header section
    st.markdown("""
    <div class="fixed-header">
        <h1 style="margin: 0;">🚨 Health Insurance Fraud Risk Scoring</h1>
        <div class="marquee">
            <span>This tool analyzes health insurance claims to detect potential fraud using multiple risk indicators. Upload your claims data below to identify suspicious claims.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar Logo ---
    logo_path = "logo.png"
    logo_base64 = get_base64_logo(logo_path)
    
    st.sidebar.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            {f'<img src="data:image/png;base64,{logo_base64}" style="max-width: 80%;">' if logo_base64 else '<p style="color: red;">⚠️ Logo Not Found</p>'}
        </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar Configuration ---
    st.sidebar.header("Data Upload & Settings")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload Claims Data (CSV)", 
        type=["csv"],
        help="Upload a CSV file containing health insurance claims data"
    )
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if this is client data format
            is_client_data = 'MemberID' in df.columns and 'Claimnumber' in df.columns
            
            if is_client_data:
                with st.spinner("Preprocessing client data format..."):
                    df = preprocess_client_data(df)
            
            # Check for required columns
            required_columns = [
                'Customer_ID', 'Claim_ID', 'Claim_Amount_KES', 'Customer_Lat', 'Customer_Lon',
                'Hospital_Lat', 'Hospital_Lon', 'Treatment_Type', 'Provider_Type', 'Claim_Description',
                'Benefit_Limit_KES', 'Policy_Start_Date', 'Submission_Timestamp', 'Co_Pay_Amount_KES'
            ]
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns after preprocessing: {', '.join(missing_cols)}")
                st.stop()
            
            # --- Data Processing ---
            with st.spinner("Calculating fraud risk scores..."):
                # Downcast numeric types to save memory
                numeric_cols = ['Claim_Amount_KES', 'Benefit_Limit_KES', 'Co_Pay_Amount_KES']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, downcast='float')
                
                df_processed = calculate_fraud_scores(df.copy())
                
                # Add region for filtering
                df_processed['Region'] = pd.cut(
                    df_processed['Customer_Lat'], 
                    bins=3, 
                    labels=['North', 'Central', 'South']
                )
                
                st.session_state.processed_data = df_processed
                                   
            # --- Analysis Method Selection ---
            st.sidebar.subheader("🔍 Analysis Method")
            
            # Categorize analysis methods
            analysis_category = st.sidebar.selectbox(
                "Select Analysis Category",
                ["Rule-Based Scoring", "Machine Learning Models"],
                index=0,
                help="Choose between rule-based scoring or machine learning approaches"
            )
            
            if analysis_category == "Machine Learning Models":
                analysis_method = st.sidebar.selectbox(
                    "Select ML Model",
                    ["Isolation Forest (Anomaly Detection)", 
                     "K-Means (Risk Clustering)", 
                     "Autoencoder (Deep Learning)"],
                    index=0,
                    help="Choose the machine learning approach for fraud detection"
                )
                
                if analysis_method == "Isolation Forest (Anomaly Detection)":
                    df_processed = detect_anomalies(df_processed)
                    threshold = st.sidebar.slider(
                        "Anomaly Threshold", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.9, 
                        step=0.05,
                        help="Higher values flag more claims as anomalous"
                    )
                    df_processed['Flagged'] = df_processed['Anomaly_Score'] > threshold
                    
                elif analysis_method == "K-Means (Risk Clustering)":
                    df_processed = cluster_claims(df_processed)
                    risk_level = st.sidebar.selectbox(
                        "Flag Risk Level",
                        ["High Risk", "High and Medium Risk"],
                        index=0,
                        help="Which clusters to flag as suspicious"
                    )
                    df_processed['Flagged'] = (df_processed['Risk_Label'] == risk_level) if risk_level == "High Risk" else (df_processed['Risk_Label'] != "Low Risk")
                    
                elif analysis_method == "Autoencoder (Deep Learning)":
                    df_processed = autoencoder_detection(df_processed)
                    threshold = st.sidebar.slider(
                        "Reconstruction Error Threshold", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.95, 
                        step=0.01,
                        help="Percentile of reconstruction error to flag"
                    )
                    error_threshold = df_processed['Reconstruction_Error'].quantile(threshold)
                    df_processed['Flagged'] = df_processed['Reconstruction_Error'] > error_threshold
                    
            else:  # Rule-Based Scoring
                analysis_method = "Rule-Based Scoring"
            
            # --- Sidebar Filters ---
            with st.sidebar.expander("🔍 Filter by Provider & Region", expanded=False):
                # Provider filter
                provider_options = sorted(df_processed['Provider_Type'].dropna().unique())
                selected_providers = st.multiselect(
                    "Provider Types",
                    options=provider_options,
                    default=provider_options
                )
                
                # Region filter
                region_options = sorted(df_processed['Region'].dropna().unique())
                selected_regions = st.multiselect(
                    "Regions",
                    options=region_options,
                    default=region_options
                )
            
            st.sidebar.subheader("⚙️ Threshold Settings")
            if analysis_category == "Rule-Based Scoring":
                threshold = st.sidebar.slider(
                    "Risk Score Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.4, 
                    step=0.05,
                    help="Claims with scores below this threshold will be flagged as suspicious"
                )
                df_processed['Flagged'] = df_processed['Final_Risk_Score'] < threshold
            
            # Apply filters
            filtered_data = df_processed[
                (df_processed['Provider_Type'].isin(selected_providers)) &
                (df_processed['Region'].isin(selected_regions))
            ]
            
            # --- Main Dashboard ---
            st.success(f"Data processed using {analysis_method}!")
            
            # Key Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Claims", len(filtered_data))
            with col2:
                st.metric("Flagged Claims", filtered_data['Flagged'].sum())
            with col3:
                st.metric("Flag Rate", f"{filtered_data['Flagged'].mean()*100:.1f}%")
            
            # --- Method-Specific Visualizations ---
            if analysis_category == "Machine Learning Models":
                with st.expander(f"🔍 {analysis_method} Analysis", expanded=True):
                    if "Isolation Forest" in analysis_method:
                        fig = px.histogram(
                            filtered_data,
                            x='Anomaly_Score',
                            color='Flagged',
                            nbins=50,
                            title="Distribution of Anomaly Scores"
                        )
                    elif "K-Means" in analysis_method:
                        fig = px.scatter(
                            filtered_data,
                            x='Final_Risk_Score',
                            y='Claim_Amount_KES',
                            color='Risk_Label',
                            hover_data=['Claim_ID'],
                            title="Claims by Risk Cluster"
                        )
                    elif "Autoencoder" in analysis_method:
                        fig = px.histogram(
                            filtered_data,
                            x='Reconstruction_Error',
                            color='Flagged',
                            nbins=50,
                            title="Distribution of Reconstruction Errors"
                        )
                    st.plotly_chart(fig, use_container_width=True)
            
            # --- Data Preview ---
            with st.expander("📋 View Processed Data", expanded=False):
                st.dataframe(filtered_data.head())
            
            # --- Fraud Score Breakdown ---
            with st.expander("📊 Fraud Score Breakdown", expanded=False):
                st.markdown("""
                **Score Components:**
                - **TCS (Treatment Consistency):** Measures consistency of treatments across claims
                - **CADS (Claim Amount Deviation):** Measures deviation from average claim amounts
                - **VFS (Visit Frequency):** Measures frequency of claims by customer
                - **PRS (Provider Risk):** Measures risk level of provider type
                - **GDS (Geographic Distance):** Measures distance between customer and provider
                - **CSS (Claim Similarity):** Measures similarity between claim descriptions
                - **PUS (Policy Utilization):** Measures utilization of policy benefits
                - **TSS (Time Submission):** Measures timing of claim submission
                - **CAS (Claim Amount Score):** Measures claim amount relative to co-pay
                """)
                
                # Show sample scores
                st.dataframe(filtered_data[[
                    'Claim_ID', 'TCS', 'CADS', 'VFS', 'PRS', 'GDS', 
                    'CSS', 'PUS', 'TSS', 'CAS', 'Final_Risk_Score'
                ]].head())
            
            # --- Visualizations ---
            st.subheader("📈 Risk Analysis Visualizations")
            
            # Score Distribution
            with st.container(border=True):
                st.markdown("### Score Distribution")
                score_chart = alt.Chart(filtered_data).mark_bar().encode(
                    alt.X("Final_Risk_Score:Q", bin=alt.Bin(maxbins=20), title="Risk Score"),
                    y='count()',
                    color=alt.condition(
                        alt.datum.Flagged == True,
                        alt.value('red'),
                        alt.value('steelblue')
                    ),
                    tooltip=['count()']
                ).properties(
                    width=700,
                    height=400
                )
                st.altair_chart(score_chart, use_container_width=True)
            
            # Risk Heatmap
            with st.container(border=True):
                st.markdown("### Risk Heatmap by Claim Type & Region")
                heatmap_data = filtered_data.groupby(['Claim_Type', 'Region'])['Final_Risk_Score'].mean().reset_index()
                
                heatmap = alt.Chart(heatmap_data).mark_rect().encode(
                    x=alt.X('Claim_Type:N', title='Claim Type'),
                    y=alt.Y('Region:N', title='Region'),
                    color=alt.Color('Final_Risk_Score:Q', 
                                  scale=alt.Scale(scheme='reds', reverse=True),
                                  legend=alt.Legend(title="Avg Risk Score")),
                    tooltip=['Claim_Type', 'Region', 'Final_Risk_Score']
                ).properties(
                    width=600,
                    height=300
                )
                st.altair_chart(heatmap, use_container_width=True)
            
            # Trend Analysis
            with st.container(border=True):
                st.markdown("### Flagged Claims Over Time")
                filtered_data['Submission_Date'] = pd.to_datetime(
                    filtered_data['Submission_Timestamp'],
                    dayfirst=True
                ).dt.date
                
                trend_data = filtered_data.groupby('Submission_Date')['Flagged'].sum().reset_index()
                
                trend_chart = alt.Chart(trend_data).mark_line(point=True).encode(
                    x=alt.X('Submission_Date:T', title='Date'),
                    y=alt.Y('Flagged:Q', title='Flagged Claims Count'),
                    tooltip=['Submission_Date:T', 'Flagged']
                ).properties(
                    width=700,
                    height=300
                )
                st.altair_chart(trend_chart, use_container_width=True)
            
            # Geographic View
            with st.container(border=True):
                st.markdown("### 🗺️ Geographic Distribution of Flagged Claims")
                
                # Prepare map data
                map_data = filtered_data[filtered_data['Flagged']][
                    ['Customer_Lat', 'Customer_Lon']
                ].dropna()
                map_data = map_data.rename(columns={
                    'Customer_Lat': 'latitude',
                    'Customer_Lon': 'longitude'
                })
                
                if not map_data.empty:
                    st.map(map_data, use_container_width=True)
                else:
                    st.info("No flagged claims with valid coordinates to display on map.")
            
            # --- Download Options ---
            st.subheader("📥 Download Results")
            
            col4, col5 = st.columns(2)
            with col4:
                # Download flagged claims
                flagged_csv = filtered_data[filtered_data['Flagged']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Flagged Claims",
                    data=flagged_csv,
                    file_name="flagged_claims.csv",
                    mime="text/csv"
                )
            
            with col5:
                # Download full results
                full_csv = filtered_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download All Results",
                    data=full_csv,
                    file_name="fraud_risk_scores.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a CSV file containing health insurance claims data to begin analysis.")
        
        # Sample data structure guidance
        with st.expander("ℹ️ Expected Data Format"):
            st.markdown("""
            The CSV file should contain at least these columns:
            
            - **Customer_ID**: Unique identifier for the customer
            - **Claim_ID**: Unique identifier for the claim
            - **Claim_Amount_KES**: Amount claimed in KES
            - **Customer_Lat**: Customer's latitude
            - **Customer_Lon**: Customer's longitude
            - **Hospital_Lat**: Treatment facility latitude
            - **Hospital_Lon**: Treatment facility longitude
            - **Treatment_Type**: Type of treatment received
            - **Provider_Type**: Type of healthcare provider
            - **Claim_Description**: Description of the claim
            - **Benefit_Limit_KES**: Policy benefit limit in KES
            - **Policy_Start_Date**: When the policy started
            - **Submission_Timestamp**: When claim was submitted
            - **Co_Pay_Amount_KES**: Customer co-pay amount in KES
            
            Additional columns will be preserved but not used in scoring.
            
            The tool also supports client data format with these key fields:
            - MemberID, Claimnumber, TotalClaimed, Ailment, ProviderType, 
            Benefit, BenefitSections, SchemeName, DOB, ClaimDate
            """)

if __name__ == "__main__":
    main()