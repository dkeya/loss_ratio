# healthcare_claims_analyzer.py
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import logging
import base64
from tempfile import NamedTemporaryFile
from io import StringIO
import os
import warnings
import dask.dataframe as dd
import shap
from fastapi import FastAPI
from pydantic import BaseModel
import optuna
import joblib
import json
from typing import List
from fastapi import HTTPException

# Machine Learning Imports
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.inspection import partial_dependence
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# EDA and Visualization
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components

# PDF Reporting
from fpdf import FPDF

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("healthcare_claims.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Class for monitoring model performance and data drift"""
    def __init__(self):
        self.performance_history = []
        self.data_drift_scores = []
        self.training_date = datetime.now()
        
    def log_performance(self, model_name, metrics):
        """Log model performance metrics"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'model': model_name,
            **metrics
        })
        
    def check_data_drift(self, current_data, reference_data):
        """Calculate data drift metrics"""
        drift_metrics = {}
        
        # For numerical columns
        num_cols = current_data.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            # Kolmogorov-Smirnov test
            from scipy.stats import ks_2samp
            stat, p = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
            drift_metrics[col] = {'ks_stat': stat, 'ks_p': p}
        
        # For categorical columns
        cat_cols = current_data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            # Population Stability Index
            ref_counts = reference_data[col].value_counts(normalize=True)
            curr_counts = current_data[col].value_counts(normalize=True)
            psi = np.sum((curr_counts - ref_counts) * np.log(curr_counts / ref_counts))
            drift_metrics[col] = {'psi': psi}
        
        self.data_drift_scores.append({
            'timestamp': datetime.now(),
            'drift_metrics': drift_metrics
        })
        return drift_metrics

class PDF(FPDF):
    """Custom PDF class for professional reporting"""
    def header(self):
        self.image('healthcare_logo.png', 10, 8, 33)
        self.set_font('DejaVuSans', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Healthcare Claims Analysis Report', 0, 0, 'C')
        self.ln(20)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVuSans', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
    
    def _add_pdf_table(self, data, col_widths=None):
        """Helper method to add tables to PDF reports"""
        if col_widths is None:
            col_width = self.w / len(data.columns)
            col_widths = [col_width] * len(data.columns)
        
        # Header
        self.set_font('DejaVuSans', 'B', 10)
        for i, col in enumerate(data.columns):
            self.cell(col_widths[i], 10, str(col), border=1)
        self.ln()
        
        # Data
        self.set_font('DejaVuSans', '', 10)
        for _, row in data.iterrows():
            for i, val in enumerate(row):
                self.cell(col_widths[i], 10, str(val), border=1)
            self.ln()

class ClaimAmountPredictor:
    """Enhanced healthcare claims analyzer with predictive modeling and group analysis"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.data = None
        self.clean_data = None
        self.feature_importance = None
        self.baseline_metrics = None
        self.category_order = ['Silver', 'Gold', 'Platinum']
        self.required_prediction_columns = None
        self.logger = logging.getLogger(__name__)
        self.available_values = {}  # Store available values for dropdowns
        self.monitor = ModelMonitor()
        self.fraud_model = None
        self.training_date = None
        
    def load_data(self, uploaded_file):
        """Optimized data loading with memory management"""
        try:
            self.logger.info(f"Loading file: {uploaded_file.name}")
            
            # Validate file type
            if not (uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.xlsx')):
                raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
            
            # Use dask for large files (>50MB)
            if uploaded_file.size > 50 * 1024 * 1024:
                st.warning("Large file detected - using optimized loading")
                
                # Save to temp file for dask
                with NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    df = dd.read_csv(tmp_path)
                    df = df.compute()  # Convert to pandas
                finally:
                    os.unlink(tmp_path)
                    
                self.logger.info(f"Loaded {len(df)} rows using dask")
            else:
                # Use pandas for smaller files
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # Excel
                    df = pd.read_excel(uploaded_file)
            
            # Validate minimum data requirements
            if len(df) < 100:
                raise ValueError("Insufficient data. Please upload at least 100 claims.")
                
            # Validate required columns
            required_cols = ['Employee_ID', 'Claim_Amount_KES', 'Submission_Date', 'Employer']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Optimize memory usage
            df = self._optimize_dataframe(df)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(' ', '_')
            self.data = df
            
            # Initialize available values for dropdowns
            self._initialize_available_values()
            
            # Validate healthcare-specific data
            self._validate_healthcare_data(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            st.error(f"Data loading error: {str(e)}")
            return None
    
    def _validate_healthcare_data(self, df):
        """Validate healthcare-specific data quality"""
        # Validate ICD-10 codes if present
        if 'Diagnosis' in df.columns:
            icd10_pattern = r'^[A-TV-Z][0-9][0-9AB](\.[0-9A-TV-Z]{1,4})?$'
            invalid_dx = ~df['Diagnosis'].str.match(icd10_pattern, na=False)
            if invalid_dx.any():
                self.logger.warning(f"Found {invalid_dx.sum()} invalid ICD-10 codes")
                
        # Validate claim amounts are positive
        if 'Claim_Amount_KES' in df.columns:
            if (df['Claim_Amount_KES'] < 0).any():
                raise ValueError("Negative claim amounts found")
                
        # Validate temporal consistency
        if 'Submission_Date' in df.columns and 'Service_Date' in df.columns:
            if (df['Submission_Date'] < df['Service_Date']).any():
                raise ValueError("Claims submitted before service date")
    
    def _initialize_available_values(self):
        """Initialize dropdown options from data"""
        if self.data is not None:
            self.available_values = {
                'Visit_Type': sorted(self.data['Visit_Type'].astype(str).unique()) if 'Visit_Type' in self.data.columns else [],
                'Diagnosis': sorted(self.data['Diagnosis'].astype(str).unique()) if 'Diagnosis' in self.data.columns else [],
                'Treatment': sorted(self.data['Treatment'].astype(str).unique()) if 'Treatment' in self.data.columns else [],
                'Provider_Name': sorted(self.data['Provider_Name'].astype(str).unique()) if 'Provider_Name' in self.data.columns else [],
                'Hospital_County': sorted(self.data['Hospital_County'].astype(str).unique()) if 'Hospital_County' in self.data.columns else [],
                'Employee_Gender': sorted(self.data['Employee_Gender'].astype(str).unique()) if 'Employee_Gender' in self.data.columns else [],
                'Category': sorted(self.data['Category'].astype(str).unique()) if 'Category' in self.data.columns else [],
                'Employer': sorted(self.data['Employer'].astype(str).unique()) if 'Employer' in self.data.columns else [],
                'Department': sorted(self.data['Department'].astype(str).unique()) if 'Department' in self.data.columns else [],
            }
    
    def _optimize_dataframe(self, df):
        """Reduce memory usage of dataframe"""
        # Downcast numeric columns more aggressively
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='unsigned' if df[col].min() >= 0 else 'integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object to category where possible
        for col in df.select_dtypes(include=['object']).columns:
            if len(df[col].unique()) / len(df[col]) < 0.9:  # Higher threshold
                df[col] = df[col].astype('category')
        
        return df
    
    def clean_and_prepare_data(self):
        """Comprehensive data cleaning pipeline with group-specific features"""
        if self.data is None:
            st.warning("No data loaded")
            return False
            
        try:
            with st.spinner("Cleaning and preparing data..."):
                # Create copy while preserving original
                df = self.data.copy()
                self.raw_data = self.data.copy()  # Preserve original
                
                # ====== Data Type Validation ======
                type_conversions = {
                    'Employee_Age': 'int',
                    'Claim_Amount_KES': 'float',
                    'Co_Payment_KES': 'float',
                    'Submission_Date': 'datetime64[ns]',
                    'Service_Date': 'datetime64[ns]',
                    'Hire_Date': 'datetime64[ns]'
                }
                for col, dtype in type_conversions.items():
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(dtype)
                        except (ValueError, TypeError):
                            if dtype == 'datetime64[ns]':
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                            else:
                                df[col] = pd.to_numeric(df[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], 
                                                      errors='coerce')
                
                # ====== Group-Specific Features ======
                # Clean employer names
                if 'Employer' in df.columns:
                    df['Employer'] = df['Employer'].str.upper().str.strip()
                
                # Create department if not exists
                if 'Department' not in df.columns and 'Division' in df.columns:
                    df['Department'] = df['Division'].str.title()
                else:
                    df['Department'] = 'General'  # Default value if no department info
                
                # Add tenure groups if hire date exists
                if 'Hire_Date' in df.columns:
                    df['Tenure'] = (datetime.now() - df['Hire_Date']).dt.days / 365
                    df['Tenure_Group'] = pd.cut(df['Tenure'],
                                              bins=[0, 1, 5, 100],
                                              labels=['<1yr', '1-5yrs', '5+yrs'])
                
                # Add salary bands if salary data exists
                if 'Salary' in df.columns:
                    df['Salary_Band'] = pd.qcut(df['Salary'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                
                # ====== Missing Values Handling ======
                missing_report = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
                missing_report['% Missing'] = (missing_report['Missing Values'] / len(df)) * 100
                
                cols_to_drop = missing_report[missing_report['% Missing'] > 70].index.tolist()
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    st.warning(f"Dropped columns with >70% missing values: {', '.join(cols_to_drop)}")
                
                # Fill missing values with intelligent defaults
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].fillna('Unknown')
                    elif df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                
                # ====== Outlier Detection and Handling ======
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_cols:
                    if col in ['Employee_ID']:  # Skip ID columns
                        continue
                        
                    # IQR Method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Z-score Method
                    z_scores = (df[col] - df[col].mean()) / df[col].std()
                    
                    # Cap outliers but keep track of modifications
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound) | (abs(z_scores) > 3)
                    if outlier_mask.any():
                        df[f'{col}_outlier'] = outlier_mask.astype(int)
                        df[col] = np.where(outlier_mask, df[col].median(), df[col])
                
                # ====== Invalid Value Detection ======
                if 'Claim_Amount_KES' in df.columns:
                    df['Claim_Amount_KES'] = df['Claim_Amount_KES'].abs()
                
                if 'Employee_Age' in df.columns:
                    df['Employee_Age'] = df['Employee_Age'].apply(
                        lambda x: x if 18 <= x <= 100 else np.nan
                    ).fillna(df['Employee_Age'].median())
                
                # ====== Deduplication Handling ======
                dup_cols = [c for c in df.columns if c not in ['Claim_Amount_KES', 'Submission_Date']]
                df = df.drop_duplicates(subset=dup_cols, keep='first')
                
                # ====== Categorical Encoding Verification ======
                categorical_cols = ['Visit_Type', 'Provider_Name', 'Hospital_County', 
                                  'Employee_Gender', 'Category', 'Employer', 'Department']
                for col in categorical_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.title().str.strip()
                        freq = df[col].value_counts(normalize=True)
                        df[col] = df[col].apply(lambda x: x if freq.get(x, 0) > 0.05 else 'Other')
                
                # ====== Inconsistent Formatting Correction ======
                currency_cols = [c for c in df.columns if '_KES' in c]
                for col in currency_cols:
                    if col in df.columns:
                        df[col] = df[col].replace('[^\d.]', '', regex=True).astype(float)
                
                date_cols = [c for c in df.columns if 'Date' in c or 'date' in c.lower()]
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # ====== Feature Engineering ======
                if 'Pre_Authorization_Required' in df.columns:
                    df['Is_Pre_Authorized'] = df['Pre_Authorization_Required'].map({'Yes': 1, 'No': 0})
                
                # Cap utilization features
                cap_cols = ['Inpatient_Cap_KES', 'Outpatient_Cap_KES', 
                            'Optical_Cap_KES', 'Dental_Cap_KES', 'Maternity_Cap_KES']
                for col in cap_cols:
                    if col in df.columns:
                        df[f'{col}_Utilization'] = df['Claim_Amount_KES'] / df[col].replace(0, np.nan)
                
                # Diagnosis and treatment grouping
                if 'Diagnosis' in df.columns:
                    df['Diagnosis_Group'] = df['Diagnosis'].str.extract(r'([A-Za-z\s]+)')[0].str.strip()
                    df['Diagnosis_Group'] = df['Diagnosis_Group'].apply(
                        lambda x: x if len(str(x)) > 3 else 'Other')
                        
                if 'Treatment' in df.columns:
                    df['Treatment_Type'] = df['Treatment'].str.extract(r'([A-Za-z\s]+)')[0].str.strip()
                    df['Treatment_Type'] = df['Treatment_Type'].apply(
                        lambda x: x if len(str(x)) > 3 else 'Other')
                
                # Age bins
                if 'Employee_Age' in df.columns:
                    df['Age_Group'] = pd.cut(df['Employee_Age'],
                                           bins=[0, 25, 35, 45, 55, 65, 100],
                                           labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
                
                # Claim amount categories
                if 'Claim_Amount_KES' in df.columns:
                    df['Claim_Size'] = pd.qcut(df['Claim_Amount_KES'],
                                             q=4,
                                             labels=['Small', 'Medium', 'Large', 'Very Large'])
                
                # Date features
                if 'Submission_Date' in df.columns:
                    df['Claim_Weekday'] = df['Submission_Date'].dt.day_name()
                    df['Claim_Month'] = df['Submission_Date'].dt.month_name()
                    df['Claim_Quarter'] = df['Submission_Date'].dt.quarter
                
                # Fraud detection features
                df['Claim_Amount_to_Mean'] = df['Claim_Amount_KES'] / df['Claim_Amount_KES'].mean()
                df['Same_Day_Claims'] = df.duplicated(subset=['Employee_ID', 'Submission_Date'], keep=False).astype(int)
                
                # Group-level features
                if 'Employer' in df.columns:
                    # Calculate employer-level statistics
                    employer_stats = df.groupby('Employer')['Claim_Amount_KES'].agg(['mean', 'std']).reset_index()
                    employer_stats.columns = ['Employer', 'Employer_Mean_Claim', 'Employer_Std_Claim']
                    df = pd.merge(df, employer_stats, on='Employer', how='left')
                    df['Employer_Z_Score'] = (df['Claim_Amount_KES'] - df['Employer_Mean_Claim']) / df['Employer_Std_Claim']
                
                self.clean_data = df
                self.logger.info("Data cleaning completed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            st.error(f"Data cleaning failed: {str(e)}")
            return False
    
    def generate_data_report(self):
        """Generate comprehensive exploratory analysis report"""
        if self.clean_data is None:
            st.warning("No cleaned data available")
            return
            
        with st.spinner("Generating data profile..."):
            try:
                profile = ProfileReport(
                    self.clean_data,
                    explorative=True,
                    minimal=True,
                    correlations=None,
                    missing_diagrams=False
                )
                st_profile_report(profile)
                
            except Exception as e:
                self.logger.error(f"Failed to generate data profile: {str(e)}")
                st.warning("Showing basic data summary instead")
                st.dataframe(self.clean_data.describe(include='all'))
    
    def preprocess_data(self, target='Claim_Amount_KES'):
        """Prepare data for modeling with enhanced validation"""
        try:
            if self.clean_data is None:
                raise ValueError("No cleaned data available")
                
            # Define features and target
            X = self.clean_data.drop(columns=[target], errors='ignore')
            y = self.clean_data[target]
            
            # Define feature types with validation
            categorical_features = [
                'Visit_Type', 'Diagnosis_Group', 'Treatment_Type', 
                'Provider_Name', 'Hospital_County', 'Employee_Gender',
                'Claim_Weekday', 'Claim_Month', 'Employer', 'Category',
                'Age_Group', 'Claim_Size', 'Department', 'Tenure_Group'
            ]
            
            numerical_features = [
                'Employee_Age', 'Co_Payment_KES', 'Is_Pre_Authorized',
                'Inpatient_Cap_KES_Utilization', 'Outpatient_Cap_KES_Utilization',
                'Optical_Cap_KES_Utilization', 'Dental_Cap_KES_Utilization',
                'Maternity_Cap_KES_Utilization', 'Claim_Amount_to_Mean',
                'Same_Day_Claims', 'Employer_Z_Score'
            ]
            
            # Validate features exist in data
            categorical_features = [f for f in categorical_features if f in X.columns]
            numerical_features = [f for f in numerical_features if f in X.columns]
            
            if not categorical_features and not numerical_features:
                raise ValueError("No valid features found for modeling")
            
            # Create preprocessing pipeline
            self.preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
            
            # Store required columns for prediction
            self.required_prediction_columns = numerical_features + categorical_features
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            st.error(f"Preprocessing error: {str(e)}")
            return None, None
    
    def train_model(self, model_type="Gradient Boosting", target='Claim_Amount_KES', 
                test_size=0.2, cv_folds=5, do_tuning=False, max_iter=20):
        """Enhanced model training with multiple algorithms"""
        try:
            X, y = self.preprocess_data(target)
            if X is None or y is None:
                return False
            
            # Split data with temporal validation if date is available
            if 'Submission_Date' in X.columns:
                X_sorted = X.sort_values('Submission_Date')
                y_sorted = y[X_sorted.index]
                split_idx = int(len(X_sorted) * (1 - test_size))
                X_train, X_test = X_sorted.iloc[:split_idx], X_sorted.iloc[split_idx:]
                y_train, y_test = y_sorted.iloc[:split_idx], y_sorted.iloc[split_idx:]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42)
        
            # Initialize models
            models = {}
            param_grids = {}
        
            if model_type == "Gradient Boosting" or model_type == "Auto Select Best":
                models['GradientBoosting'] = GradientBoostingRegressor(random_state=42)
                param_grids['GradientBoosting'] = {
                    'regressor__n_estimators': [100, 200, 300],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__max_depth': [3, 5, 7]
                }
        
            if model_type == "Random Forest" or model_type == "Auto Select Best":
                models['RandomForest'] = RandomForestRegressor(random_state=42)
                param_grids['RandomForest'] = {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [5, 10, None],
                    'regressor__min_samples_split': [2, 5, 10]
                }
        
            if model_type == "XGBoost" or model_type == "Auto Select Best":
                models['XGBoost'] = xgb.XGBRegressor(random_state=42)
                param_grids['XGBoost'] = {
                    'regressor__n_estimators': [50, 100, 200],
                    'regressor__max_depth': [3, 5, 7],
                    'regressor__learning_rate': [0.01, 0.05, 0.1],
                    'regressor__tree_method': ['gpu_hist']  # Enable GPU acceleration
                }
        
            if model_type == "Neural Network" or model_type == "Auto Select Best":
                models['NeuralNetwork'] = MLPRegressor(random_state=42, max_iter=500)
                param_grids['NeuralNetwork'] = {
                    'regressor__hidden_layer_sizes': [(50,), (50, 25), (100, 50)],
                    'regressor__activation': ['relu', 'tanh'],
                    'regressor__solver': ['adam', 'sgd']
                }
        
            # Train and evaluate each model
            results = []
            best_model = None
            best_score = -np.inf
        
            for name, model in models.items():
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('regressor', model)
                ])
            
                # Hyperparameter tuning if enabled
                if do_tuning and name in param_grids:
                    if name == 'XGBoost':
                        # Special handling for XGBoost with Optuna
                        def objective(trial):
                            params = {
                                'regressor__n_estimators': trial.suggest_int('regressor__n_estimators', 50, 500),
                                'regressor__max_depth': trial.suggest_int('regressor__max_depth', 3, 10),
                                'regressor__learning_rate': trial.suggest_float('regressor__learning_rate', 0.001, 0.1, log=True),
                                'regressor__subsample': trial.suggest_float('regressor__subsample', 0.5, 1.0),
                                'regressor__colsample_bytree': trial.suggest_float('regressor__colsample_bytree', 0.5, 1.0)
                            }
                            pipeline.set_params(**params)
                            pipeline.fit(X_train, y_train)
                            return mean_absolute_error(y_test, pipeline.predict(X_test))
                    
                        study = optuna.create_study(direction='minimize')
                        study.optimize(objective, n_trials=max_iter)
                        best_params = study.best_params
                        pipeline.set_params(**best_params)
                    else:
                        # Standard RandomizedSearchCV for other models
                        search = RandomizedSearchCV(
                            pipeline, 
                            param_grids[name], 
                            n_iter=max_iter, 
                            cv=cv_folds,
                            scoring='neg_mean_absolute_error', 
                            random_state=42
                        )
                        search.fit(X_train, y_train)
                        pipeline = search.best_estimator_
                        self.logger.info(f"Best params for {name}: {search.best_params_}")
            
                # Train model
                pipeline.fit(X_train, y_train)
            
                # Evaluate
                y_pred = pipeline.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)
            
                results.append({
                    'Model': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                })
            
                # Track best model
                if r2 > best_score:
                    best_score = r2
                    best_model = pipeline
        
            # Create results DataFrame
            results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
        
            if model_type == "Auto Select Best":
                st.success(f"Best model selected: {results_df.iloc[0]['Model']}")
                self.model = best_model
                self.baseline_metrics = {
                    'MAE': results_df.iloc[0]['MAE'],
                    'RMSE': results_df.iloc[0]['RMSE'],
                    'R2': results_df.iloc[0]['R2']
                }
            else:
                # For single model selection, use the last trained pipeline
                self.model = pipeline
                self.baseline_metrics = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2
                }
        
            # Calculate feature importance
            self._calculate_feature_importance(self.model)
        
            # Train fraud detection model
            self._train_fraud_model(X, y)
        
            # Log model performance
            self.monitor.log_performance(
                model_type,
                {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            )
        
            # Set training date
            self.training_date = datetime.now()
        
            return results_df
        
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            st.error(f"Training failed: {str(e)}")
            return None
    
    def _train_fraud_model(self, X, y):
        """Train isolation forest for fraud detection"""
        try:
            # Use preprocessor to transform data
            X_transformed = self.preprocessor.transform(X)
            
            # Train isolation forest
            self.fraud_model = IsolationForest(
                n_estimators=100,
                contamination=0.01,  # Assume 1% fraud
                random_state=42
            )
            self.fraud_model.fit(X_transformed)
            
            # Set dynamic threshold based on claim amounts
            scores = self.fraud_model.decision_function(X_transformed)
            self.fraud_threshold = np.percentile(scores, 1)  # Flag bottom 1% as potential fraud
            
        except Exception as e:
            self.logger.warning(f"Fraud model training failed: {str(e)}")
            self.fraud_model = None
    
    def _calculate_feature_importance(self, pipeline):
        """Calculate and store feature importance"""
        try:
            # Get feature names from the preprocessor
            if hasattr(pipeline.named_steps['preprocessor'], 'transformers_'):
                num_features = []
                cat_features = []
                
                for name, trans, features in pipeline.named_steps['preprocessor'].transformers_:
                    if name == 'num':
                        num_features = features
                    elif name == 'cat':
                        if hasattr(trans, 'get_feature_names_out'):
                            cat_features = trans.get_feature_names_out(features)
                        else:
                            cat_features = features
                
                all_features = np.concatenate([num_features, cat_features])
            else:
                all_features = pipeline.named_steps['preprocessor'].get_feature_names_out()
            
            # Get importance scores
            if hasattr(pipeline.named_steps['regressor'], 'feature_importances_'):
                importances = pipeline.named_steps['regressor'].feature_importances_
            elif hasattr(pipeline.named_steps['regressor'], 'coef_'):
                importances = np.abs(pipeline.named_steps['regressor'].coef_)
            else:
                importances = np.ones(len(all_features)) / len(all_features)
            
            self.feature_importance = pd.DataFrame({
                'Feature': all_features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
        except Exception as e:
            self.logger.warning(f"Could not calculate feature importance: {str(e)}")
            self.feature_importance = None

    def predict_claim_amount(self, input_data):
        """Robust claim amount prediction with input validation
        Supports both individual and group predictions
        """
        try:
            # Convert single record to DataFrame if needed
            if not isinstance(input_data, pd.DataFrame):
                input_data = pd.DataFrame([input_data])
            
            # Store original index for merging results later
            original_index = input_data.index
            
            # Validate required columns
            missing_cols = set(self.required_prediction_columns) - set(input_data.columns)
            if missing_cols:
                # Fill missing columns with sensible defaults
                defaults = {
                    'Claim_Weekday': datetime.now().strftime('%A'),
                    'Claim_Month': datetime.now().strftime('%B'),
                    'Employer': 'Unknown',
                    'Department': 'Unknown',
                    'Age_Group': '35-45',
                    'Claim_Size': 'Medium',
                    'Tenure_Group': '1-5yrs'
                }
                
                for col in missing_cols:
                    if col.endswith('_Utilization'):
                        input_data[col] = 0.5
                    elif col in defaults:
                        input_data[col] = defaults[col]
                    else:
                        input_data[col] = 0
                
                self.logger.warning(f"Filled missing columns: {missing_cols}")
            
            # Ensure all required columns are present and in correct order
            input_data = input_data.reindex(columns=self.required_prediction_columns, fill_value=0)
            
            # Convert numeric columns to float to avoid type issues
            numeric_cols = [col for col in input_data.columns if input_data[col].dtype in ['int64', 'float64']]
            for col in numeric_cols:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
            
            # Convert categorical columns to string
            categorical_cols = [col for col in input_data.columns if input_data[col].dtype == 'object']
            for col in categorical_cols:
                input_data[col] = input_data[col].astype(str)
            
            # Make predictions
            predictions = self.model.predict(input_data)
            
            # Add fraud detection
            if self.fraud_model:
                X_transformed = self.preprocessor.transform(input_data)
                fraud_scores = self.fraud_model.decision_function(X_transformed)
                fraud_flag = fraud_scores < self.fraud_threshold
            else:
                fraud_flag = np.zeros(len(predictions), dtype=bool)
            
            # Return format depends on input type
            if len(predictions) == 1:
                result = {
                    'prediction': float(predictions[0]),
                    'is_potential_fraud': bool(fraud_flag[0]),
                    'fraud_confidence': float(1 - fraud_scores[0]) if self.fraud_model else None
                }
                return result
            else:
                # For group predictions, return DataFrame with predictions
                result = input_data.copy()
                result['Predicted_Claim_Amount'] = predictions
                result['Is_Potential_Fraud'] = fraud_flag
                if self.fraud_model:
                    result['Fraud_Confidence'] = 1 - fraud_scores
                result.index = original_index  # Maintain original indexing
                return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            st.error(f"Prediction failed: {str(e)}")
            return None

    def analyze_category_shift(self, employee_id=None, group_filter=None):
        """Comprehensive benefit category impact analysis for individuals or groups"""
        try:
            if employee_id is not None:
                if employee_id not in self.clean_data['Employee_ID'].values:
                    st.error(f"Employee {employee_id} not found in data")
                    return None
                    
                # Get employee's current data
                emp_data = self.clean_data[self.clean_data['Employee_ID'] == employee_id].iloc[0].copy()
                
                # Prepare results container
                results = []
                
                # Test each category
                for category in self.category_order:
                    temp_data = emp_data.copy()
                    temp_data['Category'] = category
                    
                    # Adjust caps based on category
                    if category == 'Silver':
                        caps = [750000, 75000, 40000, 40000, 50000]
                    elif category == 'Gold':
                        caps = [1000000, 100000, 50000, 50000, 75000]
                    elif category == 'Platinum':
                        caps = [1500000, 150000, 75000, 75000, 100000]
                    
                    cap_cols = ['Inpatient_Cap_KES', 'Outpatient_Cap_KES', 
                               'Optical_Cap_KES', 'Dental_Cap_KES', 'Maternity_Cap_KES']
                    for col, cap in zip(cap_cols, caps):
                        temp_data[col] = cap
                        temp_data[f'{col}_Utilization'] = temp_data['Claim_Amount_KES'] / cap
                    
                    # Predict claim amount
                    prediction_result = self.predict_claim_amount(temp_data.to_frame().T)
                    
                    if prediction_result is not None:
                        results.append({
                            'Category': category,
                            'Predicted_Claim': prediction_result['prediction'],
                            'Is_Potential_Fraud': prediction_result['is_potential_fraud'],
                            'Fraud_Confidence': prediction_result['fraud_confidence'],
                            'Inpatient_Cap': caps[0],
                            'Outpatient_Cap': caps[1],
                            'Optical_Cap': caps[2],
                            'Dental_Cap': caps[3],
                            'Maternity_Cap': caps[4]
                        })
                
                return pd.DataFrame(results)
            
            elif group_filter is not None:
                return self.analyze_group_category_shift(group_filter)
            
            else:
                raise ValueError("Must provide either employee_id or group_filter")
            
        except Exception as e:
            self.logger.error(f"Category shift analysis failed: {str(e)}")
            st.error(f"Category shift analysis failed: {str(e)}")
            return None
    
    def analyze_group_category_shift(self, group_filter):
        """Analyze impact of benefit category changes for entire groups"""
        try:
            # Apply filters
            filtered_data = self.clean_data.copy()
            
            if group_filter.get('employers'):
                filtered_data = filtered_data[filtered_data['Employer'].isin(group_filter['employers'])]
            
            if group_filter.get('departments'):
                filtered_data = filtered_data[filtered_data['Department'].isin(group_filter['departments'])]
            
            if group_filter.get('categories'):
                filtered_data = filtered_data[filtered_data['Category'].isin(group_filter['categories'])]
            
            # Prepare results container
            results = []
            
            # Test each category scenario
            for category in self.category_order:
                temp_data = filtered_data.copy()
                temp_data['Category'] = category
                
                # Adjust caps based on category
                if category == 'Silver':
                    caps = [750000, 75000, 40000, 40000, 50000]
                elif category == 'Gold':
                    caps = [1000000, 100000, 50000, 50000, 75000]
                elif category == 'Platinum':
                    caps = [1500000, 150000, 75000, 75000, 100000]
                
                cap_cols = ['Inpatient_Cap_KES', 'Outpatient_Cap_KES', 
                           'Optical_Cap_KES', 'Dental_Cap_KES', 'Maternity_Cap_KES']
                for col, cap in zip(cap_cols, caps):
                    temp_data[col] = cap
                    temp_data[f'{col}_Utilization'] = temp_data['Claim_Amount_KES'] / cap
                
                # Predict for all employees in group
                predictions = self.predict_claim_amount(temp_data)
                
                if predictions is not None:
                    group_stats = {
                        'Category': category,
                        'Avg_Predicted_Claim': predictions['Predicted_Claim_Amount'].mean(),
                        'Total_Predicted_Claims': predictions['Predicted_Claim_Amount'].sum(),
                        'Potential_Fraud_Count': predictions['Is_Potential_Fraud'].sum(),
                        'Potential_Fraud_Amount': predictions.loc[predictions['Is_Potential_Fraud'], 'Predicted_Claim_Amount'].sum(),
                        'Inpatient_Cap': caps[0],
                        'Outpatient_Cap': caps[1],
                        'Optical_Cap': caps[2],
                        'Dental_Cap': caps[3],
                        'Maternity_Cap': caps[4]
                    }
                    results.append(group_stats)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            self.logger.error(f"Group category analysis failed: {str(e)}")
            st.error(f"Group category analysis failed: {str(e)}")
            return None

    def predict_group_claims(self, group_data):
        """Specialized method for group predictions with additional analytics"""
        try:
            if not isinstance(group_data, pd.DataFrame):
                raise ValueError("Group predictions require a DataFrame input")
            
            # Make predictions
            predictions = self.predict_claim_amount(group_data)
            
            if predictions is None:
                return None, None
                
            # Add group analytics
            predictions['Prediction_Group'] = predictions['Predicted_Claim_Amount'].apply(
                lambda x: 'High' if x > 100000 else ('Medium' if x > 50000 else 'Low'))
            
            # Calculate group statistics
            stats = {
                'Total_Predicted': predictions['Predicted_Claim_Amount'].sum(),
                'Average_Predicted': predictions['Predicted_Claim_Amount'].mean(),
                'Count_High': (predictions['Prediction_Group'] == 'High').sum(),
                'Count_Medium': (predictions['Prediction_Group'] == 'Medium').sum(),
                'Count_Low': (predictions['Prediction_Group'] == 'Low').sum(),
                'Potential_Fraud_Count': predictions['Is_Potential_Fraud'].sum(),
                'Potential_Fraud_Amount': predictions.loc[predictions['Is_Potential_Fraud'], 'Predicted_Claim_Amount'].sum()
            }
            
            return predictions, stats
            
        except Exception as e:
            self.logger.error(f"Group prediction failed: {str(e)}")
            st.error(f"Group prediction failed: {str(e)}")
            return None, None

    def visualize_group_predictions(self, predictions, stats):
        """Visualizations for group prediction results"""
        try:
            if predictions is None or stats is None:
                return
                
            # Summary metrics
            cols = st.columns(3)
            with cols[0]:
                st.metric("Total Predicted Claims", f"KES {stats['Total_Predicted']:,.2f}")
            with cols[1]:
                st.metric("Average Claim", f"KES {stats['Average_Predicted']:,.2f}")
            with cols[2]:
                st.metric("High Risk Claims", stats['Count_High'])
            
            # Fraud alert
            if stats['Potential_Fraud_Count'] > 0:
                st.warning(f"⚠️ Potential fraud detected in {stats['Potential_Fraud_Count']} claims totaling KES {stats['Potential_Fraud_Amount']:,.2f}")
            
            # Distribution chart
            chart1 = alt.Chart(predictions).mark_bar().encode(
                x='Prediction_Group',
                y='count()',
                color='Prediction_Group',
                tooltip=['count()']
            ).properties(
                title='Claim Risk Distribution'
            )
            
            # Amount distribution
            chart2 = alt.Chart(predictions).transform_density(
                'Predicted_Claim_Amount',
                as_=['Predicted_Claim_Amount', 'density'],
            ).mark_area().encode(
                x='Predicted_Claim_Amount:Q',
                y='density:Q',
                tooltip=['Predicted_Claim_Amount', 'density']
            ).properties(
                title='Predicted Amount Distribution'
            )
            
            st.altair_chart(alt.hconcat(chart1, chart2))
            
            # Top 10 highest claims
            st.write("### Top 10 Highest Predicted Claims")
            top_claims = predictions.nlargest(10, 'Predicted_Claim_Amount')
            st.dataframe(top_claims.style.format({
                'Predicted_Claim_Amount': '{:,.2f}'
            }))
            
            # SHAP explanations for top claims
            if st.checkbox("Show explanations for top claims"):
                sample = predictions.nlargest(1, 'Predicted_Claim_Amount')
                self.explain_prediction(sample)
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}")
            st.error(f"Visualization failed: {str(e)}")

    def visualize_category_impact(self, impact_df, is_group=False):
        """Enhanced visualizations for category impact for individuals or groups"""
        try:
            if impact_df is None or impact_df.empty:
                return
                
            if is_group:
                # Group visualization
                st.subheader("Group Benefit Category Impact Analysis")
                
                # Main metrics comparison
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Highest Average Claim", 
                             f"KES {impact_df['Avg_Predicted_Claim'].max():,.2f}",
                             impact_df.loc[impact_df['Avg_Predicted_Claim'].idxmax(), 'Category'])
                
                with cols[1]:
                    st.metric("Lowest Total Cost", 
                             f"KES {impact_df['Total_Predicted_Claims'].min():,.2f}",
                             impact_df.loc[impact_df['Total_Predicted_Claims'].idxmin(), 'Category'])
                
                with cols[2]:
                    st.metric("Best Fraud Prevention", 
                             f"{impact_df['Potential_Fraud_Count'].min()} cases",
                             impact_df.loc[impact_df['Potential_Fraud_Count'].idxmin(), 'Category'])
                
                # Cost comparison chart
                cost_chart = alt.Chart(impact_df).mark_bar().encode(
                    x=alt.X('Category', sort=self.category_order),
                    y='Total_Predicted_Claims',
                    color='Category',
                    tooltip=['Category', 'Total_Predicted_Claims']
                ).properties(
                    title='Total Predicted Claims by Benefit Category',
                    width=600
                )
                
                # Fraud risk comparison
                fraud_chart = alt.Chart(impact_df).mark_bar().encode(
                    x=alt.X('Category', sort=self.category_order),
                    y='Potential_Fraud_Count',
                    color='Category',
                    tooltip=['Category', 'Potential_Fraud_Count']
                ).properties(
                    title='Potential Fraud Cases by Benefit Category',
                    width=600
                )
                
                st.altair_chart(alt.vconcat(cost_chart, fraud_chart))
                
            else:
                # Individual visualization
                chart1 = alt.Chart(impact_df).mark_bar().encode(
                    x=alt.X('Category', sort=self.category_order),
                    y='Predicted_Claim',
                    color='Category',
                    tooltip=['Category', 'Predicted_Claim']
                ).properties(
                    title='Predicted Claim Amount by Benefit Category',
                    width=600
                )
                
                st.altair_chart(chart1, use_container_width=True)
                
                # Cap comparison
                cap_cols = ['Inpatient_Cap', 'Outpatient_Cap', 'Dental_Cap', 'Optical_Cap', 'Maternity_Cap']
                cap_df = impact_df.melt(id_vars=['Category'], 
                                      value_vars=cap_cols,
                                      var_name='Cap_Type', 
                                      value_name='Amount')
                
                chart2 = alt.Chart(cap_df).mark_bar().encode(
                    x='Cap_Type',
                    y='Amount',
                    color='Category',
                    column='Category',
                    tooltip=['Cap_Type', 'Amount', 'Category']
                ).properties(
                    title='Benefit Caps by Category',
                    width=150
                )
                
                st.altair_chart(chart2)
            
            # Fraud risk visualization
            if 'Is_Potential_Fraud' in impact_df.columns:
                st.write("### Fraud Risk by Benefit Category")
                fraud_chart = alt.Chart(impact_df).mark_bar().encode(
                    x='Category',
                    y='Fraud_Confidence',
                    color=alt.condition(
                        alt.datum.Is_Potential_Fraud,
                        alt.value('red'),
                        alt.value('steelblue')
                    ),
                    tooltip=['Category', 'Fraud_Confidence', 'Is_Potential_Fraud']
                ).properties(
                    title='Fraud Risk by Benefit Category'
                )
                st.altair_chart(fraud_chart)
            
            # Detailed comparison
            st.subheader("Detailed Comparison")
            
            if is_group:
                display_df = impact_df.copy()
                display_df['Avg_Predicted_Claim'] = display_df['Avg_Predicted_Claim'].apply(lambda x: f"KES {x:,.2f}")
                display_df['Total_Predicted_Claims'] = display_df['Total_Predicted_Claims'].apply(lambda x: f"KES {x:,.2f}")
                st.dataframe(display_df.set_index('Category'))
            else:
                # Calculate differences from current
                current_row = impact_df.iloc[0]
                impact_df['Change_From_Current'] = impact_df['Predicted_Claim'] - current_row['Predicted_Claim']
                impact_df['Pct_Change'] = (impact_df['Predicted_Claim'] / current_row['Predicted_Claim'] - 1) * 100
                
                # Format for display
                display_df = impact_df.copy()
                display_df['Predicted_Claim'] = display_df['Predicted_Claim'].apply(lambda x: f"KES {x:,.2f}")
                display_df['Change_From_Current'] = display_df['Change_From_Current'].apply(
                    lambda x: f"KES {x:+,.2f}")
                display_df['Pct_Change'] = display_df['Pct_Change'].apply(
                    lambda x: f"{x:+.1f}%")
                
                st.dataframe(display_df.set_index('Category'))
            
        except Exception as e:
            self.logger.error(f"Visualization error: {str(e)}")
            st.error(f"Visualization error: {str(e)}")

    def create_claim_distribution_chart(self, group_filter=None):
        """Enhanced claim distribution visualization with group filtering"""
        if self.clean_data is None or 'Claim_Amount_KES' not in self.clean_data.columns:
            return None
            
        # Apply group filters if provided
        if group_filter:
            chart_data = self.clean_data.copy()
            if group_filter.get('employers'):
                chart_data = chart_data[chart_data['Employer'].isin(group_filter['employers'])]
            if group_filter.get('departments'):
                chart_data = chart_data[chart_data['Department'].isin(group_filter['departments'])]
            if group_filter.get('categories'):
                chart_data = chart_data[chart_data['Category'].isin(group_filter['categories'])]
        else:
            chart_data = self.clean_data
            
        chart = alt.Chart(chart_data).transform_density(
            'Claim_Amount_KES',
            as_=['Claim_Amount_KES', 'density'],
            groupby=['Category']
        ).mark_area(opacity=0.5).encode(
            x='Claim_Amount_KES:Q',
            y='density:Q',
            color='Category:N'
        ).properties(
            title='Distribution of Claim Amounts by Category'
        )
        return chart
    
    def create_category_comparison_chart(self, group_filter=None):
        """Enhanced category comparison visualization with group filtering"""
        if self.clean_data is None or 'Category' not in self.clean_data.columns:
            return None
            
        # Apply group filters if provided
        if group_filter:
            chart_data = self.clean_data.copy()
            if group_filter.get('employers'):
                chart_data = chart_data[chart_data['Employer'].isin(group_filter['employers'])]
            if group_filter.get('departments'):
                chart_data = chart_data[chart_data['Department'].isin(group_filter['departments'])]
            if group_filter.get('categories'):
                chart_data = chart_data[chart_data['Category'].isin(group_filter['categories'])]
        else:
            chart_data = self.clean_data
            
        chart = alt.Chart(chart_data).mark_boxplot().encode(
            x='Category',
            y='Claim_Amount_KES',
            color='Category',
            tooltip=['Category', 'median(Claim_Amount_KES)']
        ).properties(
            title='Claim Amount Distribution by Benefit Category'
        )
        return chart
    
    def create_provider_analysis_chart(self, group_filter=None):
        """Enhanced provider analysis visualization with group filtering"""
        if self.clean_data is None or 'Provider_Name' not in self.clean_data.columns:
            return None
            
        # Apply group filters if provided
        if group_filter:
            chart_data = self.clean_data.copy()
            if group_filter.get('employers'):
                chart_data = chart_data[chart_data['Employer'].isin(group_filter['employers'])]
            if group_filter.get('departments'):
                chart_data = chart_data[chart_data['Department'].isin(group_filter['departments'])]
            if group_filter.get('categories'):
                chart_data = chart_data[chart_data['Category'].isin(group_filter['categories'])]
        else:
            chart_data = self.clean_data
            
        provider_stats = chart_data.groupby('Provider_Name').agg({
            'Claim_Amount_KES': ['mean', 'sum', 'count'],
            'Employee_Age': 'mean'
        }).nlargest(10, ('Claim_Amount_KES', 'sum')).reset_index()
        
        provider_stats.columns = ['Provider', 'Mean_Claim', 'Total_Claims', 'Claim_Count', 'Avg_Age']
        provider_stats['Efficiency'] = provider_stats['Mean_Claim'] / provider_stats['Avg_Age']
        
        chart = alt.Chart(provider_stats).mark_circle().encode(
            x='Claim_Count',
            y='Mean_Claim',
            size='Total_Claims',
            color='Efficiency',
            tooltip=['Provider', 'Mean_Claim', 'Total_Claims', 'Claim_Count', 'Efficiency']
        ).properties(
            title='Provider Cost Efficiency Analysis (Size = Total Claims)'
        )
        
        return chart
    
    def create_time_series_chart(self, group_filter=None):
        """Enhanced time series visualization with group filtering"""
        if self.clean_data is None or 'Submission_Date' not in self.clean_data.columns:
            return None
            
        # Apply group filters if provided
        if group_filter:
            chart_data = self.clean_data.copy()
            if group_filter.get('employers'):
                chart_data = chart_data[chart_data['Employer'].isin(group_filter['employers'])]
            if group_filter.get('departments'):
                chart_data = chart_data[chart_data['Department'].isin(group_filter['departments'])]
            if group_filter.get('categories'):
                chart_data = chart_data[chart_data['Category'].isin(group_filter['categories'])]
        else:
            chart_data = self.clean_data
            
        time_df = chart_data.set_index('Submission_Date').resample('M').agg({
            'Claim_Amount_KES': ['sum', 'count', 'mean', 'std']
        }).reset_index()
        
        time_df.columns = ['Date', 'Total', 'Count', 'Mean', 'Std']
        time_df['CI_lower'] = time_df['Mean'] - 1.96*time_df['Std']/np.sqrt(time_df['Count'])
        time_df['CI_upper'] = time_df['Mean'] + 1.96*time_df['Std']/np.sqrt(time_df['Count'])
        
        base = alt.Chart(time_df).encode(x='Date:T')
        line = base.mark_line(color='blue').encode(y='Mean')
        band = base.mark_area(opacity=0.3).encode(y='CI_lower', y2='CI_upper')
        
        chart = alt.layer(line, band).resolve_scale(
            y='independent'
        ).properties(
            title='Monthly Claims Trend with 95% Confidence Interval'
        )
        
        return chart

    def _identify_trend(self, series):
        """Identify trend direction and magnitude in a time series"""
        if len(series) < 2:
            return {'direction': 'insufficient data', 'percentage': 0}
        
        # Calculate percentage change from first to last value
        pct_change = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] * 100
        
        # Determine direction
        if abs(pct_change) < 5:
            direction = 'stable'
        elif pct_change > 0:
            direction = 'upward'
        else:
            direction = 'downward'
            
        return {'direction': direction, 'percentage': abs(pct_change)}

    def _filter_data(self, group_filter):
        """Filter data based on group filter criteria"""
        if not group_filter or self.clean_data is None:
            return self.clean_data
            
        filtered = self.clean_data.copy()
        
        if group_filter.get('employers'):
            filtered = filtered[filtered['Employer'].isin(group_filter['employers'])]
            
        if group_filter.get('departments'):
            filtered = filtered[filtered['Department'].isin(group_filter['departments'])]
            
        if group_filter.get('categories'):
            filtered = filtered[filtered['Category'].isin(group_filter['categories'])]
            
        return filtered

    def generate_executive_summary(self, group_filter=None):
        """Generate an enhanced executive summary with key metrics and insights"""
        data = self._filter_data(group_filter)
        
        summary = {
            "time_period": {
                "start": data['Submission_Date'].min().strftime('%Y-%m-%d'),
                "end": data['Submission_Date'].max().strftime('%Y-%m-%d')
            },
            "total_claims": len(data),
            "unique_members": data['Employee_ID'].nunique(),
            "total_claim_amount": data['Claim_Amount_KES'].sum(),
            "avg_claim_amount": data['Claim_Amount_KES'].mean(),
            "top_providers": data.groupby('Provider_Name')['Claim_Amount_KES']
                              .sum().nlargest(3).to_dict(),
            "category_distribution": data['Category'].value_counts(normalize=True).to_dict(),
            "model_performance": self.baseline_metrics,
            "key_insights": self._generate_insights(data)
        }
        
        return summary

    def _generate_insights(self, data):
        """Generate automated insights based on data patterns"""
        insights = []
        
        # Claim amount trends
        if 'Submission_Date' in data.columns and 'Claim_Amount_KES' in data.columns:
            monthly = data.set_index('Submission_Date').resample('M')['Claim_Amount_KES'].sum()
            trend = self._identify_trend(monthly)
            if trend['direction'] != 'stable':
                insights.append(
                    f"Claims show a {trend['direction']} trend ({trend['percentage']:.1f}% change) "
                    f"from {monthly.index[0].strftime('%b %Y')} to {monthly.index[-1].strftime('%b %Y')}"
                )
        
        # High-cost providers
        if 'Provider_Name' in data.columns and 'Claim_Amount_KES' in data.columns and 'Employee_Age' in data.columns:
            provider_eff = data.groupby('Provider_Name').agg({
                'Claim_Amount_KES': ['mean', 'count'],
                'Employee_Age': 'mean'
            })
            provider_eff.columns = ['avg_claim', 'claim_count', 'avg_age']
            provider_eff['efficiency'] = provider_eff['avg_claim'] / provider_eff['avg_age']
            
            inefficient = provider_eff[provider_eff['efficiency'] > provider_eff['efficiency'].quantile(0.75)]
            if len(inefficient) > 0:
                insights.append(
                    f"{len(inefficient)} providers show above-average costs relative to patient age: "
                    f"{', '.join(inefficient.index[:3])}{'...' if len(inefficient)>3 else ''}"
                )
        
        # Fraud alerts
        if self.fraud_model:
            try:
                X_transformed = self.preprocessor.transform(data)
                fraud_scores = self.fraud_model.decision_function(X_transformed)
                high_risk = (fraud_scores < self.fraud_threshold).sum()
                if high_risk > 0:
                    insights.append(
                        f"Potential fraud detected in {high_risk} claims ({high_risk/len(data):.1%} of total)"
                    )
            except Exception as e:
                self.logger.warning(f"Could not generate fraud insights: {str(e)}")
        
        return insights

    def create_enhanced_visualizations(self, group_filter=None):
        """Generate all visualizations with embedded insights"""
        data = self._filter_data(group_filter)
        visuals = []
        
        # 1. Time Series Analysis
        time_chart = self.create_time_series_chart(group_filter)
        if time_chart:
            time_insights = []
            if 'Submission_Date' in data.columns and 'Claim_Amount_KES' in data.columns:
                monthly = data.set_index('Submission_Date').resample('M')['Claim_Amount_KES'].sum()
                trend = self._identify_trend(monthly)
                time_insights.append(
                    f"Claims show a {trend['direction']} trend with {trend['percentage']:.1f}% change "
                    f"over the period"
                )
                
                # Seasonal patterns
                monthly = data.set_index('Submission_Date').resample('M').agg({
                    'Claim_Amount_KES': 'sum',
                    'Employee_ID': 'nunique'
                })
                monthly['claims_per_member'] = monthly['Claim_Amount_KES'] / monthly['Employee_ID']
                monthly['month'] = monthly.index.month
                seasonal = monthly.groupby('month')['claims_per_member'].mean()
                peak_month = seasonal.idxmax()
                time_insights.append(
                    f"Highest claim volumes typically occur in {datetime(2023, peak_month, 1).strftime('%B')}"
                )
            
            visuals.append(("claims_over_time", {
                "chart": time_chart,
                "insights": time_insights
            }))
        
        # 2. Category Comparison
        cat_chart = self.create_category_comparison_chart(group_filter)
        if cat_chart:
            cat_insights = []
            if 'Category' in data.columns and 'Claim_Amount_KES' in data.columns:
                cat_stats = data.groupby('Category')['Claim_Amount_KES'].agg(['mean', 'count'])
                highest_cat = cat_stats['mean'].idxmax()
                lowest_cat = cat_stats['mean'].idxmin()
                cat_insights.append(
                    f"Highest average claims in {highest_cat} category (KES {cat_stats.loc[highest_cat, 'mean']:,.2f})"
                )
                cat_insights.append(
                    f"Lowest average claims in {lowest_cat} category (KES {cat_stats.loc[lowest_cat, 'mean']:,.2f})"
                )
            
            visuals.append(("category_comparison", {
                "chart": cat_chart,
                "insights": cat_insights
            }))
        
        # 3. Provider Network Analysis
        prov_chart = self.create_provider_analysis_chart(group_filter)
        if prov_chart:
            prov_insights = []
            if 'Provider_Name' in data.columns and 'Claim_Amount_KES' in data.columns:
                provider_stats = data.groupby('Provider_Name').agg({
                    'Claim_Amount_KES': ['sum', 'count', 'mean'],
                    'Employee_Age': 'mean'
                })
                provider_stats.columns = ['total', 'count', 'avg_claim', 'avg_age']
                provider_stats['efficiency'] = provider_stats['avg_claim'] / provider_stats['avg_age']
                
                least_efficient = provider_stats.nlargest(1, 'avg_claim')
                most_efficient = provider_stats.nsmallest(1, 'avg_claim')
                
                prov_insights.append(
                    f"Least efficient provider: {least_efficient.index[0]} "
                    f"(KES {least_efficient['avg_claim'].values[0]:,.2f} avg claim)"
                )
                prov_insights.append(
                    f"Most efficient provider: {most_efficient.index[0]} "
                    f"(KES {most_efficient['avg_claim'].values[0]:,.2f} avg claim)"
                )
            
            visuals.append(("provider_analysis", {
                "chart": prov_chart,
                "insights": prov_insights
            }))
        
        # 4. Diagnosis-Treatment Patterns
        if 'Diagnosis' in data.columns and 'Treatment' in data.columns:
            diag_treat = data.groupby(['Diagnosis_Group', 'Treatment_Type']).size().reset_index(name='Count')
            dx_chart = alt.Chart(diag_treat).mark_rect().encode(
                x='Treatment_Type:N',
                y='Diagnosis_Group:N',
                color='Count:Q',
                tooltip=['Diagnosis_Group', 'Treatment_Type', 'Count']
            ).properties(
                title='Diagnosis-Treatment Patterns',
                width=600,
                height=400
            )
            
            dx_insights = []
            common_patterns = diag_treat.nlargest(3, 'Count')
            for _, row in common_patterns.iterrows():
                dx_insights.append(
                    f"Most common: {row['Diagnosis_Group']} treated with {row['Treatment_Type']} "
                    f"({row['Count']} cases)"
                )
            
            visuals.append(("diagnosis_patterns", {
                "chart": dx_chart,
                "insights": dx_insights
            }))
        
        return visuals

    def generate_recommendations(self, summary):
        """Generate data-driven recommendations"""
        recommendations = []
        data = self._filter_data(summary.get('group_filter'))
        
        # Cost-saving opportunities
        if 'Provider_Name' in data.columns and 'Claim_Amount_KES' in data.columns:
            prov_stats = data.groupby('Provider_Name')['Claim_Amount_KES'].agg(['mean', 'count'])
            high_cost = prov_stats[prov_stats['mean'] > prov_stats['mean'].quantile(0.75)]
            if len(high_cost) > 0:
                rec = (
                    "Consider renegotiating contracts with high-cost providers: "
                    f"{', '.join(high_cost.nlargest(3, 'mean').index.tolist())}. "
                    f"Potential savings up to {(high_cost['mean'].mean() - prov_stats['mean'].mean()) * high_cost['count'].sum():,.2f} KES annually"
                )
                recommendations.append(rec)
        
        # Benefit category optimization
        cat_impact = self.analyze_category_shift(group_filter=summary.get('group_filter'))
        if cat_impact is not None:
            best_cat = cat_impact.loc[cat_impact['Total_Predicted_Claims'].idxmin()]
            current_cat = cat_impact.iloc[0]
            if best_cat['Category'] != current_cat['Category']:
                savings = current_cat['Total_Predicted_Claims'] - best_cat['Total_Predicted_Claims']
                rec = (
                    f"Consider changing benefit category from {current_cat['Category']} to {best_cat['Category']}. "
                    f"Projected annual savings: {savings:,.2f} KES"
                )
                recommendations.append(rec)
        
        # Fraud prevention
        if self.fraud_model and any('fraud' in insight.lower() for insight in summary.get('key_insights', [])):
            rec = (
                "Implement enhanced fraud detection measures focusing on: "
                "duplicate claims, unusually high-cost procedures, and "
                "providers with abnormal billing patterns"
            )
            recommendations.append(rec)
        
        # Utilization management
        if 'Inpatient_Cap_KES_Utilization' in data.columns:
            high_util = data[data['Inpatient_Cap_KES_Utilization'] > 1]
            if len(high_util) > 0:
                rec = (
                    f"{len(high_util)} claims exceeded inpatient caps. "
                    "Consider implementing utilization review for high-cost inpatient services"
                )
                recommendations.append(rec)
        
        # Standard recommendations
        standard_recs = [
            "Review benefit design to align with actual utilization patterns",
            "Implement member education programs for high-utilization members",
            "Conduct provider performance reviews quarterly",
            "Evaluate network adequacy based on geographic distribution of claims"
        ]
        recommendations.extend(standard_recs)
        
        return recommendations

    def generate_report(self, employee_id=None, group_filter=None, format='HTML'):
        """Generate comprehensive report in HTML or PDF format for individuals or groups"""
        try:
            if self.clean_data is None or self.model is None:
                st.warning("Please clean data and train model first")
                return
                
            with st.spinner("Generating report..."):
                if format == 'PDF':
                    return self._generate_pdf_report(employee_id, group_filter)
                else:
                    return self._generate_html_report(employee_id, group_filter)
                    
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            st.error(f"Report generation failed: {str(e)}")
    
    def _generate_html_report(self, employee_id=None, group_filter=None):
        """Generate interactive HTML report with all visualizations"""
        report = []
        
        # 1. Cover Page
        report.append("<div class='report-header'>")
        report.append("<h1>Healthcare Claims Analysis Report</h1>")
        report.append(f"<p class='report-meta'>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        report.append(f"<p class='report-meta'>Model Trained on: {self.training_date.strftime('%Y-%m-%d') if self.training_date else 'N/A'}</p>")
        
        if employee_id:
            report.append(f"<p class='report-meta'>Employee ID: {employee_id}</p>")
        elif group_filter:
            report.append("<div class='group-info'>")
            report.append("<h3>Group Analysis Scope:</h3>")
            if group_filter.get('employers'):
                report.append(f"<p>Employers: {', '.join(group_filter['employers'])}</p>")
            if group_filter.get('departments'):
                report.append(f"<p>Departments: {', '.join(group_filter['departments'])}</p>")
            if group_filter.get('categories'):
                report.append(f"<p>Benefit Categories: {', '.join(group_filter['categories'])}</p>")
            report.append("</div>")
        
        report.append("</div>")
        report.append("<hr class='report-divider'>")
        
        # 2. Executive Summary
        summary = self.generate_executive_summary(group_filter)
        report.append("<section class='executive-summary'>")
        report.append("<h2>Executive Summary</h2>")
        
        # Key Metrics
        report.append("<div class='metrics-grid'>")
        report.append(f"<div class='metric-card'><h3>Time Period</h3><p>{summary['time_period']['start']} to {summary['time_period']['end']}</p></div>")
        report.append(f"<div class='metric-card'><h3>Total Claims</h3><p>{summary['total_claims']:,}</p></div>")
        report.append(f"<div class='metric-card'><h3>Unique Members</h3><p>{summary['unique_members']:,}</p></div>")
        report.append(f"<div class='metric-card'><h3>Total Claim Amount</h3><p>KES {summary['total_claim_amount']:,.2f}</p></div>")
        report.append(f"<div class='metric-card'><h3>Average Claim</h3><p>KES {summary['avg_claim_amount']:,.2f}</p></div>")
        report.append(f"<div class='metric-card'><h3>Model R²</h3><p>{summary['model_performance']['R2']:.3f}</p></div>")
        report.append("</div>")
        
        # Key Insights
        report.append("<div class='key-insights'>")
        report.append("<h3>Key Insights</h3>")
        report.append("<ul>")
        for insight in summary['key_insights']:
            report.append(f"<li>{insight}</li>")
        report.append("</ul>")
        report.append("</div>")
        report.append("</section>")
        
        # 3. Detailed Analysis - All Visualizations
        report.append("<section class='detailed-analysis'>")
        report.append("<h2>Advanced Analysis</h2>")
        
        # Create tabs for different visualization categories
        report.append("""
        <div class='tabs'>
            <button class='tab-button active' onclick="openTab(event, 'claims-distribution')">Claims Distribution</button>
            <button class='tab-button' onclick="openTab(event, 'temporal-analysis')">Temporal Analysis</button>
            <button class='tab-button' onclick="openTab(event, 'provider-network')">Provider Network</button>
            <button class='tab-button' onclick="openTab(event, 'diagnostic-patterns')">Diagnostic Patterns</button>
        </div>
        """)
        
        # Tab 1: Claims Distribution
        report.append("""
        <div id='claims-distribution' class='tab-content' style='display:block'>
            <h3>Claims Distribution</h3>
        """)
        
        # Claim amount distribution by category
        chart = self.create_claim_distribution_chart(group_filter)
        if chart:
            report.append(chart.to_html())
        
        # Category comparison chart
        cat_chart = self.create_category_comparison_chart(group_filter)
        if cat_chart:
            report.append(cat_chart.to_html())
        
        report.append("</div>")
        
        # Tab 2: Temporal Analysis
        report.append("""
        <div id='temporal-analysis' class='tab-content'>
            <h3>Temporal Analysis</h3>
        """)
        
        if 'Submission_Date' in self.clean_data.columns:
            # Monthly trends with confidence intervals
            time_chart = self.create_time_series_chart(group_filter)
            if time_chart:
                report.append(time_chart.to_html())
            
            # Claims by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = self.clean_data['Submission_Date'].dt.day_name().value_counts().reindex(day_order).reset_index()
            day_counts.columns = ['Day', 'Count']
            day_chart = alt.Chart(day_counts).mark_bar().encode(
                x=alt.X('Day', sort=day_order),
                y='Count',
                tooltip=['Day', 'Count']
            ).properties(title='Claims by Day of Week')
            report.append(day_chart.to_html())
        
        report.append("</div>")
        
        # Tab 3: Provider Network
        report.append("""
        <div id='provider-network' class='tab-content'>
            <h3>Provider Network Analysis</h3>
        """)
        
        provider_chart = self.create_provider_analysis_chart(group_filter)
        if provider_chart:
            report.append(provider_chart.to_html())
        
        report.append("</div>")
        
        # Tab 4: Diagnostic Patterns
        report.append("""
        <div id='diagnostic-patterns' class='tab-content'>
            <h3>Diagnosis-Treatment Patterns</h3>
        """)
        
        if 'Diagnosis' in self.clean_data.columns and 'Treatment' in self.clean_data.columns:
            diag_treat = self.clean_data.groupby(['Diagnosis_Group', 'Treatment_Type']).size().reset_index(name='Count')
            dx_chart = alt.Chart(diag_treat).mark_rect().encode(
                x='Treatment_Type:N',
                y='Diagnosis_Group:N',
                color='Count:Q',
                tooltip=['Diagnosis_Group', 'Treatment_Type', 'Count']
            ).properties(width=600, height=400)
            report.append(dx_chart.to_html())
        
        report.append("</div>")
        
        # Tab JavaScript
        report.append("""
        <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tabbuttons;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {
                tabbuttons[i].className = tabbuttons[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        </script>
        """)
        
        report.append("</section>")
        
        # 4. Recommendations
        report.append("<section class='recommendations'>")
        report.append("<h2>Actionable Recommendations</h2>")
        recommendations = self.generate_recommendations(summary)
        report.append("<ol>")
        for rec in recommendations:
            report.append(f"<li>{rec}</li>")
        report.append("</ol>")
        report.append("</section>")
        
        # 5. Appendices
        report.append("<section class='appendices'>")
        report.append("<h2>Appendices</h2>")
        report.append("<h3>Model Details</h3>")
        report.append("<table>")
        report.append("<tr><th>Metric</th><th>Value</th></tr>")
        for metric, value in self.baseline_metrics.items():
            report.append(f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>")
        report.append("</table>")
        
        if self.feature_importance is not None:
            report.append("<h3>Top 10 Features</h3>")
            report.append(self.feature_importance.head(10).to_html())
        report.append("</section>")
        
        # Combine all sections
        html_report = "\n".join(report)
        
        # Add CSS styling
        styled_report = f"""
        <html>
        <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .report-header {{ text-align: center; margin-bottom: 2rem; }}
            .report-meta {{ color: #666; }}
            .report-divider {{ border: 1px solid #eee; margin: 2rem 0; }}
            section {{ margin-bottom: 3rem; }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }}
            .metric-card {{
                background: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-card h3 {{ margin-top: 0; font-size: 1rem; }}
            .metric-card p {{ margin-bottom: 0; font-size: 1.2rem; font-weight: bold; }}
            .key-insights {{ background: #f0f7ff; padding: 1rem; border-radius: 8px; }}
            .visualization {{ margin-bottom: 3rem; }}
            .visualization-insights {{
                background: #f8f9fa;
                padding: 1rem;
                border-left: 4px solid #4285f4;
                margin-top: 1rem;
            }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            
            /* Tab styling */
            .tabs {{
                display: flex;
                border-bottom: 1px solid #ddd;
                margin-bottom: 1rem;
            }}
            .tab-button {{
                background-color: inherit;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 10px 16px;
                transition: 0.3s;
                font-size: 14px;
                border-bottom: 3px solid transparent;
            }}
            .tab-button:hover {{
                background-color: #f1f1f1;
            }}
            .tab-button.active {{
                border-bottom: 3px solid #4285f4;
                font-weight: bold;
            }}
            .tab-content {{
                display: none;
                padding: 6px 0;
            }}
        </style>
        </head>
        <body>
        {html_report}
        </body>
        </html>
        """
        
        # Display in app
        st.markdown(styled_report, unsafe_allow_html=True)
        
        # Download button
        st.download_button(
            label="Download Report as HTML",
            data=styled_report,
            file_name="healthcare_claims_report.html",
            mime="text/html"
        )
    
    def _generate_pdf_report(self, employee_id=None, group_filter=None):
        """Generate professional PDF report with all visualizations"""
        try:
            pdf = PDF()
            pdf.add_font('DejaVuSans', '', 'DejaVuSans.ttf', uni=True)
            pdf.add_font('DejaVuSans', 'B', 'DejaVuSans-Bold.ttf', uni=True)
            pdf.add_font('DejaVuSans', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
            pdf.alias_nb_pages()
            pdf.add_page()
            
            # 1. Cover Page
            pdf.set_font('DejaVuSans', 'B', 24)
            pdf.cell(0, 40, 'Healthcare Claims Analysis Report', 0, 1, 'C')
            pdf.set_font('DejaVuSans', '', 12)
            pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
            pdf.cell(0, 10, f"Model Trained on: {self.training_date.strftime('%Y-%m-%d') if self.training_date else 'N/A'}", 0, 1, 'C')
            
            if employee_id:
                pdf.cell(0, 10, f"Employee ID: {employee_id}", 0, 1, 'C')
            elif group_filter:
                pdf.cell(0, 10, "Group Analysis:", 0, 1, 'C')
                if group_filter.get('employers'):
                    pdf.cell(0, 10, f"Employers: {', '.join(group_filter['employers'])}", 0, 1, 'C')
                if group_filter.get('departments'):
                    pdf.cell(0, 10, f"Departments: {', '.join(group_filter['departments'])}", 0, 1, 'C')
                if group_filter.get('categories'):
                    pdf.cell(0, 10, f"Benefit Categories: {', '.join(group_filter['categories'])}", 0, 1, 'C')
            
            pdf.ln(20)
            
            # 2. Executive Summary
            summary = self.generate_executive_summary(group_filter)
            pdf.set_font('DejaVuSans', 'B', 16)
            pdf.cell(0, 10, 'Executive Summary', 0, 1)
            pdf.set_font('DejaVuSans', '', 12)
            
            # Key Metrics
            col_width = pdf.w / 3
            pdf.cell(col_width, 10, f"Time Period: {summary['time_period']['start']} to {summary['time_period']['end']}", 0, 0)
            pdf.cell(col_width, 10, f"Total Claims: {summary['total_claims']:,}", 0, 0)
            pdf.cell(col_width, 10, f"Unique Members: {summary['unique_members']:,}", 0, 1)
            
            pdf.cell(col_width, 10, f"Total Claim Amount: KES {summary['total_claim_amount']:,.2f}", 0, 0)
            pdf.cell(col_width, 10, f"Average Claim: KES {summary['avg_claim_amount']:,.2f}", 0, 0)
            pdf.cell(col_width, 10, f"Model R²: {summary['model_performance']['R2']:.3f}", 0, 1)
            
            pdf.ln(5)
            
            # Key Insights
            pdf.set_font('DejaVuSans', 'B', 12)
            pdf.cell(0, 10, 'Key Insights:', 0, 1)
            pdf.set_font('DejaVuSans', '', 10)
            for insight in summary['key_insights']:
                pdf.multi_cell(0, 8, f"• {insight}")
            
            pdf.ln(5)
            
            # 3. Detailed Analysis - All Visualizations
            pdf.add_page()
            pdf.set_font('DejaVuSans', 'B', 16)
            pdf.cell(0, 10, 'Advanced Analysis', 0, 1)
            
            # Section 1: Claims Distribution
            pdf.set_font('DejaVuSans', 'B', 14)
            pdf.cell(0, 10, 'Claims Distribution', 0, 1)
            
            # Claim amount distribution by category
            chart = self.create_claim_distribution_chart(group_filter)
            if chart:
                with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    chart.save(tmpfile.name)
                    pdf.image(tmpfile.name, x=10, w=190)
                    os.unlink(tmpfile.name)
            
            pdf.ln(5)
            
            # Category comparison chart
            cat_chart = self.create_category_comparison_chart(group_filter)
            if cat_chart:
                with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    cat_chart.save(tmpfile.name)
                    pdf.image(tmpfile.name, x=10, w=190)
                    os.unlink(tmpfile.name)
            
            pdf.ln(5)
            
            # Section 2: Temporal Analysis
            pdf.add_page()
            pdf.set_font('DejaVuSans', 'B', 14)
            pdf.cell(0, 10, 'Temporal Analysis', 0, 1)
            
            if 'Submission_Date' in self.clean_data.columns:
                # Monthly trends
                time_chart = self.create_time_series_chart(group_filter)
                if time_chart:
                    with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                        time_chart.save(tmpfile.name)
                        pdf.image(tmpfile.name, x=10, w=190)
                        os.unlink(tmpfile.name)
                
                pdf.ln(5)
                
                # Claims by day of week
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_counts = self.clean_data['Submission_Date'].dt.day_name().value_counts().reindex(day_order).reset_index()
                day_counts.columns = ['Day', 'Count']
                day_chart = alt.Chart(day_counts).mark_bar().encode(
                    x=alt.X('Day', sort=day_order),
                    y='Count',
                    tooltip=['Day', 'Count']
                ).properties(title='Claims by Day of Week')
                
                with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    day_chart.save(tmpfile.name)
                    pdf.image(tmpfile.name, x=10, w=190)
                    os.unlink(tmpfile.name)
            
            # Section 3: Provider Network
            pdf.add_page()
            pdf.set_font('DejaVuSans', 'B', 14)
            pdf.cell(0, 10, 'Provider Network Analysis', 0, 1)
            
            provider_chart = self.create_provider_analysis_chart(group_filter)
            if provider_chart:
                with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    provider_chart.save(tmpfile.name)
                    pdf.image(tmpfile.name, x=10, w=190)
                    os.unlink(tmpfile.name)
            
            # Section 4: Diagnostic Patterns
            pdf.add_page()
            pdf.set_font('DejaVuSans', 'B', 14)
            pdf.cell(0, 10, 'Diagnosis-Treatment Patterns', 0, 1)
            
            if 'Diagnosis' in self.clean_data.columns and 'Treatment' in self.clean_data.columns:
                diag_treat = self.clean_data.groupby(['Diagnosis_Group', 'Treatment_Type']).size().reset_index(name='Count')
                dx_chart = alt.Chart(diag_treat).mark_rect().encode(
                    x='Treatment_Type:N',
                    y='Diagnosis_Group:N',
                    color='Count:Q',
                    tooltip=['Diagnosis_Group', 'Treatment_Type', 'Count']
                ).properties(width=600, height=400)
                
                with NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    dx_chart.save(tmpfile.name)
                    pdf.image(tmpfile.name, x=10, w=190)
                    os.unlink(tmpfile.name)
            
            # 4. Recommendations
            pdf.add_page()
            pdf.set_font('DejaVuSans', 'B', 16)
            pdf.cell(0, 10, 'Actionable Recommendations', 0, 1)
            pdf.set_font('DejaVuSans', '', 12)
            
            recommendations = self.generate_recommendations(summary)
            for i, rec in enumerate(recommendations, 1):
                pdf.multi_cell(0, 8, f"{i}. {rec}")
                pdf.ln(2)
            
            # 5. Appendices
            pdf.add_page()
            pdf.set_font('DejaVuSans', 'B', 16)
            pdf.cell(0, 10, 'Appendices', 0, 1)
            
            # Model details
            pdf.set_font('DejaVuSans', 'B', 12)
            pdf.cell(0, 10, 'Model Details', 0, 1)
            pdf.set_font('DejaVuSans', '', 10)
            
            col_width = pdf.w / 2
            for metric, value in self.baseline_metrics.items():
                pdf.cell(col_width, 8, metric, 1)
                pdf.cell(col_width, 8, f"{value:.3f}", 1)
                pdf.ln()
            
            pdf.ln(5)
            
            # Feature importance
            if self.feature_importance is not None:
                pdf.set_font('DejaVuSans', 'B', 12)
                pdf.cell(0, 10, 'Top 10 Features', 0, 1)
                pdf.set_font('DejaVuSans', '', 8)
                
                # Create table
                col_width = pdf.w / 2
                pdf.cell(col_width, 8, 'Feature', 1)
                pdf.cell(col_width, 8, 'Importance', 1)
                pdf.ln()
                
                for _, row in self.feature_importance.head(10).iterrows():
                    pdf.cell(col_width, 8, row['Feature'], 1)
                    pdf.cell(col_width, 8, f"{row['Importance']:.3f}", 1)
                    pdf.ln()
            
            # Save to bytes
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            
            # Download button
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="healthcare_claims_report.pdf",
                mime="application/pdf"
            )
            
            return pdf_bytes
            
        except Exception as e:
            self.logger.error(f"PDF generation failed: {str(e)}")
            st.error(f"PDF report generation failed: {str(e)}")
            return None

    def save_snapshot(self, prefix=''):
        """Save current state for versioning"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create snapshots directory if not exists
            os.makedirs("snapshots", exist_ok=True)
            
            # Save data
            if self.clean_data is not None:
                self.clean_data.to_parquet(f"snapshots/{prefix}data_{timestamp}.parquet")
            
            # Save model
            if self.model is not None:
                joblib.dump(self.model, f"snapshots/{prefix}model_{timestamp}.joblib")
            
            # Save metadata
            metadata = {
                "timestamp": timestamp,
                "model_type": type(self.model).__name__ if self.model else None,
                "performance": self.baseline_metrics,
                "features": list(self.required_prediction_columns) if self.required_prediction_columns else None
            }
            
            with open(f"snapshots/{prefix}metadata_{timestamp}.json", 'w') as f:
                json.dump(metadata, f)
            
            self.logger.info(f"Saved snapshot with prefix {prefix}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {str(e)}")
            return False

    def generate_model_card(self):
        """Generate model documentation"""
        return {
            "model_type": type(self.model).__name__ if self.model else None,
            "training_date": self.training_date.strftime("%Y-%m-%d") if self.training_date else None,
            "performance": self.baseline_metrics,
            "features": list(self.required_prediction_columns) if self.required_prediction_columns else None,
            "data_statistics": self.clean_data.describe().to_dict() if self.clean_data is not None else None
        }

class ClaimInput(BaseModel):
    """Pydantic model for API input validation"""
    visit_type: str
    diagnosis: str
    treatment: str
    provider_name: str
    hospital_county: str
    employee_age: int
    employee_gender: str
    co_payment_kes: float
    pre_authorization_required: str
    category: str
    employer: str
    department: str

def create_api(predictor):
    """Create FastAPI endpoints for predictions"""
    app = FastAPI(
        title="Healthcare Claims Prediction API",
        description="API for predicting healthcare claim amounts",
        version="1.0.0"
    )
    
    @app.post("/predict", summary="Predict claim amount")
    async def predict_claim(claim: ClaimInput):
        try:
            input_data = claim.dict()
            input_data['Diagnosis_Group'] = input_data['diagnosis'].split()[0]
            input_data['Treatment_Type'] = input_data['treatment'].split()[0]
            input_data['Is_Pre_Authorized'] = 1 if input_data['pre_authorization_required'] == "Yes" else 0
            
            # Set caps based on category
            if input_data['category'] == 'Silver':
                caps = [750000, 75000, 40000, 40000, 50000]
            elif input_data['category'] == 'Gold':
                caps = [1000000, 100000, 50000, 50000, 75000]
            else:  # Platinum
                caps = [1500000, 150000, 75000, 75000, 100000]
                
            cap_cols = ['Inpatient_Cap_KES', 'Outpatient_Cap_KES', 
                       'Optical_Cap_KES', 'Dental_Cap_KES', 'Maternity_Cap_KES']
            for col, cap in zip(cap_cols, caps):
                input_data[col] = cap
                input_data[f'{col}_Utilization'] = 0.5
            
            # Add temporal features
            input_data['Claim_Weekday'] = datetime.now().strftime('%A')
            input_data['Claim_Month'] = datetime.now().strftime('%B')
            
            # Make prediction
            prediction = predictor.predict_claim_amount(input_data)
            
            if prediction is None:
                raise HTTPException(status_code=400, detail="Prediction failed")
            
            return prediction
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/predict_group", summary="Predict claims for a group")
    async def predict_group(claims: List[ClaimInput]):
        try:
            input_data = [claim.dict() for claim in claims]
            df = pd.DataFrame(input_data)
            
            # Process each claim
            for i, row in df.iterrows():
                row['Diagnosis_Group'] = row['diagnosis'].split()[0]
                row['Treatment_Type'] = row['treatment'].split()[0]
                row['Is_Pre_Authorized'] = 1 if row['pre_authorization_required'] == "Yes" else 0
                
                # Set caps based on category
                if row['category'] == 'Silver':
                    caps = [750000, 75000, 40000, 40000, 50000]
                elif row['category'] == 'Gold':
                    caps = [1000000, 100000, 50000, 50000, 75000]
                else:  # Platinum
                    caps = [1500000, 150000, 75000, 75000, 100000]
                    
                cap_cols = ['Inpatient_Cap_KES', 'Outpatient_Cap_KES', 
                           'Optical_Cap_KES', 'Dental_Cap_KES', 'Maternity_Cap_KES']
                for col, cap in zip(cap_cols, caps):
                    row[col] = cap
                    row[f'{col}_Utilization'] = 0.5
                
                # Add temporal features
                row['Claim_Weekday'] = datetime.now().strftime('%A')
                row['Claim_Month'] = datetime.now().strftime('%B')
            
            # Make predictions
            predictions = predictor.predict_claim_amount(df)
            
            if predictions is None:
                raise HTTPException(status_code=400, detail="Group prediction failed")
            
            return predictions.to_dict(orient='records')
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    return app

def render_data_upload():
    st.header("📤 Upload Claim Data")
    
    uploaded_file = st.file_uploader(
        "Choose your claims data file (CSV or Excel)",
        type=['csv', 'xlsx'],
        help="File should include claim details and employee information"
    )
    
    if uploaded_file is not None:
        with st.spinner('Processing data...'):
            df = st.session_state.predictor.load_data(uploaded_file)
        
        if df is not None:
            st.success("✅ Data loaded successfully!")
            
            # Data summary
            cols = st.columns(3)
            with cols[0]:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Total Claims", len(df))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Columns", len(df.columns))
                st.markdown("</div>", unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.metric("Unique Employees", df['Employee_ID'].nunique())
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Data preview
            with st.expander("Preview Raw Data"):
                st.dataframe(df.head())

def render_data_cleaning():
    st.header("🧹 Data Cleaning")
    
    if st.session_state.predictor.data is None:
        st.warning("Please upload data first")
        return
    
    if st.button("Clean Data"):
        success = st.session_state.predictor.clean_and_prepare_data()
        if success:
            st.success("✅ Data cleaned and prepared successfully!")
            
            # Show cleaning summary
            st.subheader("Cleaning Summary")
            
            # Missing values before/after
            st.write("**Missing Values Handling:**")
            before_missing = st.session_state.predictor.data.isnull().sum().sum()
            after_missing = st.session_state.predictor.clean_data.isnull().sum().sum()
            st.write(f"- Missing values before: {before_missing}")
            st.write(f"- Missing values after: {after_missing}")
            
            # New features added
            st.write("\n**New Features Created:**")
            original_cols = set(st.session_state.predictor.data.columns)
            new_cols = set(st.session_state.predictor.clean_data.columns) - original_cols
            for col in new_cols:
                st.write(f"- {col}")
            
            # Show cleaned data
            with st.expander("Preview Cleaned Data"):
                st.dataframe(st.session_state.predictor.clean_data.head())

def render_exploratory_analysis():
    st.header("🔍 Exploratory Analysis")
    
    if st.session_state.predictor.clean_data is None:
        st.warning("Please clean data first")
        return
    
    # Generate automatic profile report
    st.subheader("Automated Data Profile")
    st.session_state.predictor.generate_data_report()
    
    # Enhanced manual analysis sections
    st.subheader("Advanced Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Claims Distribution", 
        "Temporal Analysis", 
        "Provider Network", 
        "Diagnostic Patterns"
    ])
    
    df = st.session_state.predictor.clean_data
    
    with tab1:
        st.write("### Claim Amount Distribution by Category")
        chart = st.session_state.predictor.create_claim_distribution_chart()
        if chart:
            st.altair_chart(chart, use_container_width=True)
        
        # Interactive correlation matrix
        st.write("### Correlation Matrix")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr().stack().reset_index(name='correlation')
            chart = alt.Chart(corr).mark_rect().encode(
                x='level_0:N',
                y='level_1:N',
                color='correlation:Q',
                tooltip=['correlation']
            ).properties(width=600, height=500)
            st.altair_chart(chart, use_container_width=True)
    
    with tab2:
        st.write("### Claims by Day of Week")
        if 'Submission_Date' in df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['Submission_Date'].dt.day_name().value_counts().reindex(day_order).reset_index()
            day_counts.columns = ['Day', 'Count']
            
            chart = alt.Chart(day_counts).mark_bar().encode(
                x=alt.X('Day', sort=day_order),
                y='Count',
                tooltip=['Day', 'Count']
            )
            st.altair_chart(chart, use_container_width=True)
        
        st.write("### Monthly Trends with Confidence Intervals")
        if 'Submission_Date' in df.columns:
            monthly = df.set_index('Submission_Date').resample('M').agg({
                'Claim_Amount_KES': ['sum', 'count', 'mean', 'std']
            }).reset_index()
            monthly.columns = ['Date', 'Total', 'Count', 'Mean', 'Std']
            monthly['CI_lower'] = monthly['Mean'] - 1.96*monthly['Std']/np.sqrt(monthly['Count'])
            monthly['CI_upper'] = monthly['Mean'] + 1.96*monthly['Std']/np.sqrt(monthly['Count'])
            
            base = alt.Chart(monthly).encode(x='Date:T')
            line = base.mark_line(color='blue').encode(y='Mean')
            band = base.mark_area(opacity=0.3).encode(
                y='CI_lower',
                y2='CI_upper'
            )
            st.altair_chart(line + band, use_container_width=True)
    
    with tab3:
        st.write("### Provider Cost Efficiency Analysis")
        if 'Provider_Name' in df.columns and 'Claim_Amount_KES' in df.columns:
            # Calculate provider statistics
            provider_stats = df.groupby('Provider_Name').agg({
                'Claim_Amount_KES': ['mean', 'count', 'sum'],
                'Employee_Age': 'mean'
            }).reset_index()
        
            # Flatten multi-index columns
            provider_stats.columns = ['Provider', 'Avg_Claim', 'Claim_Count', 'Total_Claims', 'Avg_Age']
        
            # Calculate efficiency metric
            provider_stats['Efficiency'] = provider_stats['Avg_Claim'] / provider_stats['Avg_Age']
        
            # Calculate break-even point - use average claim amount if premium data isn't available
            if 'Premium_KES' in df.columns:
                total_premium = df['Premium_KES'].sum()
                total_members = df['Employee_ID'].nunique()
                break_even_claim = (total_premium / total_members) * 0.85  # 85% of premium per member
                break_even_source = "actuarial calculation (85% of premium per member)"
            else:
                break_even_claim = df['Claim_Amount_KES'].mean()
                break_even_source = "average claim amount"
        
            # Create the scatter plot with explicit data types
            scatter = alt.Chart(provider_stats).mark_circle(size=100).encode(
                x=alt.X('Claim_Count:Q', title='Number of Claims'),
                y=alt.Y('Avg_Claim:Q', title='Average Claim Amount (KES)'),
                size=alt.Size('Total_Claims:Q', title='Total Claims Amount'),
                color=alt.Color('Provider:N', legend=alt.Legend(title="Providers")),
                tooltip=[
                    alt.Tooltip('Provider:N', title='Provider'),
                    alt.Tooltip('Avg_Claim:Q', format='.2f', title='Avg Claim'),
                    alt.Tooltip('Claim_Count:Q', title='Claim Count'),
                    alt.Tooltip('Total_Claims:Q', format='.2f', title='Total Claims'),
                    alt.Tooltip('Efficiency:Q', format='.2f', title='Efficiency Score')
                ]
            ).properties(
                width=800,
                height=500,
                title=f'Provider Cost Efficiency Analysis (Break-even at KES {break_even_claim:,.2f} - {break_even_source})'
            )
        
            # Add break-even reference line
            rule = alt.Chart(pd.DataFrame({'y': [break_even_claim]})).mark_rule(
                color='red',
                strokeWidth=2,
                strokeDash=[5, 5]
            ).encode(
                y='y:Q'
            )
        
            # Create annotation data
            annotation_data = pd.DataFrame({
                'x': [10] * 2,
                'y': [break_even_claim * 0.9, break_even_claim * 1.1],
                'text': ['Profit Territory', 'Loss Territory']
            })
        
            # Add profit/loss annotations
            annotations = alt.Chart(annotation_data).mark_text(
                align='left',
                baseline='middle',
                dx=10,
                fontSize=12
            ).encode(
                x=alt.X('x:Q', title=None),
                y=alt.Y('y:Q', title=None),
                text=alt.Text('text:N')
            )
        
            # Combine all layers
            chart = (scatter + rule + annotations).configure_axis(
                labelFontSize=12,
                titleFontSize=14
            ).configure_legend(
                titleFontSize=14,
                labelFontSize=12
            )
        
            st.altair_chart(chart, use_container_width=True)
        
            # Add interpretation guide
            st.markdown(f"""
            **How to interpret this chart:**
            - Each bubble represents a healthcare provider
            - X-axis shows the number of claims from each provider
            - Y-axis shows the average claim amount
            - Bubble size represents total claims amount
            - The red dashed line shows the break-even point (KES {break_even_claim:,.2f})
                - Providers below the line are in **profit territory**
                - Providers above the line are in **loss territory**
            - Break-even source: {break_even_source}
            - Hover over bubbles to see detailed provider metrics
            """)
        
            # Add provider performance summary table
            provider_stats['Performance'] = provider_stats['Avg_Claim'].apply(
                lambda x: '✅ Profit' if x < break_even_claim else '⚠️ Loss'
            )
        
            st.write("#### Provider Performance Summary")
            st.dataframe(
                provider_stats[['Provider', 'Avg_Claim', 'Claim_Count', 'Performance']]
                .sort_values('Avg_Claim', ascending=False)
                .style.format({'Avg_Claim': 'KES {:,.2f}'})
                .apply(lambda x: ['background-color: #ffcccc' if v == '⚠️ Loss' else '' for v in x], 
                    subset=['Performance'])
            )
    
    with tab4:
        st.write("### Diagnosis-Treatment Patterns")
        if 'Diagnosis' in df.columns and 'Treatment' in df.columns:
            diag_treat = df.groupby(['Diagnosis_Group', 'Treatment_Type']).size().reset_index(name='Count')
            
            chart = alt.Chart(diag_treat).mark_rect().encode(
                x='Treatment_Type:N',
                y='Diagnosis_Group:N',
                color='Count:Q',
                tooltip=['Diagnosis_Group', 'Treatment_Type', 'Count']
            ).properties(width=600, height=400)
            st.altair_chart(chart, use_container_width=True)

def render_training():
    st.header("🤖 Advanced Model Training")
    
    if st.session_state.predictor.clean_data is None:
        st.warning("Please clean data first")
        return
    
    st.write("### Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        target_var = st.selectbox("Target Variable", 
                                ['Claim_Amount_KES', 'Inpatient_Cap_KES_Utilization'],
                                index=0)
        model_type = st.selectbox("Model Algorithm",
                                ["Gradient Boosting", "Random Forest", "XGBoost", 
                                 "Neural Network", "Auto Select Best"],
                                index=0)
    with col2:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
        
        with st.expander("Advanced Options"):
            do_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
            max_iter = st.number_input("Max Tuning Iterations", 10, 100, 20) if do_tuning else 20  # Default value when not tuning
    
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            results_df = st.session_state.predictor.train_model(
                model_type=model_type,
                target=target_var,
                test_size=test_size/100,
                cv_folds=cv_folds,
                do_tuning=do_tuning,
                max_iter=max_iter  # Now always defined
            )
        
        if results_df is not None:

            # Display results
            st.subheader("Model Comparison")
            st.dataframe(results_df.style.format({
                'MAE': '{:,.2f}',
                'RMSE': '{:,.2f}',
                'R2': '{:.3f}'
            }).background_gradient(cmap='Blues', subset=['R2']))
            
            # Visual comparison
            chart_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
            chart = alt.Chart(chart_df).mark_bar().encode(
                x='Model',
                y='Value',
                color='Model',
                column='Metric'
            ).properties(width=200)
            st.altair_chart(chart)
            
            # Show feature importance
            if st.session_state.predictor.feature_importance is not None:
                st.subheader("Feature Importance")
                chart = alt.Chart(
                    st.session_state.predictor.feature_importance.head(10)
                ).mark_bar().encode(
                    x='Importance',
                    y=alt.Y('Feature', sort='-x'),
                    color='Feature',
                    tooltip=['Feature', 'Importance']
                ).properties(title='Top 10 Important Features')
                st.altair_chart(chart, use_container_width=True)
                
                # Add SHAP explanation option
                if st.checkbox("Show SHAP summary plot"):
                    self.explain_model()

def render_prediction():
    st.header("🔮 Claim Prediction")
    
    if st.session_state.predictor.model is None:
        st.warning("Please train the model first")
        return
    
    # Prediction type selector
    prediction_type = st.radio(
        "Prediction Type",
        ["Individual Claim", "Group Claims"],
        horizontal=True,
        key='prediction_type'
    )
    
    if prediction_type == "Individual Claim":
        # Use container to match the design in the image
        with st.container():
            st.markdown("### Claim Details")
            
            # Two-column layout as shown in the image
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Visit Type")
                visit_type = st.selectbox(
                    "Visit Type",
                    options=st.session_state.predictor.available_values.get('Visit_Type', []),
                    index=0 if not st.session_state.predictor.available_values.get('Visit_Type') else None,
                    key='visit_type',
                    label_visibility='collapsed'
                )
                
                st.markdown("#### Diagnosis")
                diagnosis = st.selectbox(
                    "Diagnosis",
                    options=st.session_state.predictor.available_values.get('Diagnosis', []),
                    index=0 if not st.session_state.predictor.available_values.get('Diagnosis') else None,
                    key='diagnosis',
                    label_visibility='collapsed'
                )
                
                st.markdown("#### Treatment")
                treatment = st.selectbox(
                    "Treatment",
                    options=st.session_state.predictor.available_values.get('Treatment', []),
                    index=0 if not st.session_state.predictor.available_values.get('Treatment') else None,
                    key='treatment',
                    label_visibility='collapsed'
                )
                
                st.markdown("#### Provider Name")
                provider = st.selectbox(
                    "Provider Name",
                    options=st.session_state.predictor.available_values.get('Provider_Name', []),
                    index=0 if not st.session_state.predictor.available_values.get('Provider_Name') else None,
                    key='provider',
                    label_visibility='collapsed'
                )
                
                st.markdown("#### Hospital County")
                county = st.selectbox(
                    "Hospital County",
                    options=st.session_state.predictor.available_values.get('Hospital_County', []),
                    index=0 if not st.session_state.predictor.available_values.get('Hospital_County') else None,
                    key='county',
                    label_visibility='collapsed'
                )
            
            with col2:
                st.markdown("#### Employee Age")
                age = st.number_input(
                    "Employee Age",
                    min_value=18,
                    max_value=100,
                    value=40,
                    key='age',
                    label_visibility='collapsed'
                )
                
                st.markdown("#### Employee Gender")
                gender = st.selectbox(
                    "Employee Gender",
                    options=st.session_state.predictor.available_values.get('Employee_Gender', []),
                    index=0 if not st.session_state.predictor.available_values.get('Employee_Gender') else None,
                    key='gender',
                    label_visibility='collapsed'
                )
                
                st.markdown("#### Co-Payment Amount (KES)")
                co_payment = st.number_input(
                    "Co-Payment Amount (KES)",
                    min_value=0,
                    value=500,
                    key='co_payment',
                    label_visibility='collapsed'
                )
                
                st.markdown("#### Pre-Authorization Required")
                pre_auth = st.selectbox(
                    "Pre-Authorization Required",
                    options=["Yes", "No"],
                    index=0,
                    key='pre_auth',
                    label_visibility='collapsed'
                )
                
                st.markdown("#### Benefit Category")
                category = st.selectbox(
                    "Benefit Category",
                    options=st.session_state.predictor.available_values.get('Category', ['Silver', 'Gold', 'Platinum']),
                    index=0,
                    key='category',
                    label_visibility='collapsed'
                )
            
            # Predict button centered below the columns
            st.markdown("---")
            predict_btn = st.button("Predict Claim Amount", use_container_width=True)
            
            if predict_btn:
                input_data = {
                    'Visit_Type': visit_type,
                    'Diagnosis': diagnosis,
                    'Treatment': treatment,
                    'Provider_Name': provider,
                    'Hospital_County': county,
                    'Employee_Age': age,
                    'Employee_Gender': gender,
                    'Co_Payment_KES': co_payment,
                    'Pre_Authorization_Required': pre_auth,
                    'Category': category
                }
                
                # Add derived features
                input_data['Diagnosis_Group'] = diagnosis.split()[0] if pd.notna(diagnosis) else 'Other'
                input_data['Treatment_Type'] = treatment.split()[0] if pd.notna(treatment) else 'Other'
                input_data['Is_Pre_Authorized'] = 1 if pre_auth == "Yes" else 0
                
                # Set caps based on category
                if category == 'Silver':
                    caps = [750000, 75000, 40000, 40000, 50000]
                elif category == 'Gold':
                    caps = [1000000, 100000, 50000, 50000, 75000]
                else:  # Platinum
                    caps = [1500000, 150000, 75000, 75000, 100000]
                    
                cap_cols = ['Inpatient_Cap_KES', 'Outpatient_Cap_KES', 
                           'Optical_Cap_KES', 'Dental_Cap_KES', 'Maternity_Cap_KES']
                for col, cap in zip(cap_cols, caps):
                    input_data[col] = cap
                    input_data[f'{col}_Utilization'] = 0.5  # Default utilization
                
                # Add temporal features
                input_data['Claim_Weekday'] = datetime.now().strftime('%A')
                input_data['Claim_Month'] = datetime.now().strftime('%B')
                
                # Add employer and department if needed
                if 'Employer' not in input_data:
                    input_data['Employer'] = 'Unknown'
                if 'Department' not in input_data:
                    input_data['Department'] = 'General'
                
                # Make prediction
                with st.spinner("Calculating prediction..."):
                    prediction_result = st.session_state.predictor.predict_claim_amount(input_data)
                
                if prediction_result is not None:
                    st.success(f"""
                    ### Predicted Claim Amount: KES {prediction_result['prediction']:,.2f}
                    """)
                    
                    # Fraud alert
                    if prediction_result['is_potential_fraud']:
                        st.error(f"""
                        ⚠️ Potential fraud detected (confidence: {prediction_result['fraud_confidence']:.1%})
                        """)
                    
                    # Interpretation
                    if prediction_result['prediction'] > 100000:
                        st.warning("⚠️ High predicted claim amount - consider pre-authorization")
                    elif prediction_result['prediction'] < 20000:
                        st.info("✅ Low predicted claim amount - within typical range")
                    
                    # Show explanation
                    if st.checkbox("Show prediction explanation"):
                        st.session_state.predictor.explain_prediction(pd.DataFrame([input_data]))
    
    else:  # Group Claims
        st.write("### Group Claims Prediction")
        
        # Option 1: Upload group data
        uploaded_file = st.file_uploader(
            "Upload employee group data (CSV or Excel)",
            type=['csv', 'xlsx'],
            help="File should contain employee details for prediction"
        )
        
        # Option 2: Use existing data with filters
        use_existing = st.checkbox("Use existing cleaned data with filters")
        
        if uploaded_file is not None:
            with st.spinner('Loading group data...'):
                group_data = st.session_state.predictor.load_data(uploaded_file)
            
            if group_data is not None:
                st.success(f"Loaded {len(group_data)} records")
                st.dataframe(group_data.head())
                
                if st.button("Predict Group Claims"):
                    with st.spinner("Processing group predictions..."):
                        predictions, stats = st.session_state.predictor.predict_group_claims(group_data)
                    
                    if predictions is not None:
                        st.success("Group predictions completed!")
                        st.session_state.predictor.visualize_group_predictions(predictions, stats)
                        
                        # Download results
                        csv = predictions.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="group_claim_predictions.csv",
                            mime="text/csv"
                        )
        
        elif use_existing and st.session_state.predictor.clean_data is not None:
            st.write("#### Filter Existing Data")
            
            # Create filters - MODIFIED TO HANDLE MISSING DEPARTMENT
            cols = st.columns(3)
            with cols[0]:
                employers = st.multiselect(
                    "Employers",
                    options=st.session_state.predictor.clean_data['Employer'].unique(),
                    default=st.session_state.predictor.clean_data['Employer'].unique()
                )
            with cols[1]:
                # Check if Department exists before using it
                if 'Department' in st.session_state.predictor.clean_data.columns:
                    departments = st.multiselect(
                        "Departments",
                        options=st.session_state.predictor.clean_data['Department'].unique(),
                        default=st.session_state.predictor.clean_data['Department'].unique()
                    )
                else:
                    departments = ['General']  # Default value if no departments
                    st.warning("No Department column found - using default department")
            with cols[2]:
                categories = st.multiselect(
                    "Benefit Categories",
                    options=st.session_state.predictor.clean_data['Category'].unique(),
                    default=st.session_state.predictor.clean_data['Category'].unique()
                )
            
            # Apply filters - MODIFIED TO HANDLE MISSING DEPARTMENT
            filter_conditions = [
                (st.session_state.predictor.clean_data['Employer'].isin(employers)),
                (st.session_state.predictor.clean_data['Category'].isin(categories))
            ]
            
            if 'Department' in st.session_state.predictor.clean_data.columns:
                filter_conditions.append(
                    st.session_state.predictor.clean_data['Department'].isin(departments)
                )
            
            filtered_data = st.session_state.predictor.clean_data[np.all(filter_conditions, axis=0)]
            
            st.info(f"Filtered to {len(filtered_data)} records")
            
            if st.button("Predict Filtered Group Claims"):
                with st.spinner("Processing filtered group predictions..."):
                    predictions, stats = st.session_state.predictor.predict_group_claims(filtered_data)
                
                if predictions is not None:
                    st.success("Group predictions completed!")
                    st.session_state.predictor.visualize_group_predictions(predictions, stats)
                    
                    # Download results
                    csv = predictions.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="filtered_claim_predictions.csv",
                        mime="text/csv"
                    )
        
        elif use_existing and st.session_state.predictor.clean_data is not None:
            st.write("#### Filter Existing Data")
            
            # Create filters
            cols = st.columns(3)
            with cols[0]:
                employers = st.multiselect(
                    "Employers",
                    options=st.session_state.predictor.clean_data['Employer'].unique(),
                    default=st.session_state.predictor.clean_data['Employer'].unique()
                )
            with cols[1]:
                departments = st.multiselect(
                    "Departments",
                    options=st.session_state.predictor.clean_data['Department'].unique(),
                    default=st.session_state.predictor.clean_data['Department'].unique()
                )
            with cols[2]:
                categories = st.multiselect(
                    "Benefit Categories",
                    options=st.session_state.predictor.clean_data['Category'].unique(),
                    default=st.session_state.predictor.clean_data['Category'].unique()
                )
            
            # Apply filters
            filtered_data = st.session_state.predictor.clean_data[
                (st.session_state.predictor.clean_data['Employer'].isin(employers)) &
                (st.session_state.predictor.clean_data['Department'].isin(departments)) &
                (st.session_state.predictor.clean_data['Category'].isin(categories))
            ]
            
            st.info(f"Filtered to {len(filtered_data)} records")
            
            if st.button("Predict Filtered Group Claims"):
                with st.spinner("Processing filtered group predictions..."):
                    predictions, stats = st.session_state.predictor.predict_group_claims(filtered_data)
                
                if predictions is not None:
                    st.success("Group predictions completed!")
                    st.session_state.predictor.visualize_group_predictions(predictions, stats)
                    
                    # Download results
                    csv = predictions.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="filtered_claim_predictions.csv",
                        mime="text/csv"
                    )

def render_impact_analysis():
    st.header("🔄 Impact Analysis")
    
    if st.session_state.predictor.model is None:
        st.warning("Please train the model first")
        return
    
    if st.session_state.predictor.clean_data is None:
        st.warning("Please clean data first")
        return
                
    # Analysis type selector
    analysis_type = st.radio(
        "Analysis Type",
        ["Individual Analysis", "Group Analysis"],
        horizontal=True,
        key='impact_analysis_type'
    )
    
    if analysis_type == "Individual Analysis":
        # Select employee
        employee_id = st.selectbox(
            "Select Employee", 
            st.session_state.predictor.clean_data['Employee_ID'].unique()
        )
        
        if st.button("Analyze Category Impact"):
            with st.spinner("Analyzing impact..."):
                impact_df = st.session_state.predictor.analyze_category_shift(employee_id)
            
            if impact_df is not None:
                st.subheader(f"Impact Analysis for Employee {employee_id}")
                
                # Current category
                current_category = st.session_state.predictor.clean_data[
                    st.session_state.predictor.clean_data['Employee_ID'] == employee_id
                ]['Category'].values[0]
                
                st.info(f"Current Benefit Category: {current_category}")
                
                # Visualizations
                st.session_state.predictor.visualize_category_impact(impact_df, is_group=False)
                
                # Detailed comparison
                st.subheader("Detailed Comparison")
                
                # Calculate differences from current
                current_row = impact_df[impact_df['Category'] == current_category].iloc[0]
                impact_df['Change_From_Current'] = impact_df['Predicted_Claim'] - current_row['Predicted_Claim']
                impact_df['Pct_Change'] = (impact_df['Predicted_Claim'] / current_row['Predicted_Claim'] - 1) * 100
                
                # Format for display
                display_df = impact_df.copy()
                display_df['Predicted_Claim'] = display_df['Predicted_Claim'].apply(lambda x: f"KES {x:,.2f}")
                display_df['Change_From_Current'] = display_df['Change_From_Current'].apply(
                    lambda x: f"KES {x:+,.2f}")
                display_df['Pct_Change'] = display_df['Pct_Change'].apply(
                    lambda x: f"{x:+.1f}%")
                
                st.dataframe(display_df.set_index('Category'))
    
    else:  # Group Analysis
        st.write("### Group Selection")
        
        # Create group filters
        cols = st.columns(3)
        with cols[0]:
            employers = st.multiselect(
                "Select Employers",
                options=st.session_state.predictor.clean_data['Employer'].unique(),
                default=st.session_state.predictor.clean_data['Employer'].unique()
            )
        with cols[1]:
            departments = st.multiselect(
                "Select Departments",
                options=st.session_state.predictor.clean_data['Department'].unique(),
                default=st.session_state.predictor.clean_data['Department'].unique()
            )
        with cols[2]:
            categories = st.multiselect(
                "Filter by Current Category",
                options=st.session_state.predictor.clean_data['Category'].unique(),
                default=st.session_state.predictor.clean_data['Category'].unique()
            )
        
        group_filter = {
            'employers': employers,
            'departments': departments,
            'categories': categories
        }
        
        if st.button("Analyze Group Category Impact"):
            with st.spinner("Analyzing group impact..."):
                impact_df = st.session_state.predictor.analyze_category_shift(group_filter=group_filter)
            
            if impact_df is not None:
                st.subheader("Group Benefit Category Impact Analysis")
                
                # Current category distribution
                current_categories = st.session_state.predictor.clean_data[
                    (st.session_state.predictor.clean_data['Employer'].isin(employers)) &
                    (st.session_state.predictor.clean_data['Department'].isin(departments))
                ]['Category'].value_counts(normalize=True).reset_index()
                current_categories.columns = ['Category', 'Percentage']
                
                st.write("### Current Category Distribution")
                st.altair_chart(alt.Chart(current_categories).mark_bar().encode(
                    x='Category',
                    y='Percentage',
                    color='Category'
                ))
                
                # Impact analysis
                st.session_state.predictor.visualize_category_impact(impact_df, is_group=True)
                
                # Cost savings analysis
                st.write("### Potential Cost Savings")
                base_category = impact_df.iloc[0]['Category']
                for _, row in impact_df.iterrows():
                    if row['Category'] != base_category:
                        savings = impact_df.loc[impact_df['Category'] == base_category, 'Total_Predicted_Claims'].values[0] - row['Total_Predicted_Claims']
                        st.write(f"Changing from {base_category} to {row['Category']} could save KES {savings:,.2f} annually")

def render_reporting():
    st.header("📊 Advanced Reporting")
    
    if st.session_state.predictor.clean_data is None:
        st.warning("Please clean data first")
        return
    
    if st.session_state.predictor.model is None:
        st.warning("Please train the model first")
        return
    
    # Report configuration
    st.write("### Report Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        report_format = st.radio("Report Format", ["HTML", "PDF"], index=0)
        analysis_type = st.radio("Analysis Type", ["Group Analysis", "Individual Analysis"], index=0)
    
    with col2:
        if analysis_type == "Individual Analysis":
            employee_id = st.selectbox(
                "Select Employee", 
                st.session_state.predictor.clean_data['Employee_ID'].unique()
            )
            group_filter = None
        else:
            # Group selection options
            employers = st.multiselect(
                "Select Employers",
                options=st.session_state.predictor.clean_data['Employer'].unique(),
                default=st.session_state.predictor.clean_data['Employer'].unique()
            )
            
            # Additional group filters
            with st.expander("Advanced Group Filters"):
                departments = st.multiselect(
                    "Filter by Department",
                    options=st.session_state.predictor.clean_data['Department'].unique(),
                    default=st.session_state.predictor.clean_data['Department'].unique()
                )
                
                categories = st.multiselect(
                    "Filter by Benefit Category",
                    options=st.session_state.predictor.clean_data['Category'].unique(),
                    default=st.session_state.predictor.clean_data['Category'].unique()
                )
            
            group_filter = {
                'employers': employers,
                'departments': departments,
                'categories': categories
            }
            employee_id = None
    
    if st.button("Generate Report"):
        st.session_state.predictor.generate_report(
            employee_id=employee_id,
            group_filter=group_filter,
            format=report_format
        )
    
    # Model documentation
    st.write("### Model Documentation")
    if st.button("Generate Model Card"):
        model_card = st.session_state.predictor.generate_model_card()
        st.json(model_card)
    
    # Data snapshot
    st.write("### Data Versioning")
    if st.button("Create Snapshot"):
        if st.session_state.predictor.save_snapshot():
            st.success("Snapshot created successfully!")
        else:
            st.error("Failed to create snapshot")

def main():
    st.set_page_config(
        page_title="Healthcare Claims Analyzer",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for responsive design
    st.markdown("""
        <style>
            .main { padding: 2rem; }
            .stButton>button { background-color: #4CAF50; color: white; }
            .metric-card { 
                background-color: #f0f2f6; 
                padding: 1rem; 
                border-radius: 0.5rem; 
                margin-bottom: 1rem; 
            }
            .feature-importance { font-size: 0.9rem; }
            .section { margin-bottom: 2rem; }
            @media screen and (max-width: 768px) {
                .main { padding: 0.5rem; }
                .stButton>button { width: 100%; }
                .metric-card { margin-bottom: 0.5rem; }
                .section { margin-bottom: 1rem; }
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = ClaimAmountPredictor()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        app_mode = st.radio(
            "Select Mode",
            ["Data Upload", "Data Cleaning", "Exploratory Analysis", 
             "Model Training", "Claim Prediction", "Impact Analysis", "Reporting"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
            Comprehensive healthcare claims analysis tool with predictive modeling
            and scenario analysis capabilities.
            """)
        
        # API launch button
        if st.button("Launch API"):
            api = create_api(st.session_state.predictor)
            import uvicorn
            uvicorn.run(api, host="0.0.0.0", port=8000)
    
    # Error handling wrapper for main app sections
    def handle_errors(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                st.error(f"An error occurred: {str(e)}")
                st.stop()
        return wrapper
    
    # Main app routing
    if app_mode == "Data Upload":
        render_data_upload()
    elif app_mode == "Data Cleaning":
        render_data_cleaning()
    elif app_mode == "Exploratory Analysis":
        render_exploratory_analysis()
    elif app_mode == "Model Training":
        render_training()
    elif app_mode == "Claim Prediction":
        render_prediction()
    elif app_mode == "Impact Analysis":
        render_impact_analysis()
    elif app_mode == "Reporting":
        render_reporting()

if __name__ == "__main__":
    main()