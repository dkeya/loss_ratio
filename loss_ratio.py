import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                             AdaBoostClassifier, StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, accuracy_score,
                           confusion_matrix, f1_score, balanced_accuracy_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone
from scipy.stats import ks_2samp
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import os
import time
import json
from io import BytesIO
import hashlib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
from PIL import Image  # Added for image handling

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up the page
st.set_page_config(page_title="Loss Ratio Predictor", layout="wide")
st.title("🌀 Insurance Loss Ratio Prediction")

# Initialize session state variables
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'predictor' not in st.session_state:
    st.session_state['predictor'] = None
if 'performance_history' not in st.session_state:
    st.session_state['performance_history'] = []
if 'last_retraining' not in st.session_state:
    st.session_state['last_retraining'] = None
if 'model_versions' not in st.session_state:
    st.session_state['model_versions'] = []
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "📁 Data Management"
if 'form_data' not in st.session_state:
    st.session_state['form_data'] = {
        'pred_age': 35,
        'pred_gender': "Male",
        'pred_employment': "Full-time",
        'pred_claims': 0,
        'pred_chronic': "None",
        'pred_policy': "Inpatient Only",
        'pred_sum': 500000,
        'pred_deductible': 5000,
        'pred_copay': 10,
        'pred_family': "None"
    }

class LossRatioPredictor:
    def __init__(self):
        self.model = None
        self.df = pd.DataFrame()
        self.preprocessor = None
        self.feature_importances = None
        self.class_labels = ['Low', 'Medium', 'High', 'Very High']
        self.label_encoder = LabelEncoder()
        self.bins = [-np.inf, 0.2, 0.5, 0.8, np.inf]
        self.training_time = None
        self.feature_columns = ['Age', 'Gender', 'Employment_Type', 'Previous_Claims',
                              'Policy_Type', 'Sum_Insured', 'Deductible', 'Co_Payment',
                              'Chronic_Conditions', 'Family_History']
        self.performance_metrics = {}
        self.model_version = None
        self.model_metadata = {}

    def validate_data(self, df):
        """Perform comprehensive data validation with enhanced checks"""
        errors = []
        warnings = []

        # Check for missing values (excluding None in Chronic_Conditions and Family_History)
        missing_values = df.drop(columns=['Chronic_Conditions', 'Family_History']).isnull().sum()
        if missing_values.sum() > 0:
            errors.append("Dataset contains missing values in required fields")
            with st.expander("Missing Value Details"):
                st.write(missing_values[missing_values > 0])

        # Validate numeric ranges
        if (df['Age'] < 18).any() or (df['Age'] > 100).any():
            errors.append("Age values must be between 18 and 100")

        if (df['Sum_Insured'] < 0).any():
            errors.append("Sum Insured cannot be negative")

        if (df['Annual_Premium'] <= 0).any():
            errors.append("Annual Premium must be positive")

        # Check for outliers using multiple methods
        numeric_cols = ['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio']
        for col in numeric_cols:
            if col in df.columns:
                # Z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                if (z_scores > 3).sum() > 0:
                    warnings.append(f"Found {len(df[z_scores > 3])} potential outliers in {col} (Z-score > 3)")

                # IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outlier_threshold = q3 + 1.5 * iqr
                if (df[col] > outlier_threshold).sum() > 0:
                    warnings.append(f"Found {len(df[df[col] > outlier_threshold])} outlier(s) in {col} (IQR method)")

        # Check for data consistency
        if 'Loss_Ratio' in df.columns and 'Annual_Premium' in df.columns:
            if (df['Loss_Ratio'] * df['Annual_Premium'] < 0).any():
                errors.append("Inconsistent data: Loss Ratio and Annual Premium signs don't match")

        return errors, warnings

    def preprocess_data(self, df):
        """Bin loss ratios and prepare features with enhanced preprocessing"""
        # Enhanced outlier handling
        if 'Loss_Ratio' in df.columns:
            # Cap at 99.5th percentile instead of 99th for more conservative handling
            cap = df['Loss_Ratio'].quantile(0.995)
            df['Loss_Ratio'] = np.where(df['Loss_Ratio'] > cap, cap, df['Loss_Ratio'])

            # Floor at 0 if any negative values somehow got through
            df['Loss_Ratio'] = np.where(df['Loss_Ratio'] < 0, 0, df['Loss_Ratio'])

            # Adaptive binning if standard bins don't work
            try:
                df['Loss_Ratio_Category'] = pd.cut(
                    df['Loss_Ratio'],
                    bins=self.bins,
                    labels=self.class_labels
                )
            except ValueError:
                st.warning("Standard binning failed, using quantile-based binning")
                df['Loss_Ratio_Category'] = pd.qcut(
                    df['Loss_Ratio'],
                    q=4,
                    labels=self.class_labels
                )

            # Encode the target variable for XGBoost
            self.label_encoder.fit(self.class_labels)
            df['Loss_Ratio_Encoded'] = self.label_encoder.transform(df['Loss_Ratio_Category'])

        # Handle None values in Chronic_Conditions and Family_History
        if 'Chronic_Conditions' in df.columns:
            df['Chronic_Conditions'] = df['Chronic_Conditions'].fillna('None')
        if 'Family_History' in df.columns:
            df['Family_History'] = df['Family_History'].fillna('None')

        return df

    def handle_class_imbalance(self, X, y):
        """Apply SMOTE and random undersampling to handle class imbalance"""
        # Check if imbalance exists
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        max_samples = class_counts.max()

        if max_samples / min_samples > 2:  # Significant imbalance
            st.info(f"Applying SMOTE and RandomUnderSampler to handle class imbalance (Class counts: {dict(class_counts)})")

            # Define resampling
            over = SMOTE(sampling_strategy='not majority', k_neighbors=min(5, min_samples-1))
            under = RandomUnderSampler(sampling_strategy='not minority')

            # Create pipeline
            pipeline = ImbPipeline([
                ('preprocessor', clone(self.preprocessor)),
                ('over', over),
                ('under', under)
            ])

            # Resample
            X_res, y_res = pipeline.fit_resample(X, y)
            return X_res, y_res

        return X, y

    def train_model(self, df):
        """Train the loss ratio prediction model with enhanced options"""
        df = self.preprocess_data(df)

        # Data validation
        errors, warnings = self.validate_data(df)
        if errors:
            for error in errors:
                st.error(error)
            return None, None, None

        for warning in warnings:
            st.warning(warning)

        # Enhanced class distribution visualization
        if 'Loss_Ratio_Category' in df.columns:
            class_counts = df['Loss_Ratio_Category'].value_counts().reset_index()
            class_counts.columns = ['Loss_Ratio_Category', 'Count']
            st.write("Class distribution:", class_counts)

            fig = px.pie(
                class_counts,
                names='Loss_Ratio_Category',
                values='Count',
                title='Class Distribution',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)

            # Verify we have at least 2 classes with sufficient samples
            if len(class_counts) < 2:
                st.error("Error: Only one class detected in target variable.")
                st.warning("Attempting to adjust binning strategy...")

                try:
                    # Try more sophisticated binning
                    est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
                    df['Loss_Ratio_Category'] = est.fit_transform(df[['Loss_Ratio']]).astype(int)
                    df['Loss_Ratio_Category'] = df['Loss_Ratio_Category'].map({
                        0: 'Very Low',
                        1: 'Low',
                        2: 'Medium',
                        3: 'High'
                    })

                    new_counts = df['Loss_Ratio_Category'].value_counts()
                    st.write("New class distribution after adaptive binning:", new_counts)

                    if len(new_counts) < 2:
                        raise ValueError("Still only one class after adaptive binning")

                    self.class_labels = sorted(new_counts.index.tolist())
                    self.label_encoder.fit(self.class_labels)
                    df['Loss_Ratio_Encoded'] = self.label_encoder.transform(df['Loss_Ratio_Category'])
                except Exception as adaptive_e:
                    st.error(f"Adaptive binning failed: {str(adaptive_e)}")
                    return None, None, None

        # Define features and target
        categorical_features = ['Gender', 'Employment_Type', 'Policy_Type', 'Chronic_Conditions', 'Family_History']
        numerical_features = ['Age', 'Previous_Claims', 'Sum_Insured', 'Deductible', 'Co_Payment']

        X = df[categorical_features + numerical_features]
        y = df['Loss_Ratio_Encoded']  # Use encoded labels for training

        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Enhanced model selection with more options
        models_to_try = [
            {
                'name': 'XGBoost',
                'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                'handle_imbalance': True
            },
            {
                'name': 'GradientBoosting',
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'subsample': [0.8, 1.0]
                },
                'handle_imbalance': True
            },
            {
                'name': 'RandomForest',
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [3, 4, 5, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'handle_imbalance': False
            },
            {
                'name': 'AdaBoost',
                'model': AdaBoostClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.5, 1.0]
                },
                'handle_imbalance': False
            },
            {
                'name': 'Stacking',
                'model': StackingClassifier(
                    estimators=[
                        ('rf', RandomForestClassifier(random_state=42, n_estimators=50)),
                        ('gb', GradientBoostingClassifier(random_state=42, n_estimators=50)),
                        ('xgb', XGBClassifier(random_state=42, n_estimators=50))
                    ],
                    final_estimator=LogisticRegression(max_iter=1000),
                    cv=3
                ),
                'params': None,
                'handle_imbalance': True
            }
        ]

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        best_model = None
        best_accuracy = 0
        best_f1 = 0
        best_report = ""
        best_model_name = ""

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, model_config in enumerate(models_to_try):
            progress = (i + 1) / len(models_to_try)
            progress_bar.progress(progress)
            status_text.text(f"Training {model_config['name']}... ({int(progress*100)}%)")

            with st.spinner(f"Training {model_config['name']}..."):
                try:
                    # Handle class imbalance if needed
                    if model_config.get('handle_imbalance', False):
                        X_train_res, y_train_res = self.handle_class_imbalance(X_train, y_train)
                    else:
                        X_train_res, y_train_res = X_train, y_train

                    if model_config['params']:
                        # Try with GridSearchCV
                        pipeline = Pipeline([
                            ('preprocessor', self.preprocessor),
                            ('classifier', GridSearchCV(
                                model_config['model'],
                                model_config['params'],
                                cv=StratifiedKFold(n_splits=3),
                                scoring='balanced_accuracy',
                                n_jobs=-1
                            ))
                        ])
                    else:
                        # Try simple model
                        pipeline = Pipeline([
                            ('preprocessor', self.preprocessor),
                            ('classifier', model_config['model'])
                        ])

                    pipeline.fit(X_train_res, y_train_res)

                    # Get predictions
                    if model_config['params']:
                        model = pipeline.named_steps['classifier'].best_estimator_
                        best_params = pipeline.named_steps['classifier'].best_params_
                        st.write(f"Best params for {model_config['name']}:", best_params)
                    else:
                        model = pipeline.named_steps['classifier']

                    # Calibrate classifier if it has predict_proba
                    if hasattr(model, 'predict_proba'):
                        calibrated_model = CalibratedClassifierCV(model, cv='prefit')
                        pipeline.steps[-1] = ('classifier', calibrated_model)
                        pipeline.fit(X_train_res, y_train_res)

                    # Get feature importances based on model type
                    if model_config['name'] == 'Stacking':
                        # For stacking classifier, use average of base estimators' feature importances
                        base_importances = []
                        for name, estimator in model.named_estimators_.items():
                            if hasattr(estimator, 'feature_importances_'):
                                base_importances.append(estimator.feature_importances_)

                        if base_importances:
                            self.feature_importances = np.mean(base_importances, axis=0)
                        else:
                            self.feature_importances = None
                    elif hasattr(model, 'feature_importances_'):
                        self.feature_importances = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        # For linear models, use coefficients as importance
                        self.feature_importances = np.mean(np.abs(model.coef_), axis=0)
                    else:
                        self.feature_importances = None

                    y_pred = pipeline.predict(X_test)

                    # Convert back to original labels for evaluation
                    y_test_labels = self.label_encoder.inverse_transform(y_test)
                    y_pred_labels = self.label_encoder.inverse_transform(y_pred)

                    # Multiple evaluation metrics
                    accuracy = accuracy_score(y_test_labels, y_pred_labels)
                    balanced_acc = balanced_accuracy_score(y_test_labels, y_pred_labels)
                    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
                    report = classification_report(y_test_labels, y_pred_labels, target_names=self.class_labels, output_dict=True)

                    # Store performance metrics
                    cm = confusion_matrix(y_test_labels, y_pred_labels)
                    self.performance_metrics = {
                        'accuracy': accuracy,
                        'balanced_accuracy': balanced_acc,
                        'f1_score': f1,
                        'classification_report': report,
                        'confusion_matrix': cm.tolist(),
                        'timestamp': datetime.now().isoformat(),
                        'model_type': model_config['name']
                    }

                    # Add to performance history
                    st.session_state['performance_history'].append({
                        'model': model_config['name'],
                        'accuracy': accuracy,
                        'balanced_accuracy': balanced_acc,
                        'f1_score': f1,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Select best model based on balanced accuracy and F1
                    if balanced_acc > best_accuracy or (balanced_acc == best_accuracy and f1 > best_f1):
                        best_model = pipeline
                        best_accuracy = balanced_acc
                        best_f1 = f1
                        best_report = report
                        best_model_name = model_config['name']

                    st.success(f"{model_config['name']} succeeded with:")
                    st.write(f"- Accuracy: {accuracy:.2%}")
                    st.write(f"- Balanced Accuracy: {balanced_acc:.2%}")
                    st.write(f"- Weighted F1: {f1:.2%}")

                except Exception as e:
                    st.warning(f"{model_config['name']} failed: {str(e)}")
                    continue

        progress_bar.empty()
        status_text.empty()

        if best_model is not None:
            self.model = best_model
            self.training_time = datetime.now()
            st.session_state['last_retraining'] = self.training_time

            # Generate model version hash
            self.model_version = hashlib.md5(str(self.training_time).encode()).hexdigest()[:8]
            self.model_metadata = {
                'version': self.model_version,
                'training_time': self.training_time,
                'model_type': best_model_name,
                'performance': {
                    'accuracy': best_accuracy,
                    'f1_score': best_f1
                }
            }

            # Add to version history
            st.session_state['model_versions'].append(self.model_metadata)

            return best_model, best_accuracy, best_report
        else:
            st.error("All model attempts failed. Please check your data.")
            return None, None, None

    def check_for_drift(self, new_data):
        """Enhanced data drift detection using statistical tests"""
        if self.df.empty or new_data.empty:
            return False, {}

        drift_detected = False
        drift_report = {
            'summary': {},
            'detailed': {}
        }

        # Compare basic statistics
        numeric_cols = ['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio']
        categorical_cols = ['Gender', 'Employment_Type', 'Policy_Type', 'Chronic_Conditions', 'Family_History']

        for col in numeric_cols:
            if col in self.df.columns and col in new_data.columns:
                # KS test for distribution similarity
                ks_stat, ks_p = ks_2samp(self.df[col], new_data[col])
                drift_report['detailed'][col] = {
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'train_mean': self.df[col].mean(),
                    'new_mean': new_data[col].mean(),
                    'train_median': self.df[col].median(),
                    'new_median': new_data[col].median()
                }

                if ks_p < 0.05:  # Significant difference
                    drift_detected = True
                    drift_report['summary'][col] = {
                        'test': 'KS Test',
                        'p_value': ks_p,
                        'conclusion': 'Significant drift detected'
                    }

        for col in categorical_cols:
            if col in self.df.columns and col in new_data.columns:
                # Chi-square test for categorical variables
                train_counts = self.df[col].value_counts(normalize=True)
                new_counts = new_data[col].value_counts(normalize=True)

                # Align categories
                all_cats = list(set(train_counts.index) | set(new_counts.index))
                train_counts = train_counts.reindex(all_cats, fill_value=0)
                new_counts = new_counts.reindex(all_cats, fill_value=0)

                # Chi-square test
                chi2_stat = np.sum((train_counts - new_counts)**2 / (train_counts + 1e-9))

                drift_report['detailed'][col] = {
                    'chi2_statistic': chi2_stat,
                    'train_distribution': train_counts.to_dict(),
                    'new_distribution': new_counts.to_dict()
                }

                if chi2_stat > 10:  # Simple threshold for demonstration
                    drift_detected = True
                    drift_report['summary'][col] = {
                        'test': 'Chi-square',
                        'statistic': chi2_stat,
                        'conclusion': 'Significant drift detected'
                    }

        return drift_detected, drift_report

    def auto_retrain(self, new_data):
        """Enhanced auto-retraining with version control"""
        drift_detected, drift_report = self.check_for_drift(new_data)

        if drift_detected:
            st.warning("Data drift detected! Initiating automatic retraining...")

            with st.expander("View Detailed Drift Report"):
                st.json(drift_report)

            # Combine old and new data
            combined_df = pd.concat([self.df, new_data], ignore_index=True)

            with st.spinner("Retraining model with new data..."):
                model, accuracy, report = self.train_model(combined_df)
                if model is not None:
                    # Archive current model before replacing
                    if self.model is not None:
                        self.archive_model(reason="Auto-retraining due to data drift")

                    self.model = model
                    self.df = combined_df
                    st.session_state['model_loaded'] = True
                    st.session_state['predictor'] = self

                    st.success(f"✅ Model retrained successfully! New accuracy: {accuracy:.2%}")

                    # Show comparison with previous model
                    if st.session_state['performance_history'] and len(st.session_state['performance_history']) > 1:
                        prev_acc = st.session_state['performance_history'][-2]['accuracy']
                        st.metric("Accuracy Change", f"{accuracy:.2%}",
                                 delta=f"{(accuracy - prev_acc):.2%}")

                    return True
        return False

    def archive_model(self, reason=""):
        """Archive the current model before replacing it"""
        if self.model is None:
            return

        archive_dir = "models/archive/"
        os.makedirs(archive_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{archive_dir}archived_model_{timestamp}.pkl"

        metadata = {
            'archived_at': timestamp,
            'reason': reason,
            'performance': self.performance_metrics,
            'model_version': self.model_version
        }

        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'metadata': metadata,
            'label_encoder': self.label_encoder
        }, filename)

        st.info(f"Model archived as {filename}")

    def predict(self, input_data):
        """Enhanced prediction with more detailed output"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")

        input_df = pd.DataFrame([input_data])

        try:
            # Get numerical prediction
            prediction_encoded = self.model.predict(input_df)[0]
            # Convert back to original label
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]

            probabilities = self.model.predict_proba(input_df)[0]

            # Get SHAP values if available
            shap_values = None
            if hasattr(self.model.named_steps['classifier'], 'predict_proba'):
                try:
                    import shap
                    explainer = shap.Explainer(self.model.named_steps['classifier'])
                    shap_values = explainer.shap_values(input_df)
                except:
                    pass

            return {
                'prediction': prediction,
                'probabilities': probabilities,
                'shap_values': shap_values
            }
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

    def get_feature_importances(self):
        """Get feature importances with enhanced handling"""
        if self.feature_importances is None:
            return None

        # Get feature names after one-hot encoding
        numeric_features = ['Age', 'Previous_Claims', 'Sum_Insured', 'Deductible', 'Co_Payment']
        categorical_features = ['Gender', 'Employment_Type', 'Policy_Type', 'Chronic_Conditions', 'Family_History']

        # Get one-hot encoded column names
        ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_names = ohe.get_feature_names_out(categorical_features)

        all_features = numeric_features + list(categorical_names)

        # Handle cases where feature importances might not match (e.g., for stacked models)
        if len(self.feature_importances) != len(all_features):
            return None

        return pd.DataFrame({
            'Feature': all_features,
            'Importance': self.feature_importances
        }).sort_values('Importance', ascending=False)

    def save_model(self, path='models/loss_ratio/'):
        """Save model with enhanced versioning"""
        version = self.model_version or datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(path, exist_ok=True)
        filename = f"{path}loss_ratio_model_{version}.pkl"

        metadata = {
            'version': version,
            'features': self.feature_columns,
            'class_labels': self.class_labels,
            'training_time': self.training_time,
            'performance_metrics': self.performance_metrics,
            'model_type': str(type(self.model.named_steps['classifier'])) if self.model else None,
            'git_hash': self.get_git_hash() if self.is_git_repo() else None
        }

        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor,
            'metadata': metadata,
            'label_encoder': self.label_encoder
        }, filename)

        # Add to version history
        if version not in [v['version'] for v in st.session_state['model_versions']]:
            st.session_state['model_versions'].append({
                'version': version,
                'path': filename,
                'timestamp': datetime.now().isoformat(),
                'performance': self.performance_metrics
            })

        return filename

    def get_git_hash(self):
        """Get current git hash for reproducibility"""
        try:
            import subprocess
            return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except:
            return None

    def is_git_repo(self):
        """Check if we're in a git repository"""
        try:
            import subprocess
            subprocess.check_output(['git', 'rev-parse', '--is-inside-work-tree'])
            return True
        except:
            return False

    def generate_report(self):
        """Generate comprehensive report with enhanced information"""
        if self.model is None:
            return None

        # Feature importance
        feature_importance = self.get_feature_importances()
        feature_importance_dict = feature_importance.to_dict() if feature_importance is not None else None

        # Model type information
        model_type = str(type(self.model.named_steps['classifier']))

        # Training data summary
        train_summary = None
        if not self.df.empty and 'Loss_Ratio_Category' in self.df.columns:
            train_summary = {
                'num_samples': len(self.df),
                'class_distribution': dict(self.df['Loss_Ratio_Category'].value_counts()),
                'features': {
                    'numeric': self.df.select_dtypes(include=np.number).columns.tolist(),
                    'categorical': self.df.select_dtypes(exclude=np.number).columns.tolist()
                }
            }

        report = {
            'model_info': {
                'type': model_type,
                'version': self.model_version,
                'training_time': self.training_time.strftime('%Y-%m-%d %H:%M:%S') if self.training_time else None,
                'features': self.feature_columns,
                'classes': self.class_labels,
                'git_hash': self.get_git_hash() if self.is_git_repo() else None
            },
            'training_data': train_summary,
            'performance': self.performance_metrics,
            'feature_importance': feature_importance_dict,
            'drift_detection': {
                'last_check': datetime.now().isoformat(),
                'methods_used': ['KS Test', 'Chi-square']
            }
        }

        return report

    def get_member_list(self):
        """Return a DataFrame with member details and their loss ratio categories"""
        if self.df.empty or 'Loss_Ratio_Category' not in self.df.columns:
            return None

        # Select relevant columns for the member list
        member_cols = ['Age', 'Gender', 'Employment_Type', 'Policy_Type',
                      'Sum_Insured', 'Deductible', 'Co_Payment',
                      'Chronic_Conditions', 'Family_History',
                      'Loss_Ratio', 'Loss_Ratio_Category']

        # Create member DataFrame with calculated fields
        member_df = self.df[member_cols].copy()

        # Add risk level description
        risk_info = {
            'Low': 'Normal risk - standard pricing',
            'Medium': 'Moderate risk - slight premium adjustment',
            'High': 'High risk - requires underwriting review',
            'Very High': 'Very high risk - significant premium loading or decline'
        }

        member_df['Risk_Description'] = member_df['Loss_Ratio_Category'].map(risk_info)

        # Format numeric columns
        member_df['Loss_Ratio'] = member_df['Loss_Ratio'].apply(lambda x: f"{x:.1%}")

        return member_df

def explain_prediction(self, input_data):
    """Generate explanation for a specific prediction"""
    prediction_result = self.predict(input_data)

    if prediction_result['shap_values'] is None:
        return "Detailed explanation not available for this model type"

    # Create explanation text
    explanation = {
        'predicted_class': prediction_result['prediction'],
        'probabilities': dict(zip(self.class_labels, prediction_result['probabilities'])),
        'top_features': []
    }

    # Get feature names
    numeric_features = ['Age', 'Previous_Claims', 'Sum_Insured', 'Deductible', 'Co_Payment']
    categorical_features = ['Gender', 'Employment_Type', 'Policy_Type', 'Chronic_Conditions', 'Family_History']
    ohe = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
    categorical_names = ohe.get_feature_names_out(categorical_features)
    all_features = numeric_features + list(categorical_names)

    # Get SHAP values for the predicted class
    pred_class_idx = list(self.class_labels).index(prediction_result['prediction'])
    shap_values = prediction_result['shap_values'][pred_class_idx][0]

    # Get top contributing features
    feature_importance = pd.DataFrame({
        'feature': all_features,
        'shap_value': shap_values
    }).sort_values('shap_value', key=abs, ascending=False)

    explanation['top_features'] = feature_importance.head(5).to_dict('records')

    return explanation

def verify_model(model):
    """Enhanced model verification"""
    if isinstance(model, dict):
        required_keys = ['model', 'preprocessor', 'metadata', 'label_encoder']
        if not all(key in model for key in required_keys):
            return False

        # Check model can predict
        try:
            test_data = {k: 0 for k in model['metadata']['features']}
            input_df = pd.DataFrame([test_data])
            model['model'].predict(input_df)
            return True
        except:
            return False

    return hasattr(model, 'predict') and hasattr(model, 'predict_proba')

# Initialize predictor
if st.session_state.get('predictor') is None:
    predictor = LossRatioPredictor()
    st.session_state['predictor'] = predictor
else:
    predictor = st.session_state['predictor']

# Main app layout - using radio buttons for tabs to maintain state
tabs = ["📁 Data Management", "🔮 Prediction", "📈 Analysis", "⚙️ Model Operations", "👥 Member List"]
active_tab = st.radio(
    "Navigation",
    tabs,
    key="tab_selector",
    label_visibility="hidden",
    horizontal=True,
    index=tabs.index(st.session_state.active_tab)
)

# Update session state when tab changes
if st.session_state.active_tab != active_tab:
    st.session_state.active_tab = active_tab

# Add logo to sidebar
try:
    logo_path = "C:/Users/dkeya/Documents/projects/insurance/logo.png"
    logo = Image.open(logo_path)
    st.sidebar.image(
        logo, 
        use_container_width=True,  # Updated parameter
        caption="Insurance Analytics"  # Optional caption
    )
except FileNotFoundError:
    st.sidebar.warning("Logo image not found at specified path")
except Exception as e:
    st.sidebar.warning(f"Error loading logo: {str(e)}")

# Show content based on active tab
if st.session_state.active_tab == "📁 Data Management":
    with st.container():
        st.header("Data Preparation")

        # Data upload or sample data
        data_source = st.radio("Data source:", ["Upload your own", "Use sample data"], horizontal=True, key="data_source_radio")

        if data_source == "Upload your own":
            uploaded_file = st.file_uploader("Upload CSV/TSV file", type=['csv', 'tsv'], key="data_uploader")
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_csv(uploaded_file, sep='\t')

                    # Validate required columns
                    required_cols = ['Age', 'Gender', 'Employment_Type', 'Previous_Claims',
                                   'Policy_Type', 'Sum_Insured', 'Deductible', 'Co_Payment',
                                   'Chronic_Conditions', 'Family_History',
                                   'Last_Year_Claims', 'Annual_Premium']

                    if all(col in df.columns for col in required_cols):
                        # Enhanced data validation
                        errors, warnings = predictor.validate_data(df)

                        if errors:
                            st.error("Data validation failed. Please fix the following issues:")
                            for error in errors:
                                st.error(error)
                        else:
                            # Check for data drift if we already have a model
                            if hasattr(predictor, 'model') and predictor.model is not None and not predictor.df.empty:
                                drift_detected, drift_report = predictor.check_for_drift(df)
                                if drift_detected:
                                    st.warning("Data drift detected between new data and training data!")
                                    with st.expander("View Drift Details"):
                                        st.json(drift_report)

                                    if st.button("Auto-retrain model with new data", key="auto_retrain_btn"):
                                        predictor.auto_retrain(df)

                            predictor.df = df
                            predictor.df['Loss_Ratio'] = predictor.df['Last_Year_Claims'] / predictor.df['Annual_Premium']
                            st.success("✅ Data loaded successfully!")

                            with st.expander("View Data", expanded=False):
                                st.dataframe(predictor.df.head())

                            with st.expander("Data Statistics", expanded=False):
                                st.write(predictor.df.describe())

                            # Enhanced data visualization
                            st.subheader("Data Distribution")
                            tab1, tab2, tab3 = st.tabs(["Numerical", "Categorical", "Correlations"])

                            with tab1:
                                num_col = st.selectbox("Select numerical column",
                                                     ['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio'],
                                                     key="num_col_selector")

                                # Create two columns for side-by-side visualization
                                col1, col2 = st.columns(2)

                                with col1:
                                    # Histogram
                                    fig_hist = px.histogram(predictor.df, x=num_col, color='Gender',
                                                          nbins=50, title=f'Distribution of {num_col}')
                                    st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{num_col}_{time.time()}")

                                with col2:
                                    # Box plot
                                    fig_box = px.box(predictor.df, x='Gender', y=num_col,
                                                   color='Gender', title=f'Box Plot of {num_col} by Gender')
                                    st.plotly_chart(fig_box, use_container_width=True, key=f"box_{num_col}_{time.time()}")

                            with tab2:
                                cat_col = st.selectbox("Select categorical column",
                                                     ['Gender', 'Employment_Type', 'Policy_Type', 'Chronic_Conditions', 'Family_History'],
                                                     key="cat_col_selector")
                                fig = px.histogram(predictor.df, x=cat_col, color=cat_col,
                                                 title=f'Distribution of {cat_col}')
                                st.plotly_chart(fig, use_container_width=True, key=f"cat_dist_{cat_col}_{time.time()}")

                            with tab3:
                                num_col1 = st.selectbox("Select first numerical column",
                                                      ['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio'],
                                                      key='num_col1')
                                num_col2 = st.selectbox("Select second numerical column",
                                                      ['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio'],
                                                      key='num_col2')
                                fig = px.scatter(predictor.df, x=num_col1, y=num_col2, color='Policy_Type',
                                               title=f'{num_col1} vs {num_col2} by Policy Type')
                                st.plotly_chart(fig, use_container_width=True, key=f"scatter_{num_col1}_{num_col2}_{time.time()}")
                    else:
                        st.error(f"❌ Missing required columns. Needed: {', '.join(required_cols)}")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        else:
            if st.button("Generate Sample Data", key="generate_sample_btn"):
                df = predictor.load_sample_data()
                st.success("✅ Sample data generated!")

                with st.expander("View Sample Data", expanded=False):
                    st.dataframe(df.head())

                with st.expander("Sample Data Statistics", expanded=False):
                    st.write(df.describe())

        # Model training section
        if not predictor.df.empty:
            st.subheader("Model Training")

            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox("Select Model Type",
                                        ["Auto Select Best", "XGBoost", "Gradient Boosting",
                                         "Random Forest", "AdaBoost", "Stacking"],
                                        key="model_type_selector")

            with col2:
                handle_imbalance = st.checkbox("Handle Class Imbalance", value=True, key="handle_imbalance_check")

            if st.button("Train Loss Ratio Model", key="train_model_btn"):
                with st.spinner("Training model..."):
                    try:
                        model, accuracy, report = predictor.train_model(predictor.df)

                        if model is not None:  # Only proceed if training succeeded
                            predictor.model = model
                            st.session_state['model_loaded'] = True
                            st.session_state['predictor'] = predictor

                            st.success(f"✅ Model trained successfully!")
                            st.metric("Balanced Accuracy", f"{accuracy:.2%}")

                            with st.expander("View Classification Report", expanded=False):
                                st.text(json.dumps(report, indent=2))

                            # Enhanced model saving with version info
                            model_filename = predictor.save_model()
                            st.info(f"Model version: {predictor.model_version}")

                            # Download button
                            with open(model_filename, 'rb') as f:
                                st.download_button(
                                    label="Download Model",
                                    data=f,
                                    file_name=os.path.basename(model_filename),
                                    mime='application/octet-stream',
                                    key=f"download_model_{predictor.model_version}"
                                )

                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

elif st.session_state.active_tab == "🔮 Prediction":
    with st.container():
        st.header("Loss Ratio Prediction")

        if not hasattr(predictor, 'model') and not st.session_state.get('model_loaded'):
            st.warning("⚠️ Please train or load a model first")
        else:
            if st.session_state.get('model_loaded'):
                predictor = st.session_state['predictor']

            # Prediction form
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)

                with col1:
                    age = st.number_input("Age", min_value=18, max_value=100,
                                        value=st.session_state.form_data['pred_age'],
                                        key="pred_age")
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                        index=["Male", "Female", "Other"].index(st.session_state.form_data['pred_gender']),
                                        key="pred_gender")
                    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Contract"],
                                                  index=["Full-time", "Part-time", "Contract"].index(st.session_state.form_data['pred_employment']),
                                                  key="pred_employment")
                    previous_claims = st.number_input("Previous Claims", min_value=0,
                                                    value=st.session_state.form_data['pred_claims'],
                                                    key="pred_claims")
                    chronic_conditions = st.selectbox("Chronic Conditions",
                                                     ["None", "Hypertension", "Diabetes", "Asthma", "Heart Disease"],
                                                     index=["None", "Hypertension", "Diabetes", "Asthma", "Heart Disease"].index(st.session_state.form_data['pred_chronic']),
                                                     key="pred_chronic")

                with col2:
                    policy_type = st.selectbox("Policy Type", ["Inpatient Only", "Inpatient+Outpatient", "Comprehensive"],
                                              index=["Inpatient Only", "Inpatient+Outpatient", "Comprehensive"].index(st.session_state.form_data['pred_policy']),
                                              key="pred_policy")
                    sum_insured = st.number_input("Sum Insured (KES)", min_value=100000,
                                                value=st.session_state.form_data['pred_sum'],
                                                step=50000, key="pred_sum")
                    deductible = st.number_input("Deductible (KES)", min_value=0,
                                               value=st.session_state.form_data['pred_deductible'],
                                               step=1000, key="pred_deductible")
                    co_payment = st.number_input("Co-Payment (%)", min_value=0, max_value=50,
                                               value=st.session_state.form_data['pred_copay'],
                                               key="pred_copay")
                    family_history = st.selectbox("Family History",
                                               ["None", "Heart Disease", "Cancer", "Diabetes", "Hypertension"],
                                               index=["None", "Heart Disease", "Cancer", "Diabetes", "Hypertension"].index(st.session_state.form_data['pred_family']),
                                               key="pred_family")

                submitted = st.form_submit_button("Predict")

                if submitted:
                    # Update session state with current values
                    st.session_state.form_data = {
                        'pred_age': age,
                        'pred_gender': gender,
                        'pred_employment': employment_type,
                        'pred_claims': previous_claims,
                        'pred_chronic': chronic_conditions,
                        'pred_policy': policy_type,
                        'pred_sum': sum_insured,
                        'pred_deductible': deductible,
                        'pred_copay': co_payment,
                        'pred_family': family_history
                    }

                    input_data = {
                        'Age': age,
                        'Gender': gender,
                        'Employment_Type': employment_type,
                        'Previous_Claims': previous_claims,
                        'Policy_Type': policy_type,
                        'Sum_Insured': sum_insured,
                        'Deductible': deductible,
                        'Co_Payment': co_payment,
                        'Chronic_Conditions': chronic_conditions,
                        'Family_History': family_history
                    }

                    try:
                        prediction_result = predictor.predict(input_data)
                        prediction = prediction_result['prediction']
                        probabilities = prediction_result['probabilities']

                        # Display results
                        st.subheader("Prediction Result")

                        # Create a metric for each class with probability
                        cols = st.columns(len(predictor.class_labels))
                        for i, (col, label) in enumerate(zip(cols, predictor.class_labels)):
                            with col:
                                st.metric(
                                    label,
                                    f"{probabilities[i]:.1%}",
                                    delta="HIGH RISK" if label == prediction else None,
                                    delta_color="off" if label != prediction else "inverse"
                                )

                        # Enhanced interpretation
                        risk_info = {
                            'Low': {
                                'description': 'Normal risk profile - standard pricing applies',
                                'recommendation': 'No action needed',
                                'color': 'green'
                            },
                            'Medium': {
                                'description': 'Moderate risk - consider slight premium adjustment',
                                'recommendation': 'Review policy details, consider 5-10% premium increase',
                                'color': 'blue'
                            },
                            'High': {
                                'description': 'High risk - recommend detailed underwriting review',
                                'recommendation': 'Requires manual underwriting, consider 15-25% premium increase',
                                'color': 'orange'
                            },
                            'Very High': {
                                'description': 'Very high risk - significant premium loading or decline',
                                'recommendation': 'Consider declining or significant premium loading (30%+)',
                                'color': 'red'
                            }
                        }

                        st.info(f"""
                        **Interpretation:** {risk_info[prediction]['description']}

                        **Recommendation:** {risk_info[prediction]['recommendation']}
                        """)

                        # Visualize prediction distribution
                        st.subheader("Risk Distribution")
                        fig = px.pie(
                            names=predictor.class_labels,
                            values=probabilities,
                            title="Predicted Probability Distribution",
                            color=predictor.class_labels,
                            color_discrete_map={
                                'Low': 'green',
                                'Medium': 'blue',
                                'High': 'orange',
                                'Very High': 'red'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Show detailed explanation if available
                        if prediction_result['shap_values'] is not None:
                            with st.expander("Show Detailed Explanation", expanded=False):
                                explanation = predictor.explain_prediction(input_data)
                                st.write("**Prediction Explanation:**")
                                st.json(explanation)

                                # Visualize top features
                                if explanation.get('top_features'):
                                    top_features = pd.DataFrame(explanation['top_features'])
                                    fig = px.bar(top_features, x='feature', y='shap_value',
                                               title='Top Features Affecting Prediction',
                                               color='shap_value',
                                               color_continuous_scale='RdBu',
                                               range_color=[-abs(top_features['shap_value']).max(),
                                                           abs(top_features['shap_value']).max()])
                                    st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")

elif st.session_state.active_tab == "📈 Analysis":
    with st.container():
        st.header("Model Analysis")

        if not hasattr(predictor, 'model') and not st.session_state.get('model_loaded'):
            st.warning("⚠️ Please train or load a model first")
        else:
            if st.session_state.get('model_loaded'):
                predictor = st.session_state['predictor']

            # Feature importance
            st.subheader("Feature Importance")
            feature_importances = predictor.get_feature_importances()

            if feature_importances is not None:
                tab1, tab2 = st.tabs(["Bar Chart", "Detailed View"])

                with tab1:
                    fig = px.bar(
                        feature_importances,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importances',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.dataframe(feature_importances.style.format({'Importance': '{:.4f}'}))

                    # Export feature importance
                    if st.button("Export Feature Importance as CSV"):
                        csv = feature_importances.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="feature_importance.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("Feature importances not available for this model type")

            # Performance metrics
            if hasattr(predictor, 'performance_metrics') and predictor.performance_metrics:
                st.subheader("Model Performance")

                # Performance over time
                if st.session_state['performance_history']:
                    history_df = pd.DataFrame(st.session_state['performance_history'])

                    tab1, tab2, tab3 = st.tabs(["Accuracy", "Balanced Accuracy", "F1 Score"])

                    with tab1:
                        fig = px.line(
                            history_df,
                            x='timestamp',
                            y='accuracy',
                            color='model',
                            title='Model Accuracy Over Time',
                            markers=True,
                            labels={'accuracy': 'Accuracy'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        fig = px.line(
                            history_df,
                            x='timestamp',
                            y='balanced_accuracy',
                            color='model',
                            title='Model Balanced Accuracy Over Time',
                            markers=True,
                            labels={'balanced_accuracy': 'Balanced Accuracy'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with tab3:
                        fig = px.line(
                            history_df,
                            x='timestamp',
                            y='f1_score',
                            color='model',
                            title='Model F1 Score Over Time',
                            markers=True,
                            labels={'f1_score': 'F1 Score'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Confusion matrix
                cm = np.array(predictor.performance_metrics.get('confusion_matrix', []))
                if cm.size > 0:
                    st.subheader("Confusion Matrix")
                    fig = ff.create_annotated_heatmap(
                        cm,
                        x=predictor.class_labels,
                        y=predictor.class_labels,
                        colorscale='Blues',
                        showscale=True
                    )
                    fig.update_layout(title='Confusion Matrix')
                    st.plotly_chart(fig, use_container_width=True)

                # Enhanced classification report
                st.subheader("Classification Report")
                report_df = pd.DataFrame(predictor.performance_metrics.get('classification_report', {})).T

                # Style the report
                styled_report = report_df.style.format("{:.2f}").background_gradient(
                    cmap='Blues', subset=pd.IndexSlice[:, ['precision', 'recall', 'f1-score']]
                ).highlight_max(
                    subset=['precision', 'recall', 'f1-score'], color='lightgreen'
                ).highlight_min(
                    subset=['precision', 'recall', 'f1-score'], color='#ffcccb'
                )

                st.dataframe(styled_report)

            # Enhanced data exploration
            if not predictor.df.empty:
                st.subheader("Data Exploration")
                tab1, tab2, tab3 = st.tabs(["Distribution", "Relationships", "Advanced"])

                with tab1:
                    col1, col2 = st.columns(2)

                    with col1:
                        num_col = st.selectbox("Select numerical column",
                                            ['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio'],
                                            key='dist_num_col')
                        fig = px.histogram(
                            predictor.df,
                            x=num_col,
                            nbins=50,
                            color='Policy_Type',
                            marginal='box',
                            title=f'Distribution of {num_col} by Policy Type'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        cat_col = st.selectbox("Select categorical column",
                                             ['Gender', 'Employment_Type', 'Policy_Type', 'Chronic_Conditions', 'Family_History'],
                                             key='dist_cat_col')
                        fig = px.histogram(
                            predictor.df,
                            x=cat_col,
                            color=cat_col,
                            title=f'Distribution of {cat_col}'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    col1, col2 = st.columns(2)

                    with col1:
                        x_axis = st.selectbox("X-axis",
                                            ['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio'],
                                            key='rel_x')
                        y_axis = st.selectbox("Y-axis",
                                            ['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio'],
                                            key='rel_y')
                        color_by = st.selectbox("Color by",
                                              ['Policy_Type', 'Gender', 'Employment_Type', 'Chronic_Conditions', 'Family_History'],
                                              key='rel_color')

                        fig = px.scatter(
                            predictor.df,
                            x=x_axis,
                            y=y_axis,
                            color=color_by,
                            title=f'{x_axis} vs {y_axis} by {color_by}',
                            trendline='lowess'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        hue_col = st.selectbox("Facet by",
                                             ['None', 'Policy_Type', 'Gender', 'Employment_Type', 'Chronic_Conditions', 'Family_History'],
                                             key='rel_hue')

                        if hue_col != 'None':
                            fig = px.scatter_matrix(
                                predictor.df,
                                dimensions=['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio'],
                                color=hue_col
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = px.scatter_matrix(
                                predictor.df,
                                dimensions=['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio']
                            )
                            st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.write("Advanced statistical analysis")

                    # Correlation matrix
                    st.subheader("Correlation Matrix")
                    num_df = predictor.df.select_dtypes(include=np.number)
                    corr = num_df.corr()

                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu',
                        range_color=[-1, 1],
                        title='Correlation Matrix'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Pairplot-like visualization
                    st.subheader("Pairwise Relationships")
                    selected_cols = st.multiselect(
                        "Select columns for pairwise analysis",
                        num_df.columns.tolist(),
                        default=['Age', 'Sum_Insured', 'Annual_Premium', 'Loss_Ratio'],
                        key="pairwise_cols"
                    )

                    if len(selected_cols) >= 2:
                        fig = px.scatter_matrix(
                            predictor.df,
                            dimensions=selected_cols,
                            color='Policy_Type'
                        )
                        st.plotly_chart(fig, use_container_width=True)

elif st.session_state.active_tab == "⚙️ Model Operations":
    with st.container():
        st.header("Model Operations")

        if not hasattr(predictor, 'model') and not st.session_state.get('model_loaded'):
            st.warning("⚠️ Please train or load a model first")
        else:
            if st.session_state.get('model_loaded'):
                predictor = st.session_state['predictor']

            # Model version management
            st.subheader("Model Version Control")

            if st.session_state['model_versions']:
                versions_df = pd.DataFrame(st.session_state['model_versions'])

                # Show current version
                current_version = predictor.model_version if hasattr(predictor, 'model_version') else "N/A"
                st.info(f"Current model version: {current_version}")

                # Version history table
                st.write("Version History:")
                if 'timestamp' in versions_df.columns:
                    st.dataframe(versions_df.sort_values('timestamp', ascending=False))
                else:
                    st.warning("No timestamp information available in version history")

                # Version comparison
                if len(st.session_state['model_versions']) > 1:
                    st.subheader("Version Comparison")

                    col1, col2 = st.columns(2)
                    with col1:
                        version1 = st.selectbox("Select first version",
                                              versions_df['version'],
                                              index=0,
                                              key="version1_select")
                    with col2:
                        version2 = st.selectbox("Select second version",
                                              versions_df['version'],
                                              index=min(1, len(versions_df)-1),
                                              key="version2_select")

                    if version1 and version2:
                        perf1 = versions_df[versions_df['version'] == version1]['performance'].iloc[0]
                        perf2 = versions_df[versions_df['version'] == version2]['performance'].iloc[0]

                        comparison_df = pd.DataFrame({
                            'Metric': ['Accuracy', 'Balanced Accuracy', 'F1 Score'],
                            version1: [perf1['accuracy'], perf1.get('balanced_accuracy', 'N/A'), perf1.get('f1_score', 'N/A')],
                            version2: [perf2['accuracy'], perf2.get('balanced_accuracy', 'N/A'), perf2.get('f1_score', 'N/A')]
                        }).set_index('Metric')

                        st.dataframe(comparison_df.style.highlight_max(axis=1))
            else:
                st.warning("No model versions available")

            # Enhanced auto-retraining options
            st.subheader("Auto-Retraining Configuration")

            col1, col2 = st.columns(2)

            with col1:
                retrain_freq = st.selectbox(
                    "Retraining Frequency",
                    ["Manual", "Weekly", "Monthly", "Quarterly"],
                    help="How often to check for data drift and retrain",
                    key="retrain_freq_select"
                )

                drift_threshold = st.slider(
                    "Drift Detection Sensitivity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.05,
                    step=0.01,
                    help="P-value threshold for statistical drift detection",
                    key="drift_threshold_slider"
                )

            with col2:
                if st.button("Run Drift Detection Now", key="drift_detect_btn"):
                    if not predictor.df.empty:
                        # For demo, compare with a sample of the existing data
                        sample_df = predictor.df.sample(frac=0.3)
                        drift_detected, drift_report = predictor.check_for_drift(sample_df)

                        if drift_detected:
                            st.warning("Data drift detected!")
                            with st.expander("View Detailed Drift Report", expanded=False):
                                st.json(drift_report)

                            if st.button("Retrain Model Now", key="retrain_now_btn"):
                                if predictor.auto_retrain(sample_df):
                                    st.success("Retraining completed successfully!")
                        else:
                            st.success("No significant data drift detected")
                    else:
                        st.warning("No training data available for comparison")

            # Enhanced model reporting
            st.subheader("Model Reporting")

            if st.button("Generate Full Report", key="generate_report_btn"):
                report = predictor.generate_report()

                if report is None:
                    st.warning("No model information available to generate report")
                else:
                    # Display report in tabs
                    tab1, tab2, tab3 = st.tabs(["Summary", "Performance", "Technical Details"])

                    with tab1:
                        st.subheader("Model Summary")
                        st.write(f"**Model Type:** {report['model_info']['type']}")
                        st.write(f"**Version:** {report['model_info']['version']}")
                        st.write(f"**Training Time:** {report['model_info']['training_time']}")
                        st.write(f"**Classes:** {', '.join(report['model_info']['classes'])}")

                        if report.get('training_data'):
                            st.subheader("Training Data Summary")
                            st.write(f"**Samples:** {report['training_data']['num_samples']}")
                            st.write("**Class Distribution:**")
                            st.write(report['training_data']['class_distribution'])

                    with tab2:
                        st.subheader("Performance Metrics")
                        st.write(f"**Accuracy:** {report['performance']['accuracy']:.2%}")
                        st.write(f"**Balanced Accuracy:** {report['performance'].get('balanced_accuracy', 'N/A')}")
                        st.write(f"**F1 Score:** {report['performance'].get('f1_score', 'N/A')}")

                        st.subheader("Classification Report")
                        st.dataframe(pd.DataFrame(report['performance']['classification_report']).T)

                        st.subheader("Confusion Matrix")
                        cm = np.array(report['performance']['confusion_matrix'])
                        fig = ff.create_annotated_heatmap(
                            cm,
                            x=report['model_info']['classes'],
                            y=report['model_info']['classes'],
                            colorscale='Blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with tab3:
                        st.subheader("Feature Importance")
                        if report.get('feature_importance'):
                            fi_df = pd.DataFrame(report['feature_importance'])
                            st.dataframe(fi_df)
                        else:
                            st.warning("Feature importance not available")

                        st.subheader("Technical Details")
                        st.json(report['model_info'])

                    # Enhanced download options
                    st.subheader("Export Report")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # JSON download
                        json_report = json.dumps(report, indent=2)
                        st.download_button(
                            label="Download JSON Report",
                            data=json_report,
                            file_name="model_report.json",
                            mime="application/json"
                        )

                    with col2:
                        # HTML report
                        if st.button("Generate HTML Report", key="generate_html_btn"):
                            try:
                                from jinja2 import Template
                                template = Template("""
                                <html>
                                <head>
                                    <title>Model Report - {{ report.model_info.version }}</title>
                                    <style>
                                        body { font-family: Arial, sans-serif; margin: 20px; }
                                        h1 { color: #2c3e50; }
                                        h2 { color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 5px; }
                                        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                                        th { background-color: #f2f2f2; }
                                        .metric { background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }
                                    </style>
                                </head>
                                <body>
                                    <h1>Model Report - Version {{ report.model_info.version }}</h1>
                                    <p>Generated on {{ now }}</p>

                                    <h2>Model Information</h2>
                                    <div class="metric">
                                        <p><strong>Type:</strong> {{ report.model_info.type }}</p>
                                        <p><strong>Training Time:</strong> {{ report.model_info.training_time }}</p>
                                        <p><strong>Classes:</strong> {{ report.model_info.classes|join(', ') }}</p>
                                    </div>

                                    <h2>Performance Metrics</h2>
                                    <div class="metric">
                                        <p><strong>Accuracy:</strong> {{ "%.2f"|format(report.performance.accuracy * 100) }}%</p>
                                        <p><strong>Balanced Accuracy:</strong> {{ "%.2f"|format(report.performance.balanced_accuracy * 100) if report.performance.balanced_accuracy else 'N/A' }}%</p>
                                        <p><strong>F1 Score:</strong> {{ "%.2f"|format(report.performance.f1_score * 100) if report.performance.f1_score else 'N/A' }}%</p>
                                    </div>

                                    <h2>Feature Importance</h2>
                                    {% if report.feature_importance %}
                                        <table>
                                            <tr>
                                                <th>Feature</th>
                                                <th>Importance</th>
                                            </tr>
                                            {% for item in report.feature_importance %}
                                            <tr>
                                                <td>{{ item.Feature }}</td>
                                                <td>{{ "%.4f"|format(item.Importance) }}</td>
                                            </tr>
                                            {% endfor %}
                                        </table>
                                    {% else %}
                                        <p>Feature importance not available</p>
                                    {% endif %}
                                </body>
                                </html>
                                """)

                                html = template.render(
                                    report=report,
                                    now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                )

                                st.download_button(
                                    label="Download HTML Report",
                                    data=html,
                                    file_name="model_report.html",
                                    mime="text/html"
                                )
                            except Exception as e:
                                st.error(f"Failed to generate HTML report: {str(e)}")

                    with col3:
                        # PDF report (placeholder)
                        if st.button("Generate PDF Report", key="generate_pdf_btn"):
                            st.warning("PDF generation would be implemented with a proper PDF library")

            # Enhanced model monitoring
            st.subheader("Performance Monitoring")

            if st.session_state['performance_history']:
                history_df = pd.DataFrame(st.session_state['performance_history'])

                # Calculate performance metrics
                metrics = ['accuracy', 'balanced_accuracy', 'f1_score']

                col1, col2, col3 = st.columns(3)
                for i, metric in enumerate(metrics):
                    with [col1, col2, col3][i]:
                        if metric in history_df.columns:
                            last_value = history_df.iloc[-1][metric]
                            prev_value = history_df.iloc[-2][metric] if len(history_df) > 1 else None

                            delta = (last_value - prev_value) if prev_value is not None else None
                            delta_str = f"{delta:.2%}" if delta is not None else None

                            st.metric(
                                metric.replace('_', ' ').title(),
                                f"{last_value:.2%}",
                                delta=delta_str,
                                delta_color="inverse" if delta and delta < 0 else "normal"
                            )

                # Detailed performance history
                with st.expander("View Performance History", expanded=False):
                    st.dataframe(history_df.sort_values('timestamp', ascending=False))

                    # Export performance history
                    csv = history_df.to_csv(index=False)
                    st.download_button(
                        label="Export Performance History",
                        data=csv,
                        file_name="performance_history.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No performance history available")

elif st.session_state.active_tab == "👥 Member List":
    with st.container():
        st.header("Member List with Loss Ratio Categories")

        if not hasattr(predictor, 'model') and not st.session_state.get('model_loaded'):
            st.warning("⚠️ Please train or load a model first")
        else:
            if st.session_state.get('model_loaded'):
                predictor = st.session_state['predictor']

            if not predictor.df.empty and 'Loss_Ratio_Category' in predictor.df.columns:
                # Get member list with risk categories
                member_df = predictor.get_member_list()

                # Add filtering options
                st.subheader("Filter Members")

                col1, col2, col3 = st.columns(3)

                with col1:
                    risk_filter = st.multiselect(
                        "Filter by Risk Category",
                        options=predictor.class_labels,
                        default=predictor.class_labels,
                        key="risk_filter"
                    )

                with col2:
                    policy_filter = st.multiselect(
                        "Filter by Policy Type",
                        options=predictor.df['Policy_Type'].unique(),
                        default=predictor.df['Policy_Type'].unique(),
                        key="policy_filter"
                    )

                with col3:
                    age_range = st.slider(
                        "Filter by Age Range",
                        min_value=int(predictor.df['Age'].min()),
                        max_value=int(predictor.df['Age'].max()),
                        value=(int(predictor.df['Age'].min()), int(predictor.df['Age'].max())),
                        key="age_filter"
                    )

                # Apply filters
                filtered_df = member_df[
                    (member_df['Loss_Ratio_Category'].isin(risk_filter)) &
                    (member_df['Policy_Type'].isin(policy_filter)) &
                    (member_df['Age'].between(age_range[0], age_range[1]))
                ]

                # Show summary statistics
                st.subheader("Summary Statistics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Members", len(filtered_df))

                with col2:
                    avg_age = filtered_df['Age'].mean()
                    st.metric("Average Age", f"{avg_age:.1f} years")

                with col3:
                    risk_dist = filtered_df['Loss_Ratio_Category'].value_counts(normalize=True)
                    st.metric("Highest Risk Group",
                            f"{risk_dist.idxmax()} ({risk_dist.max():.1%})")

                # Show the member table
                st.subheader("Member Details")

                # Pagination
                page_size = st.selectbox("Rows per page", [10, 25, 50, 100], key="page_size")
                total_pages = (len(filtered_df) // page_size) + 1

                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="page_num")
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size

                # Display the paginated data
                st.dataframe(
                    filtered_df.iloc[start_idx:end_idx].style.apply(
                        lambda x: ['background: #e6f3ff' if x.name % 2 == 0 else '' for i in x],
                        axis=1
                    ),
                    height=(page_size + 1) * 35 + 3,
                    use_container_width=True
                )

                # Download options
                st.subheader("Export Data")

                col1, col2 = st.columns(2)

                with col1:
                    # CSV download
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="member_list.csv",
                        mime="text/csv"
                    )

                with col2:
                    # Excel download
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        filtered_df.to_excel(writer, index=False, sheet_name='Member List')
                    st.download_button(
                        label="Download as Excel",
                        data=output.getvalue(),
                        file_name="member_list.xlsx",
                        mime="application/vnd.ms-excel"
                    )

                # Visualizations
                st.subheader("Risk Distribution Visualizations")

                tab1, tab2 = st.tabs(["Risk Categories", "Demographics"])

                with tab1:
                    # Risk category distribution
                    fig = px.pie(
                        filtered_df,
                        names='Loss_Ratio_Category',
                        title='Distribution of Risk Categories',
                        color='Loss_Ratio_Category',
                        color_discrete_map={
                            'Low': 'green',
                            'Medium': 'blue',
                            'High': 'orange',
                            'Very High': 'red'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    # Age distribution by risk category
                    fig = px.box(
                        filtered_df,
                        x='Loss_Ratio_Category',
                        y='Age',
                        color='Loss_Ratio_Category',
                        title='Age Distribution by Risk Category',
                        color_discrete_map={
                            'Low': 'green',
                            'Medium': 'blue',
                            'High': 'orange',
                            'Very High': 'red'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Gender distribution
                    fig = px.histogram(
                        filtered_df,
                        x='Gender',
                        color='Loss_Ratio_Category',
                        barmode='group',
                        title='Gender Distribution by Risk Category',
                        color_discrete_map={
                            'Low': 'green',
                            'Medium': 'blue',
                            'High': 'orange',
                            'Very High': 'red'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No member data available. Please load or generate data first.")

# Enhanced model loading option
st.sidebar.header("Model Management")
uploaded_model = st.sidebar.file_uploader("Upload trained model", type=['pkl'], key="model_uploader")
if uploaded_model:
    try:
        loaded = joblib.load(uploaded_model)
        if verify_model(loaded):
            if isinstance(loaded, dict):  # New format
                predictor.model = loaded['model']
                predictor.preprocessor = loaded['preprocessor']
                predictor.label_encoder = loaded['label_encoder']
                if 'metadata' in loaded:
                    predictor.class_labels = loaded['metadata'].get('class_labels', predictor.class_labels)
                    predictor.training_time = loaded['metadata'].get('training_time')
                    predictor.performance_metrics = loaded['metadata'].get('performance_metrics', {})
                    predictor.model_version = loaded['metadata'].get('version')

                    # Add to version history if not already present
                    if predictor.model_version and predictor.model_version not in [v['version'] for v in st.session_state['model_versions']]:
                        st.session_state['model_versions'].append({
                            'version': predictor.model_version,
                            'timestamp': predictor.training_time.isoformat() if predictor.training_time else datetime.now().isoformat(),
                            'performance': predictor.performance_metrics
                        })
            else:  # Old format
                predictor.model = loaded

            st.session_state['model_loaded'] = True
            st.session_state['predictor'] = predictor
            st.sidebar.success("✅ Model loaded successfully!")

            # Display model info
            st.sidebar.markdown("### Model Information")
            if hasattr(predictor, 'training_time') and predictor.training_time:
                st.sidebar.write(f"Trained: {predictor.training_time.strftime('%Y-%m-%d %H:%M')}")
            st.sidebar.write(f"Classes: {', '.join(predictor.class_labels)}")
            if hasattr(predictor, 'model_version') and predictor.model_version:
                st.sidebar.write(f"Version: {predictor.model_version}")

            # Add to performance history
            if hasattr(predictor, 'performance_metrics') and predictor.performance_metrics:
                st.session_state['performance_history'].append({
                    'model': str(type(predictor.model)),
                    'accuracy': predictor.performance_metrics.get('accuracy', 0),
                    'balanced_accuracy': predictor.performance_metrics.get('balanced_accuracy', 0),
                    'f1_score': predictor.performance_metrics.get('f1_score', 0),
                    'timestamp': datetime.now().isoformat()
                })
        else:
            st.sidebar.error("Uploaded file is not a valid model")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")

# Model version selector
if st.session_state.get('model_versions'):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Versions")

    version_to_load = st.sidebar.selectbox(
        "Select version to load",
        [v['version'] for v in st.session_state['model_versions']],
        index=len(st.session_state['model_versions']) - 1,
        key="version_selector"
    )

    if st.sidebar.button("Load Selected Version", key="load_version_btn"):
        # In a real app, we would load the model from storage
        st.sidebar.warning("In a full implementation, this would load the selected model version")

# Display current model status
st.sidebar.markdown("---")
if st.session_state.get('model_loaded'):
    st.sidebar.success("Model Ready for Predictions")
    if hasattr(predictor, 'training_time') and predictor.training_time:
        st.sidebar.write(f"Last trained: {predictor.training_time.strftime('%Y-%m-%d %H:%M')}")
    if hasattr(predictor, 'performance_metrics') and predictor.performance_metrics:
        st.sidebar.write(f"Accuracy: {predictor.performance_metrics.get('accuracy', 0):.2%}")
    if hasattr(predictor, 'model_version') and predictor.model_version:
        st.sidebar.write(f"Version: {predictor.model_version}")
else:
    st.sidebar.warning("No Model Loaded")

# Documentation link
st.sidebar.markdown("---")
st.sidebar.markdown("[📚 Documentation](#)")  # Would link to actual docs in production

# Version information in footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Version:** 1.1.0")
st.sidebar.markdown("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d"))

# Main content footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    padding: 10px;
    font-size: 0.8em;
}
</style>
<div class="footer">
    <p>© 2023 Insurance Analytics Platform | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></p>
</div>
""", unsafe_allow_html=True)

# Add some custom CSS to prevent overlap with footer
st.markdown("""
<style>
    .main > div {
        padding-bottom: 60px;
    }
</style>
""", unsafe_allow_html=True)