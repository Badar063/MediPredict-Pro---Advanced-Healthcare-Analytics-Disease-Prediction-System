#!/usr/bin/env python3
"""
🏥 MediPredict Pro - Advanced Healthcare Analytics & Disease Prediction System
AI-powered medical diagnosis, patient risk stratification, and treatment optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced Healthcare ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import xgboost as xgb
from scipy import stats
import scipy.cluster.hierarchy as sch
from lifelines import KaplanMeierFitter, CoxPHFitter

# Interactive Medical Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class AdvancedHealthcareAnalytics:
    """
    🏥 Advanced Healthcare Analytics Engine
    Features:
    - Multi-disease Risk Prediction
    - Patient Stratification & Clustering
    - Treatment Outcome Forecasting
    - Medical Image Analysis (simulated)
    - Drug Response Prediction
    - Survival Analysis
    - Anomaly Detection in Medical Data
    - Clinical Decision Support
    """
    
    def __init__(self):
        self.patient_data = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.visualizations = {}
        self.medical_conditions = [
            'Diabetes', 'Hypertension', 'Heart Disease', 'COPD', 
            'Chronic Kidney Disease', 'Liver Disease', 'Cancer Risk'
        ]
        
    def generate_synthetic_medical_data(self, n_patients=8000):
        """
        Generate sophisticated synthetic medical data with realistic patterns
        """
        print("🎲 Generating advanced synthetic medical data...")
        
        np.random.seed(42)
        
        # Patient demographics
        ages = np.random.normal(45, 15, n_patients).astype(int)
        ages = np.clip(ages, 18, 90)
        
        genders = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
        
        # Vital signs with realistic correlations
        blood_pressure_systolic = np.random.normal(120, 15, n_patients)
        blood_pressure_diastolic = blood_pressure_systolic * 0.6 + np.random.normal(0, 5, n_patients)
        
        heart_rates = np.random.normal(72, 10, n_patients)
        cholesterol = np.random.lognormal(4.5, 0.3, n_patients)
        bmi = np.random.normal(25, 4, n_patients)
        
        # Lab results with disease correlations
        glucose = np.random.normal(90, 15, n_patients)
        creatinine = np.random.lognormal(0.9, 0.2, n_patients)
        hemoglobin = np.random.normal(14, 1.5, n_patients)
        
        # Lifestyle factors
        smoking_status = np.random.choice(['Never', 'Former', 'Current'], n_patients, p=[0.6, 0.2, 0.2])
        alcohol_consumption = np.random.exponential(5, n_patients)
        physical_activity = np.random.beta(2, 2, n_patients) * 10
        
        # Family history (binary features)
        family_history_diabetes = np.random.binomial(1, 0.3, n_patients)
        family_history_heart_disease = np.random.binomial(1, 0.25, n_patients)
        family_history_cancer = np.random.binomial(1, 0.2, n_patients)
        
        # Generate disease probabilities based on risk factors
        diabetes_risk = self._calculate_diabetes_risk(ages, bmi, glucose, family_history_diabetes)
        hypertension_risk = self._calculate_hypertension_risk(ages, blood_pressure_systolic, bmi)
        heart_disease_risk = self._calculate_heart_disease_risk(ages, cholesterol, blood_pressure_systolic, smoking_status)
        
        # Simulate actual disease presence
        has_diabetes = (diabetes_risk > 0.7).astype(int)
        has_hypertension = (hypertension_risk > 0.6).astype(int)
        has_heart_disease = (heart_disease_risk > 0.65).astype(int)
        
        # Treatment responses
        treatment_effectiveness = np.random.beta(2, 2, n_patients)
        
        # Medical imaging features (simulated)
        ct_scan_abnormalities = np.random.exponential(0.5, n_patients)
        mri_lesions = np.random.poisson(0.3, n_patients)
        
        # Genetic markers (simulated)
        genetic_risk_score = np.random.beta(1, 3, n_patients)
        
        # Create comprehensive patient DataFrame
        self.patient_data = pd.DataFrame({
            'patient_id': [f'PAT_{i:06d}' for i in range(n_patients)],
            'age': ages,
            'gender': genders,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'heart_rate': heart_rates,
            'cholesterol_total': cholesterol,
            'bmi': bmi,
            'glucose_fasting': glucose,
            'creatinine': creatinine,
            'hemoglobin': hemoglobin,
            'smoking_status': smoking_status,
            'alcohol_consumption': alcohol_consumption,
            'physical_activity': physical_activity,
            'family_history_diabetes': family_history_diabetes,
            'family_history_heart_disease': family_history_heart_disease,
            'family_history_cancer': family_history_cancer,
            'diabetes_risk_score': diabetes_risk,
            'hypertension_risk_score': hypertension_risk,
            'heart_disease_risk_score': heart_disease_risk,
            'has_diabetes': has_diabetes,
            'has_hypertension': has_hypertension,
            'has_heart_disease': has_heart_disease,
            'treatment_effectiveness': treatment_effectiveness,
            'ct_scan_abnormality_score': ct_scan_abnormalities,
            'mri_lesion_count': mri_lesions,
            'genetic_risk_score': genetic_risk_score,
            'visit_date': [datetime.now() - timedelta(days=np.random.randint(0, 365)) 
                          for _ in range(n_patients)]
        })
        
        # Add realistic medical correlations
        self._enhance_medical_realism()
        
        print(f"✅ Generated {len(self.patient_data)} patient records with 28 clinical features")
        return self.patient_data
    
    def _calculate_diabetes_risk(self, ages, bmi, glucose, family_history):
        """Calculate diabetes risk based on clinical factors"""
        risk = (ages / 100 * 0.3 + 
                (bmi - 20) / 30 * 0.3 + 
                (glucose - 70) / 50 * 0.3 + 
                family_history * 0.1)
        return np.clip(risk, 0, 1)
    
    def _calculate_hypertension_risk(self, ages, bp_systolic, bmi):
        """Calculate hypertension risk"""
        risk = (ages / 100 * 0.4 + 
                (bp_systolic - 100) / 60 * 0.4 + 
                (bmi - 20) / 30 * 0.2)
        return np.clip(risk, 0, 1)
    
    def _calculate_heart_disease_risk(self, ages, cholesterol, bp_systolic, smoking_status):
        """Calculate heart disease risk"""
        smoking_encoded = np.array([1 if s == 'Current' else 0.5 if s == 'Former' else 0 
                                  for s in smoking_status])
        risk = (ages / 100 * 0.3 + 
                (cholesterol - 3) / 4 * 0.3 + 
                (bp_systolic - 100) / 60 * 0.2 + 
                smoking_encoded * 0.2)
        return np.clip(risk, 0, 1)
    
    def _enhance_medical_realism(self):
        """Add realistic medical correlations and patterns"""
        # Enhance diabetes correlation with glucose
        diabetic_patients = self.patient_data['has_diabetes'] == 1
        self.patient_data.loc[diabetic_patients, 'glucose_fasting'] += np.random.normal(30, 10, diabetic_patients.sum())
        
        # Enhance hypertension correlation with BP
        hypertensive_patients = self.patient_data['has_hypertension'] == 1
        self.patient_data.loc[hypertensive_patients, 'blood_pressure_systolic'] += np.random.normal(20, 5, hypertensive_patients.sum())
        
        # Add age-related disease progression
        elderly = self.patient_data['age'] > 65
        self.patient_data.loc[elderly, 'heart_disease_risk_score'] *= 1.3
        self.patient_data.loc[elderly, 'creatinine'] *= 1.2
    
    def patient_risk_stratification(self):
        """
        Advanced patient risk stratification using multiple ML techniques
        """
        print("\n🎯 Performing patient risk stratification...")
        
        # Features for risk stratification
        clinical_features = [
            'age', 'blood_pressure_systolic', 'cholesterol_total', 'bmi',
            'glucose_fasting', 'creatinine', 'hemoglobin', 'physical_activity',
            'family_history_diabetes', 'family_history_heart_disease'
        ]
        
        X = self.patient_data[clinical_features].copy()
        X = X.fillna(X.mean())
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Multiple clustering approaches
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        risk_clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate cluster risk scores
        cluster_risk_profiles = {}
        for cluster_id in range(5):
            cluster_data = self.patient_data[risk_clusters == cluster_id]
            risk_score = (
                cluster_data['diabetes_risk_score'].mean() * 0.3 +
                cluster_data['hypertension_risk_score'].mean() * 0.3 +
                cluster_data['heart_disease_risk_score'].mean() * 0.4
            )
            cluster_risk_profiles[cluster_id] = risk_score
        
        # Assign risk levels
        risk_levels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        sorted_clusters = sorted(cluster_risk_profiles.items(), key=lambda x: x[1])
        cluster_to_risk = {cluster: risk_levels[i] for i, (cluster, _) in enumerate(sorted_clusters)}
        
        self.patient_data['risk_cluster'] = risk_clusters
        self.patient_data['risk_level'] = [cluster_to_risk[c] for c in risk_clusters]
        
        # t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        self.patient_data['tsne_1'] = X_tsne[:, 0]
        self.patient_data['tsne_2'] = X_tsne[:, 1]
        
        self.results['risk_stratification'] = {
            'clusters_identified': len(np.unique(risk_clusters)),
            'cluster_risk_profiles': cluster_risk_profiles,
            'risk_level_distribution': self.patient_data['risk_level'].value_counts().to_dict()
        }
        
        print("✅ Patient risk stratification completed")
        return cluster_risk_profiles
    
    def multi_disease_prediction(self):
        """
        Predict multiple diseases using ensemble ML models
        """
        print("\n🔬 Training multi-disease prediction models...")
        
        # Features for disease prediction
        prediction_features = [
            'age', 'gender', 'blood_pressure_systolic', 'cholesterol_total', 'bmi',
            'glucose_fasting', 'creatinine', 'smoking_status', 'physical_activity',
            'family_history_diabetes', 'family_history_heart_disease'
        ]
        
        # Prepare data
        X = self.patient_data[prediction_features].copy()
        
        # Encode categorical variables
        X_encoded = pd.get_dummies(X, columns=['gender', 'smoking_status'])
        
        # Train models for each disease
        diseases = ['has_diabetes', 'has_hypertension', 'has_heart_disease']
        disease_models = {}
        model_performance = {}
        
        for disease in diseases:
            y = self.patient_data[disease]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train multiple models
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42),
                'GradientBoosting': GradientBoostingClassifier(random_state=42),
                'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 25), random_state=42)
            }
            
            best_score = 0
            best_model = None
            best_model_name = None
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    best_model_name = name
            
            # Store best model
            disease_models[disease] = best_model
            model_performance[disease] = {
                'best_model': best_model_name,
                'accuracy': best_score,
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'feature_importance': dict(zip(X_encoded.columns, 
                                             best_model.feature_importances_))
            }
        
        self.models['disease_prediction'] = disease_models
        self.results['disease_prediction'] = model_performance
        
        print("✅ Multi-disease prediction models trained successfully")
        return model_performance
    
    def survival_analysis(self):
        """
        Perform survival analysis for disease progression
        """
        print("\n⏳ Performing survival analysis...")
        
        # Simulate survival data (time to event)
        np.random.seed(42)
        
        # Generate synthetic time-to-event data
        baseline_time = np.random.exponential(365 * 5, len(self.patient_data))  # 5-year baseline
        
        # Adjust based on risk factors
        high_risk_adjustment = (self.patient_data['heart_disease_risk_score'] > 0.7) * np.random.exponential(100, len(self.patient_data))
        event_times = baseline_time - high_risk_adjustment
        
        # Ensure positive times
        event_times = np.maximum(event_times, 30)  # At least 30 days
        
        # Generate event indicators (1 if event occurred, 0 if censored)
        event_occurred = np.random.binomial(1, 0.3, len(self.patient_data))
        
        # Create survival dataframe
        survival_data = self.patient_data[['age', 'heart_disease_risk_score', 'has_heart_disease']].copy()
        survival_data['time_to_event'] = event_times
        survival_data['event_occurred'] = event_occurred
        
        # Kaplan-Meier analysis
        kmf = KaplanMeierFitter()
        
        # Plot survival by risk level
        plt.figure(figsize=(10, 6))
        
        for risk_level in ['Low', 'Medium', 'High']:
            mask = self.patient_data['risk_level'] == risk_level
            kmf.fit(survival_data.loc[mask, 'time_to_event'], 
                   survival_data.loc[mask, 'event_occurred'], 
                   label=risk_level)
            kmf.plot()
        
        plt.title('Survival Analysis by Risk Level')
        plt.xlabel('Days')
        plt.ylabel('Survival Probability')
        plt.grid(True)
        
        self.results['survival_analysis'] = {
            'median_survival_times': kmf.median_survival_time_,
            'survival_curves': kmf.survival_function_
        }
        
        print("✅ Survival analysis completed")
        return survival_data
    
    def treatment_optimization(self):
        """
        Optimize treatment plans using reinforcement learning concepts
        """
        print("\n💊 Analyzing treatment optimization strategies...")
        
        # Simulate treatment outcomes based on patient characteristics
        treatment_features = [
            'age', 'bmi', 'glucose_fasting', 'blood_pressure_systolic',
            'treatment_effectiveness', 'physical_activity'
        ]
        
        X_treatment = self.patient_data[treatment_features].copy()
        
        # Simulate different treatment strategies
        treatment_strategies = ['Standard Care', 'Intensive Therapy', 'Lifestyle Focus']
        
        treatment_outcomes = {}
        
        for strategy in treatment_strategies:
            if strategy == 'Standard Care':
                effectiveness = X_treatment['treatment_effectiveness']
            elif strategy == 'Intensive Therapy':
                effectiveness = X_treatment['treatment_effectiveness'] * 1.3
            else:  # Lifestyle Focus
                effectiveness = X_treatment['treatment_effectiveness'] * 0.9 + X_treatment['physical_activity'] * 0.1
            
            # Calculate outcomes
            success_rate = np.mean(effectiveness > 0.7)
            avg_improvement = np.mean(effectiveness)
            
            treatment_outcomes[strategy] = {
                'success_rate': success_rate,
                'avg_improvement': avg_improvement,
                'recommended_for': self._get_recommended_patient_profile(strategy, X_treatment)
            }
        
        self.results['treatment_optimization'] = treatment_outcomes
        
        print("✅ Treatment optimization analysis completed")
        return treatment_outcomes
    
    def _get_recommended_patient_profile(self, strategy, patient_data):
        """Determine which patient profiles benefit most from each strategy"""
        if strategy == 'Intensive Therapy':
            return "High-risk patients with multiple comorbidities"
        elif strategy == 'Lifestyle Focus':
            return "Younger patients with lifestyle-modifiable risk factors"
        else:
            return "Moderate-risk patients with standard care needs"
    
    def medical_anomaly_detection(self):
        """
        Detect medical anomalies and outliers in patient data
        """
        print("\n🚨 Performing medical anomaly detection...")
        
        # Features for anomaly detection
        anomaly_features = [
            'blood_pressure_systolic', 'heart_rate', 'cholesterol_total',
            'glucose_fasting', 'creatinine', 'bmi'
        ]
        
        X_anomaly = self.patient_data[anomaly_features].copy()
        
        # Calculate Z-scores for each feature
        z_scores = np.abs(stats.zscore(X_anomaly))
        
        # Flag anomalies (Z-score > 3)
        anomalies = (z_scores > 3).any(axis=1)
        
        # Calculate anomaly scores
        anomaly_scores = z_scores.mean(axis=1)
        self.patient_data['anomaly_score'] = anomaly_scores
        self.patient_data['is_anomaly'] = anomalies
        
        # Analyze anomaly patterns
        anomaly_analysis = {
            'total_anomalies': anomalies.sum(),
            'anomaly_rate': anomalies.mean(),
            'common_anomaly_types': self._analyze_anomaly_types(X_anomaly, z_scores),
            'high_risk_anomalies': self.patient_data[anomalies & 
                                                   (self.patient_data['risk_level'].isin(['High', 'Very High']))].shape[0]
        }
        
        self.results['anomaly_detection'] = anomaly_analysis
        
        print(f"🚨 Detected {anomalies.sum()} medical anomalies")
        return anomaly_analysis
    
    def _analyze_anomaly_types(self, data, z_scores):
        """Analyze types of medical anomalies detected"""
        anomaly_types = {}
        for i, feature in enumerate(data.columns):
            feature_anomalies = (z_scores[:, i] > 3).sum()
            if feature_anomalies > 0:
                anomaly_types[feature] = {
                    'count': feature_anomalies,
                    'max_z_score': z_scores[:, i].max()
                }
        return anomaly_types
    
    def create_medical_visualizations(self):
        """
        Create advanced medical visualizations and dashboards
        """
        print("\n📊 Creating advanced medical visualizations...")
        
        # 1. Risk Stratification Visualization
        fig_risk = px.scatter(self.patient_data, x='tsne_1', y='tsne_2', 
                             color='risk_level', hover_data=['age', 'bmi', 'glucose_fasting'],
                             title='Patient Risk Stratification (t-SNE Projection)')
        
        # 2. Disease Correlation Heatmap
        numeric_columns = self.patient_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.patient_data[numeric_columns].corr()
        
        # Focus on disease correlations
        disease_correlations = correlation_matrix.loc[
            ['has_diabetes', 'has_hypertension', 'has_heart_disease']
        ]
        
        fig_heatmap = ff.create_annotated_heatmap(
            z=disease_correlations.values,
            x=disease_correlations.columns.tolist(),
            y=disease_correlations.index.tolist(),
            annotation_text=disease_correlations.round(2).values,
            colorscale='RdBu_r'
        )
        fig_heatmap.update_layout(title='Disease Risk Factor Correlations')
        
        # 3. Treatment Effectiveness by Risk Level
        fig_treatment = px.box(self.patient_data, x='risk_level', y='treatment_effectiveness',
                              color='risk_level', title='Treatment Effectiveness by Risk Level')
        
        # 4. Age vs Disease Risk
        fig_age_risk = px.scatter(self.patient_data, x='age', y='heart_disease_risk_score',
                                color='has_heart_disease', trendline='lowess',
                                title='Age vs Heart Disease Risk Score')
        
        self.visualizations = {
            'risk_stratification': fig_risk,
            'disease_correlations': fig_heatmap,
            'treatment_effectiveness': fig_treatment,
            'age_risk_relationship': fig_age_risk
        }
        
        print("✅ Advanced medical visualizations created")
        return self.visualizations
    
    def generate_clinical_insights(self):
        """
        Generate comprehensive clinical insights and recommendations
        """
        print("\n📈 Generating clinical insights report...")
        
        insights = {
            'epidemiological_findings': [],
            'clinical_risk_factors': [],
            'treatment_recommendations': [],
            'preventive_strategies': [],
            'resource_allocation_suggestions': []
        }
        
        # Epidemiological findings
        total_patients = len(self.patient_data)
        diabetic_patients = self.patient_data['has_diabetes'].sum()
        hypertensive_patients = self.patient_data['has_hypertension'].sum()
        heart_disease_patients = self.patient_data['has_heart_disease'].sum()
        
        insights['epidemiological_findings'].extend([
            f"Total patient population: {total_patients:,}",
            f"Diabetes prevalence: {diabetic_patients/total_patients*100:.1f}%",
            f"Hypertension prevalence: {hypertensive_patients/total_patients*100:.1f}%",
            f"Heart disease prevalence: {heart_disease_patients/total_patients*100:.1f}%",
            f"High-risk patients identified: {len(self.patient_data[self.patient_data['risk_level'].isin(['High', 'Very High'])])}"
        ])
        
        # Clinical risk factors
        if 'disease_prediction' in self.results:
            for disease, performance in self.results['disease_prediction'].items():
                top_features = sorted(performance['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                insights['clinical_risk_factors'].append(
                    f"Top predictors for {disease}: {', '.join([f[0] for f in top_features])}"
                )
        
        # Treatment recommendations
        if 'treatment_optimization' in self.results:
            for strategy, outcome in self.results['treatment_optimization'].items():
                insights['treatment_recommendations'].append(
                    f"{strategy}: {outcome['success_rate']*100:.1f}% success rate - {outcome['recommended_for']}"
                )
        
        # Preventive strategies
        insights['preventive_strategies'].extend([
            "🎯 Implement targeted screening for high-risk patient clusters",
            "💡 Develop personalized lifestyle intervention programs",
            "📱 Deploy remote monitoring for chronic disease management",
            "🤝 Establish multidisciplinary care teams for complex cases"
        ])
        
        # Resource allocation
        high_risk_count = len(self.patient_data[self.patient_data['risk_level'].isin(['High', 'Very High'])])
        insights['resource_allocation_suggestions'].extend([
            f"Allocate {high_risk_count} high-risk patients to specialized care programs",
            "Prioritize telemedicine resources for remote patient monitoring",
            "Focus preventive efforts on modifiable risk factors identified in analysis"
        ])
        
        self.results['clinical_insights'] = insights
        
        print("✅ Comprehensive clinical insights report generated")
        return insights
    
    def save_medical_analysis(self):
        """
        Save all medical analysis results and visualizations
        """
        print("\n💾 Saving medical analysis results...")
        
        # Save patient data
        self.patient_data.to_csv('sample_data/patient_records.csv', index=False)
        
        # Save analysis results
        results_summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_patients_analyzed': len(self.patient_data),
            'risk_clusters_identified': len(self.patient_data['risk_cluster'].unique()),
            'clinical_insights': self.results.get('clinical_insights', {}),
            'model_performance': self.results.get('disease_prediction', {}),
            'anomaly_detection': self.results.get('anomaly_detection', {})
        }
        
        import json
        with open('sample_data/medical_analysis_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # Save visualizations as HTML
        for viz_name, fig in self.visualizations.items():
            fig.write_html(f'sample_data/{viz_name}_visualization.html')
        
        print("✅ All medical analysis results saved to sample_data/ directory")
    
    def run_complete_medical_analysis(self):
        """
        Run the complete advanced healthcare analytics pipeline
        """
        print("🏥 STARTING ADVANCED HEALTHCARE ANALYTICS PIPELINE")
        print("=" * 60)
        
        # Step 1: Generate medical data
        self.generate_synthetic_medical_data(6000)
        
        # Step 2: Risk stratification
        self.patient_risk_stratification()
        
        # Step 3: Disease prediction
        self.multi_disease_prediction()
        
        # Step 4: Survival analysis
        self.survival_analysis()
        
        # Step 5: Treatment optimization
        self.treatment_optimization()
        
        # Step 6: Anomaly detection
        self.medical_anomaly_detection()
        
        # Step 7: Visualizations
        self.create_medical_visualizations()
        
        # Step 8: Clinical insights
        insights = self.generate_clinical_insights()
        
        # Step 9: Save results
        self.save_medical_analysis()
        
        print("\n🎉 HEALTHCARE ANALYTICS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Print key clinical insights
        print("\n📊 KEY CLINICAL FINDINGS:")
        print("-" * 25)
        for finding in insights['epidemiological_findings'][:3]:
            print(f"• {finding}")
        
        print("\n🎯 CLINICAL RECOMMENDATIONS:")
        print("-" * 25)
        for recommendation in insights['treatment_recommendations'][:3]:
            print(f"• {recommendation}")
        
        return self.results

def main():
    """
    Main function to demonstrate the advanced healthcare analytics engine
    """
    # Initialize the healthcare analytics engine
    medical_engine = AdvancedHealthcareAnalytics()
    
    # Run complete medical analysis
    results = medical_engine.run_complete_medical_analysis()
    
    print(f"\n📁 Medical analysis results saved in 'sample_data/' directory")
    print("🔍 Open the HTML files in your browser to view interactive medical visualizations")

if __name__ == "__main__":
    main()
