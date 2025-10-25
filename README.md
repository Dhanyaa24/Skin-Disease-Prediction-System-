# Skin-Disease-Prediction-System-
Skin Disease Prediction System
A machine learning-based diagnostic tool for classifying dermatological conditions using clinical and histopathological features. Built with scikit-learn and Streamlit.
Overview
This project implements an AI-powered assessment system that predicts six common skin diseases with 98.61% accuracy. The system uses an SVM classifier trained on the UCI Dermatology Dataset and provides an intuitive web interface for symptom assessment.
Diseases Detected

Psoriasis
Seborrheic dermatitis
Lichen planus
Pityriasis rosea
Chronic dermatitis
Pityriasis rubra pilaris

Features

Interactive Questionnaire: User-friendly symptom assessment with 34 clinical and histopathological features
High Accuracy: 98.61% classification accuracy using optimized SVM model
Optional Lab Results: Supports both clinical-only and comprehensive (with biopsy) assessments
Detailed Information: Provides disease descriptions, care tips, and trusted medical resources
Smart Validation: Detects minimal symptoms to avoid false diagnoses

Tech Stack

Machine Learning: scikit-learn (SVM, Random Forest)
Frontend: Streamlit
Data Processing: pandas, numpy
Model Persistence: joblib

Model Performance

Algorithm: Support Vector Machine (SVM) with RBF kernel
Accuracy: 98.61%
Cross-Validation: 10-fold CV with mean score of 96.83%
Features: 34 clinical and histopathological attributes
Dataset: UCI Dermatology Dataset (358 instances)

Project Structure
├── SDP.ipynb                    # Model training notebook
├── app.py                       # Streamlit web application
├── skin_disease_model.pkl       # Trained SVM model
├── scaler.pkl                   # Feature scaler
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
Installation
bash# Clone repository
git clone https://github.com/yourusername/skin-disease-prediction.git
cd skin-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
Usage

Answer clinical symptom questions (age, redness, scaling, itching, etc.)
Optionally add lab/biopsy results for improved accuracy
View predicted condition with care recommendations
Access trusted medical resources for further information

Model Training
The model uses:

Train-Test Split: 80-20 with stratification
Feature Scaling: StandardScaler normalization
Hyperparameter Tuning: GridSearchCV with 5-fold cross-validation
Class Imbalance Handling: Balanced class weights

Key hyperparameters:

C: 1.0
Gamma: scale
Kernel: RBF
Class weight: balanced

Important Disclaimer
This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for proper medical care.
Dataset

Source: UCI Machine Learning Repository - Dermatology Dataset
Instances: 358 patients
Features: 34 attributes (clinical + histopathological)
Classes: 6 skin disease types

Future Enhancements

 Add image-based diagnosis using CNN
 Implement SHAP for model interpretability
 Support multiple languages
 Add patient history tracking
 Deploy to cloud platform (Heroku/AWS)
 Integrate with medical databases

Acknowledgments

UCI Machine Learning Repository for the Dermatology Dataset
scikit-learn and Streamlit communities
Medical professionals who contributed to dataset annotation

Note: This is an academic project demonstrating machine learning applications in healthcare. Clinical validation would be required before any real-world medical use.
