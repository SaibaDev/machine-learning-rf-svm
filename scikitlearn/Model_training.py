#PROTEGO Random Forest & Support Vector Machine - Model Training

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)


file_path = 'Extracted_features\parsed_Fextracted1.csv' 
data = pd.read_csv(file_path)


features = data.drop(['url', 'label'], axis=1) 
labels = data['label']
logging.debug("#Extract features and labels") #DEBUG


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
logging.debug("# Split the data into training and testing sets")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) #XTEST
logging.debug("# Standardize the features (optional but often improves performance")


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
logging.debug(" Train Random Forest Classifier")


svm_model = SVC(kernel='linear')#SVC
svm_model.fit(X_train_scaled, y_train)
logging.debug("# Train Support Vector Machine (SVM) Classifier")


voting_model = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model)], voting='hard')
voting_model.fit=(X_train_scaled, y_train) #voting_model.fit


joblib.dump(rf_model, 'random_forest_model.joblib')
joblib.dump(svm_model, 'svm_model.joblib')
joblib.dump(voting_model, 'voting_model.joblib')
joblib.dump(scaler, 'scaler.joblib')  # semi colon

logging.debug("# Save the trained models and scaler to files")

