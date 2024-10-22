import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pymongo import MongoClient
import logging

logging.basicConfig(level=logging.DEBUG)

client = MongoClient('mongodb://localhost:27017/')
db = client['URLDATA2']
collection = db['url2_extracted']

data = list(collection.find({}, {'_id': 0}))
df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)  
features = df.drop(['url', 'label'], axis=1)
labels = df['label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

rf_model.fit(X_train_scaled, y_train)
logging.debug("Training Random Forest")


svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
logging.debug("Training SVM")


logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_scaled, y_train)
logging.debug("Trainingd Logistic Regression")


voting_model = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model), ('logreg', logreg_model)], voting='hard')
voting_model.fit(X_train_scaled, y_train)
logging.debug("Training Voting Classifier")


joblib.dump(rf_model, 'random_forest_model2.joblib')
joblib.dump(svm_model, 'svm_model2.joblib')
joblib.dump(logreg_model, 'logistic_regression_model2.joblib')
joblib.dump(voting_model, 'ensemblemodel2.joblib')
joblib.dump(scaler, 'scaler2.joblib')
logging.debug("Saved models and scaler")
