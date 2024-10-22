import pandas as pd
import logging 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#initialize muna ang logging

logging.basicConfig(level=logging.DEBUG)

#file path
file_path ='Extracted_features\parsed.extracted.csv'
data = pd.read_csv(file_path)

features= data.drop(['url','label'], axis=1)
labels=data['label']
logging.debug("extract features and labels")

logging.basicConfig(level=logging.DEBUG)


#split
X_train,X_test,y_train,y_test = train_test_split(features,labels, test_size=0.2, random_state=50)
logging.debug("split")


#preprocess
scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
logging.debug("standardize the features")

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#train rf
rf_model = RandomForestClassifier(n_estimators=100,random_state=50)
rf_model.fit(X_train_scaled,y_train)
logging.debug("train random forest")



#train svm
svm_model=SVC(kernel='linear')
svm_model.fit(X_train_scaled,y_train)
logging.debug("training SVM")



#voting classifier
voting_model=VotingClassifier(estimators=[('rf',rf_model),('svm',svm_model)],voting='hard')
voting_model.fit=VotingClassifier(X_train_scaled,y_train)
logging.debug("Voting classfier for combined model")


joblib.dump(rf_model,'random_forest_model.joblib')
joblib.dump(svm_model,'svm_model.joblib')
joblib.dump(voting_model,'voting_classfier')


