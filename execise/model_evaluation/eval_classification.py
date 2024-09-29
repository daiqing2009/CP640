import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
BreastCancerData = load_breast_cancer()

# Define X (independent variables) and y (dependent variables)
X = BreastCancerData.data
y = BreastCancerData.target

# Print size of the dataset
print("Breast cancer dataset dimensions: {}".format(X.shape))    

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

# Import logistic regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Build a logstic regression object
LogReg = LogisticRegression(solver = 'newton-cg')

# Train(fit) model
LogReg.fit(X_train, y_train)

# Predict the data
y_train_predict = LogReg.predict(X_train)
y_test_predict = LogReg.predict(X_test)

# Count percentage of correct predictions
print("The performance of the model:")
print("------------------------------")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print('Confusion matrix for training set')
print(confusion_matrix(y_train,y_train_predict))

print('Confusion matrix for test set')
print(confusion_matrix(y_test, y_test_predict),'\n')

# Accuracy Score
from sklearn.metrics import accuracy_score
print('Accuracy of the model for training set: %.3f' % accuracy_score(y_train,y_train_predict))
print('Accuracy of the model for test set: %.3f\n' % accuracy_score(y_test,y_test_predict))

# Precision score
from sklearn.metrics import precision_score
print('Precision of the model for training set: %.3f' % precision_score(y_train,y_train_predict))
print('Precision of the model for test set: %.3f\n' % precision_score(y_test,y_test_predict))

# Recall score
from sklearn.metrics import recall_score
print('Recall of the model for training set: %.3f' % recall_score(y_train,y_train_predict))
print('Recall of the model for test set: %.3f\n' % recall_score(y_test,y_test_predict))

# F1 Score
from sklearn.metrics import f1_score
print('F1-measure of the model for training set: %.3f' % f1_score(y_train,y_train_predict))
print('F1-measure of the model for test set: %.3f\n' % f1_score(y_test,y_test_predict))

# AUC score
from sklearn.metrics import roc_auc_score
print('AUC score of the model for training set: %.3f' % roc_auc_score(y_train,y_train_predict))
print('AUC score of the model for test set: %.3f\n' % roc_auc_score(y_test,y_test_predict))
      
# Log loss
print('Log loss of the model for training set: %.3f' % log_loss(y_train,y_train_predict))
print('Log loss of the model for test set: %.3f\n' % log_loss(y_test,y_test_predict))