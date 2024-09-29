import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
BreastCancerData = load_breast_cancer()

# Define X (independent variables) and y (dependent variables)
X = BreastCancerData.data
y = BreastCancerData.target 

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Import logistic regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Build a logstic regression object
LogReg = LogisticRegression(solver = 'newton-cg')

# Train(fit) model
LogReg.fit(X, y)

# Import cross-validate function from sckit-learn
from sklearn.model_selection import cross_validate
# Perform cross-validation with K=5 (cv=5) and "accuracy" as performance measure
cv_results = cross_validate(LogReg, X, y, cv=5, scoring ='accuracy')
# Store results
cv_scores = cv_results['test_score'] 
# Print cross-validation results
print("Cross-validation score for each of the folds: ", [float('{:.3f}'.format(x)) for x in cv_scores])
print("Mean cross-validation score (or cross-validation score): %0.3f (+/- %0.3f)" % (cv_scores.mean(), cv_scores.std() * 2))
