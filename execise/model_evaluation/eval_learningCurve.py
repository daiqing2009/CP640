# import matplotlib package for plots
import matplotlib.pyplot as plt
# Import learning_curve function from scikit-learn package
from sklearn.model_selection import learning_curve
# Crete data for learning curve
train_sizes, train_scores, test_scores = learning_curve(LogReg, X, y, cv=5, scoring ='accuracy',
train_sizes=np.linspace(.1, 1.0, 10))
# Create mean of train and test scores
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# Plot learning curve lines (mean of training and test scores)
plt.plot(train_sizes, train_mean, '--',  label="Training score")
plt.plot(train_sizes, test_mean,  label="Cross-validation score")

# Add title and labels and show the plot
plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.legend(loc="best")
plt.tight_layout()
plt.show()