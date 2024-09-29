# Import validation_curve function from scikit-learn package
from sklearn.model_selection import validation_curve
# Define the range of parameter to be tested
param_range = np.arange(0.1,10,0.1)
# Calculate accuracy on training and test set using range of parameter values
train_scores, test_scores = validation_curve(LogReg, X, y, param_name="C", param_range=param_range, cv=5, scoring="accuracy")


# Calculate mean for training and test scores
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

# Plot validation curve lines (mean of training and test scores)
plt.plot(param_range, train_mean, '--',label="Training score")
plt.plot(param_range, test_mean, label="Cross-validation score")

# Add title and labels and show the plot
plt.title("Validation Curve")
plt.ylim([0.75, 1.0])
plt.xlabel("Value of regularization term")
plt.ylabel("Accuracy Score")
plt.tight_layout()
plt.show()