"""
Dataset Description:
---------------------

The synthetic dataset used in this project is generated following the methodology described in:

Křížek, P., Kittler, J., & Hlaváč, V. (2007). Improving Stability of Feature Selection Methods.
In: Kropatsch, W.G., Kampel, M., Hanbury, A. (eds) Computer Analysis of Images and Patterns.
CAIP 2007. Lecture Notes in Computer Science, vol 4673. Springer, Berlin, Heidelberg.
DOI: https://doi.org/10.1007/978-3-540-74272-2_115

The data consists of samples from a 20-dimensional normal distribution with structured mean vectors
and covariance matrices to simulate various feature qualities, including:
- Independent informative features
- Correlated feature pairs
- Features with decreasing discriminative power
- Pure noise features

This design ensures controlled complexity and is intended to challenge feature selection algorithms.

Modeling Process:
-----------------
1. Data Generation:
The dataset is generated with a specified number of samples per class, ensuring a balanced two-class problem.

2. Custom Fold Split:
StratifiedKFold is used to ensure that each fold has the same proportion of class labels.
Fold labels are assigned to each sample for custom fold splitting.

3. Generating Test Combinations:
Unique combinations of test folds are generated to evaluate the model's performance across different test set sizes.

4. Storing AUC/ACC Scores:
AUC and accuracy scores are stored for each combination of test folds.

5. Repeating the Experiment:
The experiment is repeated multiple times to assess the variance in accuracy as the number of test folds increases.
For each repeat, the model is trained on one fold and tested on various combinations of the remaining folds.

6. Plotting Results:
A boxplot is created to visualize the accuracy scores for different numbers of test folds.
A horizontal line is drawn at the theoretical accuracy to compare the achieved accuracy with the expected performance.

@author: mithp
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import itertools
import matplotlib.pyplot as plt

# Define the data generation function
def generate_data(num_samples_per_class):
    # Define the parameters for the data generation
    mu_1_3 = np.array([0.635, 0.635, 0.635])
    sigma_1_3 = np.identity(3)
    
    mu_4_6 = np.array([0.5, 0.4, 0])
    sigma_4_6 = np.array([[1.05, 0.48, 0.95],
                          [0.48, 1.00, 0.20],
                          [0.95, 0.20, 1.05]])
    
    mu_7_13 = np.array([0.636, 0.546, 0.455, 0.364, 0.273, 0.182, 0.091])
    sigma_7_13 = np.identity(7)
    
    mu_14_20 = np.zeros(7)
    sigma_14_20 = np.identity(7)
    
    # Generate samples for each block
    block_1_3 = np.random.multivariate_normal(mu_1_3, sigma_1_3, num_samples_per_class)
    block_4_6 = np.random.multivariate_normal(mu_4_6, sigma_4_6, num_samples_per_class)
    block_7_13 = np.random.multivariate_normal(mu_7_13, sigma_7_13, num_samples_per_class)
    block_14_20 = np.random.multivariate_normal(mu_14_20, sigma_14_20, num_samples_per_class)
    
    # Combine all blocks to form the dataset
    X_class1 = np.hstack((block_1_3, block_4_6, block_7_13, block_14_20))
    
    # Generate samples for the second class (negative mean values)
    X_class2 = -X_class1
    
    # Combine both classes to form the final dataset
    X = np.vstack((X_class1, X_class2))
    
    # Generate labels for the classes (binary labels: 0 and 1)
    y = np.hstack((np.ones(num_samples_per_class), np.zeros(num_samples_per_class)))
    
    return X, y

# Number of samples per class
num_samples_per_class = 500

# Generate the data
X, y = generate_data(num_samples_per_class)

# Number of folds
n_folds = 5
n_repeats = 5

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Assign fold numbers to each sample
fold_labels = np.zeros(len(y))
for fold, (_, test_index) in enumerate(skf.split(X, y)):
    fold_labels[test_index] = fold + 1

# Function to generate unique combinations of test folds
def generate_test_combinations(folds):
    combinations = []
    for i in range(1, len(folds) + 1):
        combinations.extend(itertools.combinations(folds, i))
    return combinations

# Store AUC/ACC scores
auc_scores = {i: [] for i in range(1, n_folds)}
acc_scores = {i: [] for i in range(1, n_folds)}

# Repeat the experiment
for repeat in range(n_repeats):
    for train_fold in range(1, n_folds + 1):
        test_folds = [i for i in range(1, n_folds + 1) if i != train_fold]
        test_combinations = generate_test_combinations(test_folds)
        
        for test_comb in test_combinations:
            test_indices = np.hstack([np.where(fold_labels == fold)[0] for fold in test_comb])
            train_indices = np.where(fold_labels == train_fold)[0]
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            # Train the model
            model = LogisticRegression()
            model.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            
            # Compute AUC
            auc = roc_auc_score(y_test, y_pred_prob)
            auc_scores[len(test_comb)].append(auc)
            
            # Predict labels for the test set
            y_pred = model.predict(X_test)
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            acc_scores[len(test_comb)].append(accuracy)

# Theoretical classification accuracy from the paper: https://doi.org/10.1007/978-3-540-74272-2_115
theoretical_accuracy = 0.977

# Plotting the accuracy scores and AUC scores with a horizontal line at the theoretical accuracy
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Accuracy subplot
axs[0].boxplot([acc_scores[i] for i in range(1, n_folds)], labels=[str(i) for i in range(1, n_folds)])
axs[0].axhline(y=theoretical_accuracy, color='r', linestyle='--', label='Theoretical Accuracy')
axs[0].set_xlabel('Number of Test Folds')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Accuracy Scores vs Number of Test Folds')
axs[0].legend()

# AUC subplot
axs[1].boxplot([auc_scores[i] for i in range(1, n_folds)], labels=[str(i) for i in range(1, n_folds)])
axs[1].set_xlabel('Number of Test Folds')
axs[1].set_ylabel('AUC')
axs[1].set_title('AUC vs Number of Test Folds')
axs[1].legend()

plt.tight_layout()
plt.show()
