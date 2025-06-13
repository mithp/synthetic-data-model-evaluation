# Synthetic Data Generation and Model Evaluation

## Overview

This repository contains code for generating synthetic data for a two-class classification problem and evaluating the performance of machine learning models using custom fold splits. The dataset is designed to simulate different qualities of features, including independent features, nested pairs, features with decreasing discriminatory ability, and noise features. The repository also includes scripts for training models, performing cross-validation, and visualizing the results.

## Features

- **Data Generation**: Generate synthetic data with specified mean values and covariance matrices for different feature blocks.
- **Custom Fold Splitting**: Use StratifiedKFold to ensure balanced class proportions in each fold and assign fold labels for custom splitting.
- **Model Training**: Train machine learning models (e.g., Logistic Regression, XGBoost) on the generated data.
- **Model Evaluation**: Evaluate model performance using metrics such as AUC (Area Under the Curve) and accuracy.
- **Cross-Validation**: Perform K-Fold cross-validation to assess model performance across multiple folds.
- **Visualization**: Create subplots to visualize accuracy scores and AUC scores with a horizontal line indicating theoretical accuracy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mithp/synthetic-data-model-evaluation.git
   ```

2. Navigate to the project directory:
   ```bash
   cd synthetic-data-model-evaluation
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Generation

Generate synthetic data with a specified number of samples per class:
```python
num_samples_per_class = 500
X, y = generate_data(num_samples_per_class)
```

### Custom Fold Splitting

Assign fold labels to each sample and generate unique combinations of test folds:
```python
n_folds = 10
n_repeats = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
fold_labels = np.zeros(len(y))
for fold, (_, test_index) in enumerate(skf.split(X, y)):
    fold_labels[test_index] = fold + 1
```

### Model Training and Evaluation

Train a model and evaluate its performance using custom fold splits:
```python
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
accuracy = accuracy_score(y_test, y_pred)
```

### Visualization

Create subplots to visualize accuracy scores and AUC scores with a horizontal line at theoretical accuracy:
```python
fig, axs = plt.subplots(2, 1, figsize=(10, 12))
axs[0].boxplot([acc_scores[i] for i in range(1, n_folds)], labels=[str(i) for i in range(1, n_folds)])
axs[0].axhline(y=theoretical_accuracy, color='r', linestyle='--', label='Theoretical Accuracy')
axs[0].set_xlabel('Number of Test Folds')
axs[0].set_ylabel('Accuracy')
axs[0].set_title('Accuracy Scores vs Number of Test Folds')
axs[0].legend()
axs[1].boxplot([auc_scores[i] for i in range(1, n_folds)], labels=[str(i) for i in range(1, n_folds)])
axs[1].axhline(y=theoretical_accuracy, color='r', linestyle='--', label='Theoretical Accuracy')
axs[1].set_xlabel('Number of Test Folds')
axs[1].set_ylabel('AUC')
axs[1].set_title('AUC vs Number of Test Folds')
axs[1].legend()
plt.tight_layout()
plt.show()
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
