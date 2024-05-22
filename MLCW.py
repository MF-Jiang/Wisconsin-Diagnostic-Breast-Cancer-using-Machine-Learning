# Task 1
# Load data
import numpy as np

data_file = './wdbc.data'

data = np.genfromtxt(data_file, delimiter=',', dtype=str)
labels = data[:, 1]

# Replace 'M' with 1, 'B' with 0
labels = np.where(labels == 'M', 1, 0)

data = np.genfromtxt(data_file, delimiter=',')
features = data[:, 2:]

# Print to verify
print("Labels:", labels[:5])
print("Features:", features[:5])

#  Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=169, random_state=42)


print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Dimensionality reduction by PCA
from sklearn.decomposition import PCA

dimensions = [5, 10, 15, 20]

all_X_train_pca = []
all_X_test_pca = []

for dim in dimensions:
    pca = PCA(n_components=dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    all_X_train_pca.append(X_train_pca)
    all_X_test_pca.append(X_test_pca)

    print(f"\nDimension: {dim}")
    print("Explained Variance Ratio (first 3 components):", pca.explained_variance_ratio_[:3])
    print("Cumulative Explained Variance Ratio:", np.cumsum(pca.explained_variance_ratio_))
    print("Reduced Training Data Shape:", X_train_pca.shape)
    print("Reduced Testing Data Shape:", X_test_pca.shape)

all_X_train_pca.append(X_train)
all_X_test_pca.append(X_test)
dimensions.append(30)

# Task 2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Function to evaluate KNN models and return evaluation metrics
def evaluate_knn(X_train, X_test, y_train, y_test, k, dimension):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[:, :dimension], y_train)

    y_pred = knn.predict(X_test[:, :dimension])

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    return accuracy, precision, recall, f1


# Perform KNN with different feature dimensions and K values
results = {'dimension': [], 'k': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for i, dim in enumerate(dimensions):
    X_train_pca = all_X_train_pca[i]
    X_test_pca = all_X_test_pca[i]

    for k_value in [1, 3, 5, 7, 9]:
        accuracy, precision, recall, f1 = evaluate_knn(X_train_pca, X_test_pca, y_train, y_test, k_value, dim)

        # Save results
        results['dimension'].append(dim)
        results['k'].append(k_value)
        results['accuracy'].append(accuracy)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1'].append(f1)
        # print(results)

# Create a DataFrame for visualization
import pandas as pd

results_df = pd.DataFrame(results)
print(results_df)

import matplotlib.pyplot as plt

# Performance Variation with Different K Values
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics = ['accuracy', 'precision', 'recall', 'f1']

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]

    # Plot for each K value
    for k_value in [1, 3, 5, 7, 9]:
        k_results = results_df[results_df['k'] == k_value]
        ax.plot(k_results['dimension'], k_results[metric], label=f'K={k_value}')

    ax.set_title(metric.capitalize())
    ax.set_xlabel('Dimension')
    ax.set_ylabel(metric.capitalize())
    ax.legend()

plt.tight_layout()
plt.show()

# Performance Variation with Different Feature Dimensions
import matplotlib.pyplot as plt

# Plotting with matplotlib
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Define dimensions
dimensions = [5, 10, 15, 20, 30]

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]

    # Plot for each dimension
    for dimension in dimensions:
        dimension_results = results_df[results_df['dimension'] == dimension]
        ax.plot(dimension_results['k'], dimension_results[metric], label=f'Dimension={dimension}')

    ax.set_title(metric.capitalize())
    ax.set_xlabel('K')
    ax.set_ylabel(metric.capitalize())
    ax.legend()

plt.tight_layout()
plt.show()

# Task 3
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

# Define the combination of hyperparameters to be tested
layers = [1, 2, 3, 4]
nodes = [50, 100, 150]
learning_rates = [0.001, 0.01, 0.1]
alphas = [0, 0.0001, 0.001, 0.01]
# dimensions = [5, 10, 15, 20, 30]
dimensions = [5]
solvers = ['adam']

results_mlp = {'layers': [], 'nodes': [], 'learning_rate': [], 'alpha': [], 'dimension': [], 'solver': [],
               'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for dimension in dimensions:
    if dimension != 30:
        pca = PCA(n_components=dimension)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    else:
        X_train_pca = X_train
        X_test_pca = X_test
    for layer in layers:
        for node in nodes:
            for learning_rate in learning_rates:
                for alpha in alphas:
                    for solver in solvers:
                        hidden_layer = tuple([node] * layer)
                        mlp = MLPClassifier(random_state=42)
                        mlp.set_params(hidden_layer_sizes=hidden_layer, learning_rate_init=learning_rate,
                                       alpha=alpha, solver=solver)

                        # Use 5-fold cross-validation to train and validate the model
                        cross_val_scores = cross_val_score(mlp, X_train_pca, y_train, cv=5, scoring='accuracy')
                        avg_accuracy = cross_val_scores.mean()

                        # Record results
                        results_mlp['layers'].append(layer)
                        results_mlp['nodes'].append(node)
                        results_mlp['learning_rate'].append(learning_rate)
                        results_mlp['alpha'].append(alpha)
                        results_mlp['dimension'].append(dimension)
                        results_mlp['solver'].append(solver)
                        results_mlp['accuracy'].append(avg_accuracy)

                        # Train the final model (on the entire training set)
                        mlp.fit(X_train_pca, y_train)

                        # Evaluate the model on the test set
                        y_pred = mlp.predict(X_test_pca)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)

                        # Record test set results
                        results_mlp['precision'].append(precision)
                        results_mlp['recall'].append(recall)
                        results_mlp['f1'].append(f1)

# Organize the result into a DataFrame or other appropriate format
import pandas as pd

results_mlp_df = pd.DataFrame(results_mlp)

# Find the index for the best parameter
best_index = results_mlp_df['accuracy'].idxmax()

# Print the best parameters
best_layers = results_mlp_df.loc[best_index, 'layers']
best_nodes = results_mlp_df.loc[best_index, 'nodes']
best_learning_rate = results_mlp_df.loc[best_index, 'learning_rate']
best_alpha = results_mlp_df.loc[best_index, 'alpha']
best_dimension = results_mlp_df.loc[best_index, 'dimension']
best_solver = results_mlp_df.loc[best_index, 'solver']
best_accuracy = results_mlp_df.loc[best_index, 'accuracy']

print(f'Best Layers: {best_layers}')
print(f'Best Nodes: {best_nodes}')
print(f'Best Learning Rate: {best_learning_rate}')
print(f'Best Alpha: {best_alpha}')
print(f'Best Dimension: {best_dimension}')
print(f'Best Solver: {best_solver}')
print(f'Best Accuracy: {best_accuracy}')

# Task 4
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import time

if best_dimension != 30:
    pca = PCA(n_components=best_dimension)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
else:
    X_train_pca = X_train
    X_test_pca = X_test

hidden_layer = tuple([best_nodes] * best_layers)

start_time = time.time()
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_pca, y_train)
knn_prediction = knn_model.predict(X_test_pca)
knn_time = time.time() - start_time
knn_train_prediction = knn_model.predict(X_train_pca)

start_time = time.time()
mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer, learning_rate_init=best_learning_rate, alpha=best_alpha,
                          solver=best_solver)
mlp_model.fit(X_train_pca, y_train)
mlp_prediction = mlp_model.predict(X_test_pca)
mlp_time = time.time() - start_time
mlp_train_prediction = mlp_model.predict(X_train_pca)

print(f'KNN Time: {knn_time}')
print(f'MLP Time: {mlp_time}')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

knn_accuracy = accuracy_score(y_test, knn_prediction)
print(f'KNN Accuracy: {knn_accuracy}')
mlp_accuracy = accuracy_score(y_test, mlp_prediction)
print(f'MLP Accuracy: {mlp_accuracy}')

knn_precision = precision_score(y_test, knn_prediction)
print(f'KNN Precision: {knn_precision}')
mlp_precision = precision_score(y_test, mlp_prediction)
print(f'MLP Precision: {mlp_precision}')

knn_recall = recall_score(y_test, knn_prediction)
print(f'KNN Recall: {knn_recall}')
mlp_recall = recall_score(y_test, mlp_prediction)
print(f'MLP Recall: {mlp_recall}')

knn_f1 = f1_score(y_test, knn_prediction)
print(f'KNN F1: {knn_f1}')
mlp_f1 = f1_score(y_test, mlp_prediction)
print(f'MLP F1: {mlp_f1}')

knn_train_accuracy = accuracy_score(y_train, knn_train_prediction)
knn_test_accuracy = accuracy_score(y_test, knn_prediction)

mlp_train_accuracy = accuracy_score(y_train, mlp_train_prediction)
mlp_test_accuracy = accuracy_score(y_test, mlp_prediction)

knn_overfitting_index = knn_train_accuracy - knn_test_accuracy
mlp_overfitting_index = mlp_train_accuracy - mlp_test_accuracy

print(f'KNN Overfitting Index: {knn_overfitting_index}')
print(f'MLP Overfitting Index: {mlp_overfitting_index}')

knn_overfitting_rate = knn_train_accuracy / knn_test_accuracy
mlp_overfitting_rate = mlp_train_accuracy / mlp_test_accuracy

print(f'KNN Overfitting Rate: {knn_overfitting_rate}')
print(f'MLP Overfitting Rate: {mlp_overfitting_rate}')

knn_overfitting_percentage = ((knn_train_accuracy - knn_test_accuracy) / knn_test_accuracy) * 100
mlp_overfitting_percentage = ((mlp_train_accuracy - mlp_test_accuracy) / mlp_test_accuracy) * 100

print(f'KNN Overfitting Percentage: {knn_overfitting_percentage}%')
print(f'MLP Overfitting Percentage: {mlp_overfitting_percentage}%')

