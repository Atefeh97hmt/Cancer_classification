import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# خواندن داده‌ها
file = pd.read_excel('data_cancer.xlsx')
data = file.values[:, 2:].astype(float)
labels, labels_idx = np.unique(file.values[:, 1], return_inverse=True)
n_samples, n_features = data.shape
n_train_samples = int(0.7 * n_samples)
idx_random = np.random.permutation(n_samples)
data_train = data[idx_random[:n_train_samples]]
labels_train = labels_idx[idx_random[:n_train_samples]]
data_test = data[idx_random[n_train_samples:]]
labels_test = labels_idx[idx_random[n_train_samples:]]
n_test_samples = len(labels_test)
n_classes = len(np.unique(labels_idx))

# انتخاب ویژگی‌ها با استفاده از FDR
FDR = np.zeros(n_features)
for feature in range(n_features):
    mean_0 = np.mean(data_train[labels_train == 0, feature])
    var_0 = np.var(data_train[labels_train == 0, feature])
    mean_1 = np.mean(data_train[labels_train == 1, feature])
    var_1 = np.var(data_train[labels_train == 1, feature])
    FDR[feature] = (mean_0 - mean_1) ** 2 / (var_0 ** 2 + var_1 ** 2)
selected_features = FDR.argsort()[-5:]  # انتخاب 5 ویژگی برتر
data_train = data_train[:, selected_features]
data_test = data_test[:, selected_features]

# الگوریتم KNN
k_values = [1, 3, 5, 7, 9]
accuracies = []
for k in k_values:
    predictions = np.zeros(n_test_samples)
    for i in range(n_test_samples):
        distances = np.linalg.norm(data_test[i] - data_train, axis=1)
        nearest_neighbors = labels_train[np.argsort(distances)[:k]]
        predictions[i] = np.argmax(np.bincount(nearest_neighbors))
    accuracies.append(np.sum(predictions == labels_test) / n_test_samples)
best_k = k_values[np.argmax(accuracies)]
print('بهترین KNN:', accuracies[np.argmax(accuracies)])
plt.plot(k_values, accuracies, '*-')
plt.grid()
plt.xlabel('K')
plt.ylabel('دقت')
plt.show()

# min-Mean-Distance
centers = np.zeros((n_classes, data_train.shape[1]))
for i in range(n_classes):
    centers[i] = np.mean(data_train[labels_train == i], axis=0)
predictions_mmd = np.argmin(np.linalg.norm(data_test[:, None] - centers, axis=2), axis=1)
accuracy_mmd = np.sum(predictions_mmd == labels_test) / n_test_samples
print('دقت min-Mean-Distance:', accuracy_mmd)

# Bayes
means = np.zeros((n_classes, data_train.shape[1]))
covariances = []
priors = np.zeros(n_classes)
for i in range(n_classes):
    class_data = data_train[labels_train == i]
    means[i] = np.mean(class_data, axis=0)
    covariances.append(np.cov(class_data, rowvar=False))
    priors[i] = len(class_data) / n_train_samples
predictions_bayes = np.zeros(n_test_samples)
for i in range(n_test_samples):
    posteriors = np.zeros(n_classes)
    for j in range(n_classes):
        posteriors[j] = priors[j] * \
            multivariate_normal.pdf(data_test[i], mean=means[j], cov=covariances[j])
    predictions_bayes[i] = np.argmax(posteriors)
accuracy_bayes = np.sum(predictions_bayes == labels_test) / n_test_samples
print('دقت Bayes:', accuracy_bayes)
