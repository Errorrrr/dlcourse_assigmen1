import time

import numpy as np
import matplotlib.pyplot as plt
from dataset import load_svhn
from knn import KNN
from metrics import binary_classification_metrics, multiclass_accuracy
import timeit

train_X, train_y, test_X, test_y = load_svhn("data", max_train=1000, max_test=100)
# samples_per_class = 5  # Number of samples per class to visualize
# plot_index = 1
# for example_index in range(samples_per_class):
#     for class_index in range(10):
#         plt.subplot(5, 10, plot_index)
#         image = train_X[train_y == class_index][example_index]
#         plt.imshow(image.astype(np.uint8))
#         plt.axis('off')
#         plot_index += 1

# First, let's prepare the labels and the source data

# Only select 0s and 9s
#binary_train_mask = (train_y == 0) | (train_y == 9)
#binary_train_X = train_X[binary_train_mask]
#binary_train_y = train_y[binary_train_mask] == 0

# binary_test_mask = (test_y == 0) | (test_y == 9)
# binary_test_X = test_X[binary_test_mask]
# binary_test_y = test_y[binary_test_mask] == 0

# Reshape to 1-dimensional array [num_samples, 32*32*3]
#binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
#binary_test_X = binary_test_X.reshape(binary_test_X.shape[0], -1)

#knn_classifier = KNN(k=1)
#knn_classifier.fit(binary_train_X, binary_train_y)

#dists = knn_classifier.compute_distances_two_loops(binary_test_X)
#assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

#dists = knn_classifier.compute_distances_one_loop(binary_test_X)
#assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

#dists = knn_classifier.compute_distances_no_loops(binary_test_X)
#assert np.isclose(dists[0, 10], np.sum(np.abs(binary_test_X[0] - binary_train_X[10])))

#%%

# Lets look at the performance difference
#start_time = time.time()
#knn_classifier.compute_distances_two_loops(binary_test_X)
#print("--- %s seconds ---" % (time.time() - start_time))
#start_time = time.time()
#knn_classifier.compute_distances_one_loop(binary_test_X)
#print("--- %s seconds ---" % (time.time() - start_time))
#start_time = time.time()
#knn_classifier.compute_distances_no_loops(binary_test_X)
#print("--- %s seconds ---" % (time.time() - start_time))

#prediction = knn_classifier.predict(binary_test_X)

#precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
#print("KNN with k = %s" % knn_classifier.k)
#print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))
#%%

# Let's put everything together and run KNN with k=3 and see how we do
#knn_classifier_3 = KNN(k=3)
#knn_classifier_3.fit(binary_train_X, binary_train_y)
#prediction = knn_classifier_3.predict(binary_test_X)

#precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
#print("KNN with k = %s" % knn_classifier_3.k)
#print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))



# Find the best k using cross-validation based on F1 score
# num_folds = 11
#
#
# binary_train_mask = (train_y == 0) | (train_y == 9)
# binary_train_X = train_X[binary_train_mask]
# binary_train_y = train_y[binary_train_mask] == 0
# binary_train_X = binary_train_X.reshape(binary_train_X.shape[0], -1)
#
# # to (11,11,3072)
# train_folds_X = binary_train_X.reshape(num_folds, binary_train_X.shape[0]/num_folds, binary_train_X.shape[1])
# train_folds_y = binary_train_y.reshape(num_folds, binary_train_y.shape[0]/num_folds, 1)
# k_choices = [1, 2, 4, 3, 5, 8, 10, 15, 20, 25, 50]
# k_to_f1 = {}  # dict mapping k values to mean F1 scores (int -> float)
#
# for ka in k_choices:
#     avg_f1 = 0.0
#     for i in range(num_folds):
#         valid = train_folds_X[i]
#         valid_y = train_folds_y[i]
#         train = np.delete(train_folds_X, i, 0).reshape(num_folds*(num_folds-1), train_folds_X.shape[2])
#         train_y = np.delete(train_folds_y, i, 0).reshape(num_folds*(num_folds-1), 1)
#
#         knn_classifier_3 = KNN(k=ka)
#         knn_classifier_3.fit(train, train_y)
#         prediction = knn_classifier_3.predict(valid)
#
#         precision, recall, f1, accuracy = binary_classification_metrics(prediction, valid_y)
#         avg_f1 += f1
#     avg_f1 = avg_f1/num_folds
#     k_to_f1[ka] = avg_f1
#
# for k in sorted(k_to_f1):
#     print('k = %d, f1 = %f' % (k, k_to_f1[k]))
#
# best_k = 2
#
# best_knn_classifier = KNN(k=best_k)
# best_knn_classifier.fit(binary_train_X, binary_train_y)
# prediction = best_knn_classifier.predict(binary_test_X)
#
# precision, recall, f1, accuracy = binary_classification_metrics(prediction, binary_test_y)
# print("Best KNN with k = %s" % best_k)
# print("Accuracy: %4.2f, Precision: %4.2f, Recall: %4.2f, F1: %4.2f" % (accuracy, precision, recall, f1))


train_X = train_X.reshape(train_X.shape[0], -1)
test_X = test_X.reshape(test_X.shape[0], -1)

# knn_classifier = KNN(k=5)
# knn_classifier.fit(train_X, train_y)
#
# predict = knn_classifier.predict(test_X)
#
# accuracy = multiclass_accuracy(predict, test_y)
# print("Accuracy: %4.2f" % accuracy)
#
# # Find the best k using cross-validation based on accuracy
# num_folds = 100
#
# train_folds_X = train_X.reshape(num_folds, train_X.shape[0]/num_folds, train_X.shape[1])
# train_folds_y = train_y.reshape(num_folds, train_y.shape[0]/num_folds, 1)
#
# k_choices = [1, 2, 3, 5, 8, 10, 15, 20, 25, 50]
# k_to_accuracy = {}
#
# for ka in k_choices:
#     avg_acc = 0.0
#     for i in range(num_folds):
#         valid = train_folds_X[i]
#         valid_y = train_folds_y[i]
#         train = np.delete(train_folds_X, i, 0).reshape(train_folds_X.shape[1]*(num_folds-1), train_folds_X.shape[2])
#         train_y = np.delete(train_folds_y, i, 0).reshape(train_folds_X.shape[1]*(num_folds-1), 1)
#
#         knn_classifier_3 = KNN(k=ka)
#         knn_classifier_3.fit(train, train_y)
#         prediction = knn_classifier_3.predict(valid)
#
#         precision, recall, f1, accuracy = binary_classification_metrics(prediction, valid_y)
#         avg_acc += accuracy
#     avg_acc = avg_acc/num_folds
#     k_to_accuracy[ka] = avg_acc
#
# for k in sorted(k_to_accuracy):
#     print('k = %d, accuracy = %f' % (k, k_to_accuracy[k]))

best_k = 2

best_knn_classifier = KNN(k=best_k)
best_knn_classifier.fit(train_X, train_y)
prediction = best_knn_classifier.predict(test_X)

# Accuracy should be around 20%!
accuracy = multiclass_accuracy(prediction, test_y)
print("Accuracy: %4.2f" % accuracy)