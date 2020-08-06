def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    true_positive = 0.0
    true_negative = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for i in range(prediction.shape[0]):
        if prediction[i] and ground_truth[i]:
            true_positive += 1.0
        elif prediction[i] and not ground_truth[i]:
            false_positive += 1.0
        elif not prediction[i] and not ground_truth[i]:
            true_negative += 1.0
        elif not prediction[i] and ground_truth[i]:
            false_negative += 1.0
    precision = true_positive / (false_positive+true_positive)
    recall = true_positive / (true_positive+false_negative)
    accuracy = (true_positive + true_negative) / prediction.shape[0]
    f1 = 2*precision*recall / (precision+recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    positive_count = 0.0
    for i in range(prediction.shape[0]):
        if prediction[i] == ground_truth[i]:
            positive_count += 1.0
    return positive_count/prediction.shape[0]
