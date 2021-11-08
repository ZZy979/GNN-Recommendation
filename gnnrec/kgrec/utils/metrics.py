def precision_at_k(y_true, y_pred, k):
    y_true = set(y_true)
    y_pred = set(y_pred[:k])
    return len(set(y_true & y_pred)) / k


def recall_at_k(y_true, y_pred, k):
    y_true = set(y_true)
    y_pred = set(y_pred[:k])
    return len(set(y_true & y_pred)) / len(y_true)
