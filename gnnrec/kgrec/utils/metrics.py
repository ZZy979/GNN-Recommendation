def precision_at_k(y_true, y_pred, k):
    """计算Precision@k = TP@k / (TP@k + FP@k) = TP@k / k

    :param y_true: List[int] 真实相关结果id
    :param y_pred: List[int] 预测结果id
    :param k: int 只考虑前k个预测结果
    :return: float Precision@k
    """
    return len(set(y_true) & set(y_pred[:k])) / k


def recall_at_k(y_true, y_pred, k):
    """计算Recall@k = TP@k / (TP@k + FN@k) = TP@k / P, P为真实相关结果个数

    :param y_true: List[int] 真实相关结果id
    :param y_pred: List[int] 预测结果id
    :param k: int 只考虑前k个预测结果
    :return: float Recall@k
    """
    return len(set(y_true) & set(y_pred[:k])) / len(y_true)
