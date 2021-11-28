import numpy as np
from sklearn.metrics import ndcg_score

__all__ = ['precision_at_k', 'recall_at_k', 'K', 'calc_metrics', 'METRICS_STR']


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


K = [5, 10, 20, 50, 100]  # nDCG@k和Recall@k中k的取值


def calc_metrics(field_ids, author_rank, true_relevance, predict_rank):
    """计算学者排名的评价指标

    :param field_ids: List[int] 领域id列表
    :param author_rank: Dict[int, List[int]] {field_id: [paper_id]} 真实学者排名
    :param true_relevance: ndarray(N_field, N_author) 领域-学者真实相关性得分
    :param predict_rank: Dict[int, (tensor(n), tensor(n)] {field_id: ([author_id, score])} 预测排名得分
    :return: Dict[int, float], Dict[int, float] {k: nDCG@k}, {k: Recall@k}
    """
    ndcg_scores, recall_scores = {k: [] for k in K}, {k: [] for k in K}
    for i, f in enumerate(field_ids):
        aid, score = predict_rank[f]
        pred_aid = aid[score.argsort(descending=True)]
        y_true = true_relevance[i, aid][np.newaxis]
        y_score = score.numpy()[np.newaxis]
        for k in K:
            ndcg_scores[k].append(ndcg_score(y_true, y_score, k=k, ignore_ties=True))
            recall_scores[k].append(recall_at_k(author_rank[f], pred_aid.tolist(), k))
    return {k: sum(s) / len(s) for k, s in ndcg_scores.items()}, \
           {k: sum(s) / len(s) for k, s in recall_scores.items()}


METRICS_STR = ' | '.join(f'nDCG@{k}={{0[{k}]:.4f}}' for k in K) + '\n' \
              + ' | '.join(f'Recall@{k}={{1[{k}]:.4f}}' for k in K)
