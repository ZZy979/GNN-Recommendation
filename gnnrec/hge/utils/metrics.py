import torch


@torch.no_grad()
def accuracy(logits, labels, evaluator):
    """计算准确率

    :param logits: tensor(N, C) 预测概率
    :param labels: tensor(N, 1) 正确标签
    :param evaluator: ogb.nodeproppred.Evaluator
    :return: float 准确率
    """
    predict = logits.argmax(dim=1, keepdim=True)
    return evaluator.eval({'y_true': labels, 'y_pred': predict})['acc']
