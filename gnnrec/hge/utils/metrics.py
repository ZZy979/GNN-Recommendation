import torch
from sklearn.metrics import f1_score


def accuracy(predict, labels, evaluator=None):
    """计算准确率

    :param predict: tensor(N) 预测标签
    :param labels: tensor(N) 正确标签
    :param evaluator: ogb.nodeproppred.Evaluator
    :return: float 准确率
    """
    if evaluator is not None:
        y_true, y_pred = labels.unsqueeze(dim=1), predict.unsqueeze(dim=1)
        return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})['acc']
    else:
        return torch.sum(predict == labels).item() / labels.shape[0]


def macro_f1_score(predict, labels):
    """计算Macro-F1得分

    :param predict: tensor(N) 预测标签
    :param labels: tensor(N) 正确标签
    :return: float Macro-F1得分
    """
    return f1_score(labels.numpy(), predict.long().numpy(), average='macro')


@torch.no_grad()
def evaluate(
        model, loader, g, labels, num_classes, predict_ntype,
        train_idx, val_idx, test_idx, evaluator=None):
    """评估模型性能

    :param model: nn.Module GNN模型
    :param loader: NodeDataLoader 图数据加载器
    :param g: DGLGraph 图
    :param labels: tensor(N) 顶点标签
    :param num_classes: int 类别数
    :param predict_ntype: str 目标顶点类型
    :param train_idx: tensor(N_train) 训练集顶点id
    :param val_idx: tensor(N_val) 验证集顶点id
    :param test_idx: tensor(N_test) 测试集顶点id
    :param evaluator: ogb.nodeproppred.Evaluator
    :return: train_acc, val_acc, test_acc, train_f1, val_f1, test_f1
    """
    model.eval()
    logits = torch.zeros(g.num_nodes(predict_ntype), num_classes, device=train_idx.device)
    for input_nodes, output_nodes, blocks in loader:
        logits[output_nodes[predict_ntype]] = model(blocks, blocks[0].srcdata['feat'])
    return calc_metrics(logits, labels, train_idx, val_idx, test_idx, evaluator)


@torch.no_grad()
def evaluate_full(model, g, labels, train_idx, val_idx, test_idx):
    """评估模型性能(full-batch)

    :param model: nn.Module GNN模型
    :param g: DGLGraph 图
    :param labels: tensor(N) 顶点标签
    :param train_idx: tensor(N_train) 训练集顶点id
    :param val_idx: tensor(N_val) 验证集顶点id
    :param test_idx: tensor(N_test) 测试集顶点id
    :return: train_acc, val_acc, test_acc, train_f1, val_f1, test_f1
    """
    model.eval()
    logits = model(g, g.ndata['feat'])
    return calc_metrics(logits, labels, train_idx, val_idx, test_idx)


def calc_metrics(logits, labels, train_idx, val_idx, test_idx, evaluator=None):
    predict = logits.detach().cpu().argmax(dim=1)
    labels = labels.cpu()
    train_acc = accuracy(predict[train_idx], labels[train_idx], evaluator)
    val_acc = accuracy(predict[val_idx], labels[val_idx], evaluator)
    test_acc = accuracy(predict[test_idx], labels[test_idx], evaluator)
    train_f1 = macro_f1_score(predict[train_idx], labels[train_idx])
    val_f1 = macro_f1_score(predict[val_idx], labels[val_idx])
    test_f1 = macro_f1_score(predict[test_idx], labels[test_idx])
    return train_acc, val_acc, test_acc, train_f1, val_f1, test_f1


METRICS_STR = ' | '.join(
    f'{split} {metric} {{:.4f}}'
    for metric in ('Acc', 'Macro-F1') for split in ('Train', 'Val', 'Test')
)
