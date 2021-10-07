import torch


@torch.no_grad()
def accuracy(logits, labels, evaluator):
    """计算准确率

    :param logits: tensor(N, C) 预测概率
    :param labels: tensor(N) 正确标签
    :param evaluator: ogb.nodeproppred.Evaluator
    :return: float 准确率
    """
    if evaluator is not None:
        predict = logits.argmax(dim=1, keepdim=True)
        return evaluator.eval({'y_true': labels.unsqueeze(dim=1), 'y_pred': predict})['acc']
    else:
        return torch.sum(logits.argmax(dim=1) == labels).item() * 1.0 / len(labels)


@torch.no_grad()
def evaluate(
        model, loader, g, labels, num_classes, predict_ntype,
        train_idx, val_idx, test_idx, evaluator):
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
    :param evaluator: ogb.nodeproppred.Evaluator or None
    :return: train_acc, val_acc, test_acc
    """
    model.eval()
    logits = torch.zeros(g.num_nodes(predict_ntype), num_classes, device=train_idx.device)
    for input_nodes, output_nodes, blocks in loader:
        logits[output_nodes[predict_ntype]] = model(blocks, blocks[0].srcdata['feat'])
    train_acc = accuracy(logits[train_idx], labels[train_idx], evaluator)
    val_acc = accuracy(logits[val_idx], labels[val_idx], evaluator)
    test_acc = accuracy(logits[test_idx], labels[test_idx], evaluator)
    return train_acc, val_acc, test_acc
