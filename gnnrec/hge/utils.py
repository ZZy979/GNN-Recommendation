import random

import dgl
import numpy as np
import torch
from dgl.utils import extract_node_subframes, extract_edge_subframes, set_new_frames
from ogb.nodeproppred import DglNodePropPredDataset


def set_random_seed(seed):
    """设置Python, numpy, PyTorch的随机数种子

    :param seed: int 随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device):
    """返回指定的GPU设备

    :param device: int GPU编号，-1表示CPU
    :return: torch.device
    """
    return torch.device(f'cuda:{device}' if device >= 0 and torch.cuda.is_available() else 'cpu')


def load_ogbn_mag(path, add_reverse_edge=False, device=None):
    """加载ogbn-mag数据集

    :param path: str 数据集所在目录
    :param add_reverse_edge: bool, optional 是否添加反向边，默认为False
    :param device: torch.device, optional 将图和数据移动到指定的设备上，默认为CPU
    :return: dataset, g, features, labels, train_idx, val_idx, test_idx
    """
    if device is None:
        device = torch.device('cpu')
    data = DglNodePropPredDataset('ogbn-mag', path)
    g, labels = data[0]
    if add_reverse_edge:
        g = add_reverse_edges(g)
    g = g.to(device)
    features = g.nodes['paper'].data['feat']
    labels = labels['paper'].to(device)
    split_idx = data.get_idx_split()
    train_idx = split_idx['train']['paper'].to(device)
    val_idx = split_idx['valid']['paper'].to(device)
    test_idx = split_idx['test']['paper'].to(device)
    return data, g, features, labels, train_idx, val_idx, test_idx


def add_reverse_edges(g):
    """给异构图的每种边添加反向边，返回新的异构图

    :param g: DGLGraph 异构图
    :return: DGLGraph 添加反向边之后的异构图
    """
    data = {}
    for stype, etype, dtype in g.canonical_etypes:
        u, v = g.edges(etype=(stype, etype, dtype))
        data[(stype, etype, dtype)] = u, v
        data[(dtype, etype + '_rev', stype)] = v, u
    new_g = dgl.heterograph(data, {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    node_frames = extract_node_subframes(g, None)
    edge_frames = extract_edge_subframes(g, None)
    set_new_frames(new_g, node_frames=node_frames, edge_frames=edge_frames)
    return new_g


def accuracy(logits, labels, evaluator):
    """计算准确率

    :param logits: tensor(N, C) 预测概率
    :param labels: tensor(N, 1) 正确标签
    :param evaluator: ogb.nodeproppred.Evaluator
    :return: float 准确率
    """
    predict = logits.argmax(dim=1, keepdim=True)
    return evaluator.eval({'y_true': labels, 'y_pred': predict})['acc']
