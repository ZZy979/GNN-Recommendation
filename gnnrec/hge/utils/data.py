import dgl
import torch
from dgl.utils import extract_node_subframes, set_new_frames
from ogb.nodeproppred import DglNodePropPredDataset


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
    set_new_frames(new_g, node_frames=node_frames)
    return new_g
