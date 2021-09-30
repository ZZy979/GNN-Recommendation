import dgl
import dgl.function as fn
import torch
from dgl.utils import extract_node_subframes, set_new_frames
from gensim.models import Word2Vec
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def load_ogbn_mag(path, add_reverse_edge=False, device=None, reverse_self=True):
    """加载ogbn-mag数据集

    :param path: str 数据集所在目录
    :param add_reverse_edge: bool, optional 是否添加反向边，默认为False
    :param device: torch.device, optional 将图和数据移动到指定的设备上，默认为CPU
    :param reverse_self: bool, optional 起点和终点类型相同时是否添加反向边，默认为True
    :return: g, features, labels, num_classes, train_idx, val_idx, test_idx, evaluator
    """
    if device is None:
        device = torch.device('cpu')
    data = DglNodePropPredDataset('ogbn-mag', path)
    g, labels = data[0]
    if add_reverse_edge:
        g = add_reverse_edges(g, reverse_self)
    g = g.to(device)
    features = g.nodes['paper'].data['feat']
    labels = labels['paper'].to(device)
    split_idx = data.get_idx_split()
    train_idx = split_idx['train']['paper'].to(device)
    val_idx = split_idx['valid']['paper'].to(device)
    test_idx = split_idx['test']['paper'].to(device)
    return g, features, labels, data.num_classes, train_idx, val_idx, test_idx, Evaluator(data.name)


def add_reverse_edges(g, reverse_self=True):
    """给异构图的每种边添加反向边，返回新的异构图

    :param g: DGLGraph 异构图
    :param reverse_self: bool, optional 起点和终点类型相同时是否添加反向边，默认为True
    :return: DGLGraph 添加反向边之后的异构图
    """
    data = {}
    for stype, etype, dtype in g.canonical_etypes:
        u, v = g.edges(etype=(stype, etype, dtype))
        data[(stype, etype, dtype)] = u, v
        if stype != dtype or reverse_self:
            data[(dtype, etype + '_rev', stype)] = v, u
    new_g = dgl.heterograph(data, {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    node_frames = extract_node_subframes(g, None)
    set_new_frames(new_g, node_frames=node_frames)
    return new_g


def average_node_feat(g):
    """ogbn-mag数据集没有输入特征的顶点取邻居平均"""
    message_func, resuce_func = fn.copy_u('feat', 'm'), fn.mean('m', 'feat')
    g.multi_update_all({
        'writes_rev': (message_func, resuce_func),
        'has_topic': (message_func, resuce_func)
    }, 'sum')
    g.multi_update_all({'affiliated_with': (message_func, resuce_func)}, 'sum')


def load_pretrained_node_embed(g, path):
    """ogbn-mag数据集没有输入特征的顶点加载预训练的顶点特征"""
    model = Word2Vec.load(path)
    for ntype in ('author', 'field_of_study', 'institution'):
        g.nodes[ntype].data['feat'] = torch.from_numpy(
            model.wv[[f'{ntype}_{i}' for i in range(g.num_nodes(ntype))]]
        ).to(g.device)
