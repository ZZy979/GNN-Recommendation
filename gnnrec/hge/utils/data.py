import dgl
import dgl.function as fn
import torch
from dgl.utils import extract_node_subframes, set_new_frames
from gensim.models import Word2Vec
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from gnnrec.config import DATA_DIR
from gnnrec.kgrec.data import OAGCSVenueDataset


def load_data(name, device='cpu', add_reverse_edge=True, reverse_self=False):
    """加载数据集

    :param name: str 数据集名称
    :param device: torch.device, optional 将图和数据移动到指定的设备上，默认为CPU
    :param add_reverse_edge: bool, optional 是否添加反向边，默认为True
    :param reverse_self: bool, optional 起点和终点类型相同时是否添加反向边，默认为False
    :return: g, features, labels, num_classes, predict_ntype, train_mask, val_mask, test_mask, evaluator
    """
    if name == 'ogbn-mag':
        return load_ogbn_mag(device, add_reverse_edge, reverse_self)
    elif name == 'oag-cs':
        data = OAGCSVenueDataset()
        predict_ntype = 'paper'
    else:
        raise ValueError(f'未知数据集{name}')
    g = data[0]
    if add_reverse_edge:
        g = add_reverse_edges(g, reverse_self)
    g = g.to(device)
    features = g.nodes[predict_ntype].data['feat']
    labels = g.nodes[predict_ntype].data['label']
    train_mask = g.nodes[predict_ntype].data['train_mask'].nonzero(as_tuple=True)[0]
    val_mask = g.nodes[predict_ntype].data['val_mask'].nonzero(as_tuple=True)[0]
    test_mask = g.nodes[predict_ntype].data['test_mask'].nonzero(as_tuple=True)[0]
    return g, features, labels, data.num_classes, predict_ntype, train_mask, val_mask, test_mask, None


def load_ogbn_mag(device, add_reverse_edge, reverse_self):
    """加载ogbn-mag数据集

    :param device: torch.device 将图和数据移动到指定的设备上，默认为CPU
    :param add_reverse_edge: bool 是否添加反向边
    :param reverse_self: bool 起点和终点类型相同时是否添加反向边
    :return: g, features, labels, num_classes, predict_ntype, train_mask, val_mask, test_mask, evaluator
    """
    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g, labels = data[0]
    if add_reverse_edge:
        g = add_reverse_edges(g, reverse_self)
    g = g.to(device)
    features = g.nodes['paper'].data['feat']
    labels = labels['paper'].squeeze(dim=1).to(device)
    split_idx = data.get_idx_split()
    train_idx = split_idx['train']['paper'].to(device)
    val_idx = split_idx['valid']['paper'].to(device)
    test_idx = split_idx['test']['paper'].to(device)
    evaluator = Evaluator(data.name)
    return g, features, labels, data.num_classes, 'paper', train_idx, val_idx, test_idx, evaluator


def add_reverse_edges(g, reverse_self=False):
    """给异构图的每种边添加反向边，返回新的异构图

    :param g: DGLGraph 异构图
    :param reverse_self: bool, optional 起点和终点类型相同时是否添加反向边，默认为False
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
    message_func, reduce_func = fn.copy_u('feat', 'm'), fn.mean('m', 'feat')
    g.multi_update_all({
        'writes_rev': (message_func, reduce_func),
        'has_topic': (message_func, reduce_func)
    }, 'sum')
    g.multi_update_all({'affiliated_with': (message_func, reduce_func)}, 'sum')


def load_pretrained_node_embed(g, word2vec_path, concat=False):
    """为没有输入特征的顶点加载预训练的顶点特征

    :param g: DGLGraph 异构图
    :param word2vec_path: str 预训练的word2vec模型路径
    :param concat: bool, optional 如果为True则将预训练特征与原输入特征拼接
    """
    model = Word2Vec.load(word2vec_path)
    for ntype in g.ntypes:
        embed = torch.from_numpy(model.wv[[f'{ntype}_{i}' for i in range(g.num_nodes(ntype))]]) \
            .to(g.device)
        if 'feat' in g.nodes[ntype].data:
            if concat:
                g.nodes[ntype].data['feat'] = torch.cat([g.nodes[ntype].data['feat'], embed], dim=1)
        else:
            g.nodes[ntype].data['feat'] = embed
