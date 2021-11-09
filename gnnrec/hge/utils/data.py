import dgl
import dgl.function as fn
import torch
from gensim.models import Word2Vec
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from gnnrec.config import DATA_DIR
from gnnrec.hge.data import ACMDataset, DBLPDataset
from gnnrec.kgrec.data import OAGVenueDataset


def load_data(name, device='cpu', add_reverse_edge=True, reverse_self=True):
    """加载数据集

    :param name: str 数据集名称 acm, dblp, ogbn-mag, oag-venue
    :param device: torch.device, optional 将图和数据移动到指定的设备上，默认为CPU
    :param add_reverse_edge: bool, optional 是否添加反向边，默认为True
    :param reverse_self: bool, optional 起点和终点类型相同时是否添加反向边，默认为True
    :return: dataset, g, features, labels, predict_ntype, train_mask, val_mask, test_mask, evaluator
    """
    if name == 'ogbn-mag':
        return load_ogbn_mag(device, add_reverse_edge, reverse_self)
    elif name == 'acm':
        data = ACMDataset()
    elif name == 'dblp':
        data = DBLPDataset()
    elif name == 'oag-venue':
        data = OAGVenueDataset()
    else:
        raise ValueError(f'load_data: 未知数据集{name}')
    g = data[0]
    predict_ntype = data.predict_ntype
    # ACM和DBLP数据集已添加反向边
    if add_reverse_edge and name not in ('acm', 'dblp'):
        g = add_reverse_edges(g, reverse_self)
    g = g.to(device)
    features = g.nodes[predict_ntype].data['feat']
    labels = g.nodes[predict_ntype].data['label']
    train_mask = g.nodes[predict_ntype].data['train_mask'].nonzero(as_tuple=True)[0]
    val_mask = g.nodes[predict_ntype].data['val_mask'].nonzero(as_tuple=True)[0]
    test_mask = g.nodes[predict_ntype].data['test_mask'].nonzero(as_tuple=True)[0]
    return data, g, features, labels, predict_ntype, train_mask, val_mask, test_mask, None


def load_ogbn_mag(device, add_reverse_edge, reverse_self):
    """加载ogbn-mag数据集

    :param device: torch.device 将图和数据移动到指定的设备上，默认为CPU
    :param add_reverse_edge: bool 是否添加反向边
    :param reverse_self: bool 起点和终点类型相同时是否添加反向边
    :return: dataset, g, features, labels, predict_ntype, train_mask, val_mask, test_mask, evaluator
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
    return data, g, features, labels, 'paper', train_idx, val_idx, test_idx, evaluator


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
    for ntype in g.ntypes:
        new_g.nodes[ntype].data.update(g.nodes[ntype].data)
    for etype in g.canonical_etypes:
        new_g.edges[etype].data.update(g.edges[etype].data)
    return new_g


def one_hot_node_feat(g):
    for ntype in g.ntypes:
        if 'feat' not in g.nodes[ntype].data:
            g.nodes[ntype].data['feat'] = torch.eye(g.num_nodes(ntype), device=g.device)


def average_node_feat(g):
    """ogbn-mag数据集没有输入特征的顶点取邻居平均"""
    message_func, reduce_func = fn.copy_u('feat', 'm'), fn.mean('m', 'feat')
    g.multi_update_all({
        'writes_rev': (message_func, reduce_func),
        'has_topic': (message_func, reduce_func)
    }, 'sum')
    g.multi_update_all({'affiliated_with': (message_func, reduce_func)}, 'sum')


def load_pretrained_node_embed(g, node_embed_path, concat=False):
    """为没有输入特征的顶点加载预训练的顶点特征

    :param g: DGLGraph 异构图
    :param node_embed_path: str 预训练的word2vec模型路径
    :param concat: bool, optional 如果为True则将预训练特征与原输入特征拼接
    """
    model = Word2Vec.load(node_embed_path)
    for ntype in g.ntypes:
        embed = torch.from_numpy(model.wv[[f'{ntype}_{i}' for i in range(g.num_nodes(ntype))]]) \
            .to(g.device)
        if 'feat' in g.nodes[ntype].data:
            if concat:
                g.nodes[ntype].data['feat'] = torch.cat([g.nodes[ntype].data['feat'], embed], dim=1)
        else:
            g.nodes[ntype].data['feat'] = embed


def add_node_feat(g, method, node_embed_path=None, concat=False):
    """为没有输入特征的顶点添加输入特征

    :param g: DGLGraph 异构图
    :param method: str one-hot, average（仅用于ogbn-mag数据集）, pretrained
    :param node_embed_path: str 预训练的word2vec模型路径
    :param concat: bool, optional 如果为True则将预训练特征与原输入特征拼接
    """
    if method == 'one-hot':
        one_hot_node_feat(g)
    elif method == 'average':
        average_node_feat(g)
    elif method == 'pretrained':
        load_pretrained_node_embed(g, node_embed_path, concat)
    else:
        raise ValueError(f'add_node_feat: 未知方法{method}')
