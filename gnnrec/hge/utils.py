import random

import dgl
import numpy as np
import torch
from dgl.utils import set_new_frames, extract_node_subframes


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def add_reverse_edges(g):
    data = {}
    for stype, etype, dtype in g.canonical_etypes:
        u, v = g.edges(etype=(stype, etype, dtype))
        data[(stype, etype, dtype)] = u, v
        if stype != dtype:
            data[(dtype, etype + '_rev', stype)] = v, u
    new_g = dgl.heterograph(data, {ntype: g.num_nodes(ntype) for ntype in g.ntypes})
    set_new_frames(new_g, node_frames=extract_node_subframes(g, None))
    return new_g
