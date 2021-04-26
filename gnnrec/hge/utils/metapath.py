def metapath_adj(g, metapath):
    """返回给定元路径连接的起点和终点类型的邻接矩阵。

    :param g: DGLGraph
    :param metapath: List[str or (str, str, str)] 元路径，边类型列表
    :return: scipy.sparse.csr_matrix
    """
    adj = 1
    for etype in metapath:
        adj *= g.adj(etype=etype, scipy_fmt='csr')
    return adj
